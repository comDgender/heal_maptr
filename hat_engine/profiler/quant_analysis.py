# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import copy
import logging
import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import torch

from hat_engine.data.transforms.compose_transform import Compose
from hat_engine.registry import OBJECT_REGISTRY, Registry
from hat_engine.utils.package_helper import (
    check_packages_available,
    require_packages,
)

try:
    from horizon_plugin_profiler import (  # noqa E402
        QuantAnalysis as HorizonQuantAnalysis,
    )
except ImportError:
    HorizonQuantAnalysis = None

logger = logging.getLogger(__name__)

QUANTANALYSIS_REGISTER = Registry("QUANT_ANALYSIS_REGISTRY")


class _MappedIterable:
    """Re-iterable wrapper: apply ``transform`` per batch without materializing
    the full dataloader (ImageNet / nuScenes val would be too slow)."""

    def __init__(self, data, transform):
        self._data = data
        self._transform = transform

    def __iter__(self):
        for item in self._data:
            yield self._transform(item)

    def __len__(self):
        return len(self._data)


def _replace_none(obj, fill=0.0):
    """Replace nested None so profiler dump can ``sum(values)``."""
    if obj is None:
        return fill
    if isinstance(obj, dict):
        return {k: _replace_none(v, fill) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_replace_none(v, fill) for v in obj)
    return obj


def _patch_profiler_metric_dump() -> None:
    """Avoid dump crash when some outputs have None metrics.

    ``horizon_plugin_profiler.find_bad_case._summarize_metric_values``
    does ``sum(values)`` and raises TypeError if any value is None.
    Detection heads and 1-step smoke often hit that path; the bad-case
    search itself already finished.
    """
    try:
        from horizon_plugin_profiler import find_bad_case as fbc
    except ImportError:
        return
    orig = getattr(fbc, "_summarize_metric_values", None)
    if orig is None or getattr(orig, "_heal_none_safe", False):
        return

    def _safe_summarize(*args, **kwargs):
        try:
            return orig(*args, **kwargs)
        except TypeError as exc:
            if "NoneType" not in str(exc):
                raise
            logger.warning(
                "profiler bad-case dump hit None metrics (%s); "
                "treating missing values as 0.",
                exc,
            )
            args = tuple(_replace_none(a) for a in args)
            kwargs = {k: _replace_none(v) for k, v in kwargs.items()}
            return orig(*args, **kwargs)

    _safe_summarize._heal_none_safe = True  # type: ignore[attr-defined]
    fbc._summarize_metric_values = _safe_summarize


def _patch_get_bad_case_log() -> None:
    """Log which output's BadCase triggers the worse_than crash.

    Patches ``BadCaseReport.get_bad_case`` so that when ``candidate.worse_than``
    raises (None vs float), we know which output_name / metric was being
    compared.
    """
    try:
        from horizon_plugin_profiler import find_bad_case as fbc
    except ImportError:
        return
    cls = getattr(fbc, "BadCaseReport", None)
    if cls is None:
        return
    orig = cls.get_bad_case
    if getattr(orig, "_heal_logged", False):
        return

    def _logged_get_bad_case(self, *args, **kwargs):
        try:
            return orig(self, *args, **kwargs)
        except (TypeError, RuntimeError) as exc:
            # Some outputs produce BadCase.error=None (e.g. non-Tensor
            # children inside one2many_outs). Skip those candidates and
            # let other outputs decide the bad case.
            if "NoneType" not in str(exc) and "bad case" not in str(exc):
                raise
            metric = kwargs.get("metric", args[0] if args else None)
            index_kw = kwargs.get("index")
            if index_kw is not None or metric is None:
                raise
            skipped = []
            chosen = None
            for out_name, output_info in (
                self.output_to_badcase_info.items()
            ):
                if metric.name not in output_info.topk_bad_cases:
                    continue
                cand = output_info.get_worst_bad_case(metric.name)
                if cand is None:
                    continue
                if cand.error is None:
                    skipped.append((out_name, metric.name))
                    continue
                if chosen is None or cand.worse_than(chosen):
                    chosen = cand
            for s in skipped:
                logger.warning(
                    "get_bad_case skipping %r (error=None)", s
                )
            if chosen is None:
                raise
            logger.info(
                "get_bad_case picked data_idx=%s error=%s "
                "(skipped %d None-error outputs)",
                chosen.data_idx,
                chosen.error,
                len(skipped),
            )
            kwargs = dict(kwargs)
            kwargs.pop("metric", None)
            kwargs["index"] = chosen.data_idx
            return orig(self, **kwargs)

    _logged_get_bad_case._heal_logged = True  # type: ignore[attr-defined]
    cls.get_bad_case = _logged_get_bad_case


@OBJECT_REGISTRY.register
class QuantAnalysis:
    """Quantization models precision analysis.

    This class helps analyze quantization precision errors between two models.

    Args:
        model: origin float model
        baseline_model_convert_pipeline: baseline model convert pipeline
        analysis_model_convert_pipeline: analysis model convert pipeline
        baseline_model: baseline model. If set, will skip baseline model
            convert pipeline.
        analysis_model: analysis model. If set, will skip analysis model
            convert pipeline.
        analysis_model_type: analysis model type. Support four types:
        - "fake_quant": float model vs calibration model.
        - "pt_lut": qat/calibration model vs lut model.
        - "export": lut model vs qat bc model.
        - "convert": lut model vs quantize bc model.
        out_dir: path to save advisor and comparsion result.
        post_process: post process function which performs on model output.
        data_generator: input dataloader or a custom iterable object. It
            must generate a data each iteration. Example:
        - torch dataloader: data_generator = torch.utils.data.DataLoader()
        - a custom generator: data_generator = [x for x in [1, 2, 3]]
        batch_transforms: batch transforms
        num_steps: num of steps to find bad case
        device_id: device id
        bad_case_input: manually set bad case. If set, will skip auto find
            bad case process.
        analysis_pipeline: analysis pipeline. It is a list of analysis
            operations to do. If None, will do default analysis pipelines.
    """

    @require_packages("horizon_plugin_profiler>=2.1.6")
    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        baseline_model_convert_pipeline: Optional[Union[Dict, List]] = None,
        analysis_model_convert_pipeline: Optional[Union[Dict, List]] = None,
        baseline_model: Optional[torch.nn.Module] = None,
        analysis_model: Optional[torch.nn.Module] = None,
        analysis_model_type: str = "fake_quant",
        out_dir: str = "./quant_analysis",
        post_process: Optional[Callable] = None,
        dataloader: Optional[Iterable] = None,
        batch_transforms: Optional[List] = None,
        num_steps: Optional[int] = None,
        device_id: Optional[int] = None,
        bad_case_input: Optional[Any] = None,
        analysis_pipeline: Optional[Any] = None,
    ):
        if baseline_model is not None:
            assert baseline_model_convert_pipeline is None, (
                "baseline_model and baseline_model_convert_pipeline can't be "
                "set at the same time."
            )
            self.baseline_model = baseline_model
        else:
            assert (
                baseline_model_convert_pipeline is not None
                and model is not None
            ), (
                "baseline_model should be provided directly or converted by "
                "baseline_model_convert_pipeline."
            )
            self.baseline_model = baseline_model_convert_pipeline(
                copy.deepcopy(model)
            )

        if analysis_model is not None:
            assert analysis_model_convert_pipeline is None, (
                "analysis_model and analysis_model_convert_pipeline can't be "
                "set at the same time."
            )
            self.analysis_model = analysis_model
        else:
            assert (
                analysis_model_convert_pipeline is not None
                and model is not None
            ), (
                "analysis_model should be provided directly or converted by "
                "analysis_model_convert_pipeline."
            )
            self.analysis_model = analysis_model_convert_pipeline(
                copy.deepcopy(model)
            )

        from horizon_plugin_pytorch.quantization import (
            FakeQuantState,
            set_fake_quantize,
        )


        self.baseline_model.eval()
        set_fake_quantize(self.baseline_model, FakeQuantState.VALIDATION)
        self.analysis_model.eval()
        set_fake_quantize(self.analysis_model, FakeQuantState.VALIDATION)

        self.device_id = device_id
        self.dataloader = dataloader
        self.bad_case_input = bad_case_input
        self.analysis_pipeline = analysis_pipeline
        self.num_steps = num_steps
        self.out_dir = out_dir
        self.post_process = post_process

        if batch_transforms:
            if isinstance(batch_transforms, (list, tuple)):
                batch_transforms = Compose(batch_transforms)
        self.batch_transforms = batch_transforms

        default_pipeline = [
            dict(  # noqa C408
                analysis_type="AutoFindBadCase",
            ),
            dict(  # noqa C408
                analysis_type="Run",
            ),
            dict(  # noqa C408
                analysis_type="ComparePerLayer",
            ),
            dict(  # noqa C408
                analysis_type="Sensitivity",
                metric="L1",
                reverse=False,
            ),
        ]
        if self.analysis_pipeline is None:
            self.analysis_pipeline = default_pipeline
        self.analysis_pipeline = [
            self._with_analysis_step_defaults(step)
            for step in self.analysis_pipeline
        ]

        if check_packages_available(
            "horizon_plugin_profiler<=2.2.4", raise_exception=False
        ):
            self.quant_analysis = HorizonQuantAnalysis(
                baseline_model=self.baseline_model,
                analysis_model=self.analysis_model,
                analysis_model_type=analysis_model_type,
                post_process=self.post_process,
                out_dir=self.out_dir,
            )
        else:
            self.quant_analysis = HorizonQuantAnalysis(
                baseline_model=self.baseline_model,
                analysis_model=self.analysis_model,
                analysis_model_type=analysis_model_type,
                device_ids=self.device_id,
                post_process=self.post_process,
                out_dir=self.out_dir,
            )

    def _with_analysis_step_defaults(self, pipeline):
        """Fill dataloader / device / num_steps for a custom short pipeline.

        Smoke and CLI ``--opts`` can pass
        ``[{analysis_type: AutoFindBadCase}, {analysis_type: Run}]`` without
        repeating the solver's dataloader and device.
        """
        step = dict(pipeline)
        atype = step.get("analysis_type")
        if atype == "AutoFindBadCase":
            if self.batch_transforms is not None:
                data_generator = _MappedIterable(
                    self.dataloader, self.batch_transforms
                )
            else:
                data_generator = self.dataloader
            step.setdefault("data_generator", data_generator)
            step.setdefault("num_steps", self.num_steps)
            step.setdefault("metric", "L1")
            step.setdefault("device", self.device_id)
            step.setdefault("custom_metric_func", None)
            step.setdefault("custom_metric_order_seq", None)
        elif atype in ("Run", "Sensitivity"):
            step.setdefault("device", self.device_id)
        return step

    @QUANTANALYSIS_REGISTER.register
    @QUANTANALYSIS_REGISTER.alias("AutoFindBadCase")
    def _auto_find_bad_case(self, pipeline):
        if self.bad_case_input is not None:
            logger.info("Directly use bad case input. Skip auto find.")
            self.quant_analysis.set_bad_case(self.bad_case_input)
        else:
            if "device" not in pipeline:
                pipeline["device"] = self.device_id
            logger.info("Auto find bad case...")
            _patch_profiler_metric_dump()
            try:
                self.quant_analysis.auto_find_bad_case(**pipeline)
                logger.info(f"Save bad case in {self.out_dir}.")
                self.quant_analysis.save_bad_case()
            except TypeError as exc:
                # Dump still crashed (or assignment never ran). Run() needs
                # a bad case; fall back to the first batch.
                if "NoneType" not in str(exc):
                    raise
                logger.warning(
                    "auto_find_bad_case dump failed (%s); "
                    "using the first batch as bad case.",
                    exc,
                )
                self.quant_analysis.set_bad_case(self._first_analysis_batch())

    @QUANTANALYSIS_REGISTER.register
    @QUANTANALYSIS_REGISTER.alias("SetModelProfilerDir")
    def _set_model_profiler_dir(self, pipeline):
        logger.info("Set model profiler dir...")
        self.quant_analysis.set_model_profiler_dir(**pipeline)

    @QUANTANALYSIS_REGISTER.register
    @QUANTANALYSIS_REGISTER.alias("LoadBadCase")
    def _load_bad_case(self, pipeline):
        filename = (
            os.path.join(self.out_dir, "badcase.pt")
            if pipeline.get("filename", None) is None
            else pipeline["filename"]
        )
        logger.info(f"Load bad case from {filename}")
        self.quant_analysis.load_bad_case(filename)

    def _first_analysis_batch(self):
        item = next(iter(self.dataloader))
        if self.batch_transforms is not None:
            item = self.batch_transforms(item)
        return item

    @QUANTANALYSIS_REGISTER.register
    @QUANTANALYSIS_REGISTER.alias("Run")
    def _run(self, pipeline):
        logger.info("Run model...")
        _patch_get_bad_case_log()
        device = pipeline.get("device", self.device_id)
        try:
            self.quant_analysis.run(device)
        except TypeError as exc:
            # 1-step / detection heads often leave metric.error as None;
            # profiler then crashes in worse_than (error > other.error).
            if "NoneType" not in str(exc):
                raise
            logger.warning(
                "profiler run() could not pick a bad case (%s); "
                "falling back to the first batch.",
                exc,
            )
            self.quant_analysis.set_bad_case(self._first_analysis_batch())
            try:
                self.quant_analysis.run(device)
            except TypeError as exc2:
                if "NoneType" not in str(exc2):
                    raise
                logger.warning(
                    "profiler run() still failed after first-batch "
                    "fallback (%s); skipping Run step.",
                    exc2,
                )

    @QUANTANALYSIS_REGISTER.register
    @QUANTANALYSIS_REGISTER.alias("ComparePerLayer")
    def _run_and_compare_per_layer(self, pipeline):
        logger.info("Compare per layer...")
        self.quant_analysis.compare_per_layer()

    @QUANTANALYSIS_REGISTER.register
    @QUANTANALYSIS_REGISTER.alias("Sensitivity")
    def _sensitivity(self, pipeline):
        logger.info("Run sensitivity analysis...")
        if "device" not in pipeline:
            pipeline["device"] = self.device_id
        self.quant_analysis.sensitivity(**pipeline)

    @QUANTANALYSIS_REGISTER.register
    @QUANTANALYSIS_REGISTER.alias("Clean")
    def _clean(self, pipeline):
        logger.info("Clean...")
        self.quant_analysis.clean()

    def __call__(self):
        for pipeline in self.analysis_pipeline:
            pipeline_type = pipeline.pop("analysis_type", None)
            try:
                QUANTANALYSIS_REGISTER.get(pipeline_type)(self, pipeline)
            except KeyError:
                raise ValueError(f"Unknown analysis type {pipeline_type}")
            except Exception as e:
                raise e
