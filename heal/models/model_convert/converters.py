import csv
import logging
import os
import weakref
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    import horizon_plugin_pytorch as horizon
except ImportError:
    horizon = None

from hat_engine.models.model_convert.converters import BaseConverter
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils import qconfig_manager
from hat_engine.utils.checkpoint import load_state_dict
from hat_engine.utils.global_var import set_value
from hat_engine.utils.logger import MSGColor, format_msg

__all__ = [
    "Float2CalibrationV2",
    "Float2QATV2",
    "SetQuantRoundingMode",
    "SetGradientCheckpointing",
    "CastModulesToTorchDType",
    "FreezeNamedModules",
    "ModuleConverterWrapper",
    "LoadHFSafetensors",
    "LinearMinMaxStat",
]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class Float2CalibrationV2(BaseConverter):
    """Define the process of convert float model to calibration model.

    Args:
        convert_mode: convert mechanism, can be choosen from ('eager',
            'symbolic', 'jit', 'jit-strip').
        hybrid: only used when convert_mode == 'symbolic', please refer to the
            doc of `horizon.quantization.prepare_qat_fx` for more info.
        hybrid_dict: only used when convert_mode == 'symbolic', please refer to
            the doc of `horizon.quantization.prepare_qat_fx` for more info.
        optimize_graph: whether to do some process on origin model for special
            purpose. Currently only support using torch.fx to fix cat input
            scale(only used on Bernoulli).
        qconfig_setter: set qconfig automatically. Value is an qconfig setter
            in horizon_plugin_pytorch.quantization.qconfig_template.
        example_inputs: example inputs for tracing graph. When using
            'jit'/'jit-strip' convert_mode or template qconfig setter, one of
            example_inputs and example_data_loader should be provided.
        example_data_loader: example data loader to get example inputs for
            tracing graph.  When using 'jit'/'jit-strip' convert_mode or
            template qconfig setter, one of example_inputs and
            example_data_loader should be provided.
        batch_transforms: batch transforms on example data loader.
    """

    def __init__(
        self,
        convert_mode="eager",
        hybrid=False,
        hybrid_dict=None,
        optimize_graph=False,
        qconfig_setter=None,
        example_inputs=None,
        example_data_loader=None,
        batch_transforms=None,
    ):
        super(Float2CalibrationV2, self).__init__(convert_mode)
        self.hybrid = hybrid
        self.hybrid_dict = hybrid_dict
        self.optimize_graph = optimize_graph
        self.qconfig_setter = qconfig_setter
        self.example_inputs = example_inputs
        self.example_data_loader = example_data_loader
        self.transforms = batch_transforms

        assert not (
            example_inputs is not None and example_data_loader is not None
        ), (
            "'example_inputs' and 'example_data_loader' can't be provided at"
            " the same time."
        )
        if self.example_data_loader is not None:
            self.example_inputs = next(iter(self.example_data_loader))
        if self.transforms is not None:
            self.example_inputs = self.transforms(self.example_inputs)

        self.use_unified_prepare = (
            hasattr(horizon.quantization, "prepare")
            and not hybrid
            and hybrid_dict is None
            and not optimize_graph
            and convert_mode in ("jit", "jit-strip")
        )
        self.use_qconfig_template = (
            self.qconfig_setter is not None and self.example_inputs is not None
        )

    def __call__(self, model):
        """Convert float model to calibration-ready model.

        Args:
            model: Input float model.

        Returns:
            Converted calibration model.
        """
        qconfig_manager.set_qconfig_mode(
            qconfig_manager.QconfigMode.CALIBRATION
        )
        if not self.use_qconfig_template:
            model.qconfig = qconfig_manager.get_default_qconfig()
            if hasattr(model, "set_qconfig"):
                model.set_qconfig()
            else:
                raise RuntimeError(
                    "use qconfig template or set `set_qconfig()` in model"
                )

        model.eval()
        set_value("preparing_qat", True)
        if self.use_unified_prepare:
            method = None
            for i in horizon.quantization.PrepareMethod:
                if i.value == self.convert_mode:
                    method = i
            model = horizon.quantization.prepare(
                model,
                example_inputs=self.example_inputs,
                qconfig_setter=self.qconfig_setter,
                method=method,
            )
        elif self.convert_mode == "eager":
            horizon.quantization.prepare_qat(
                model,
                inplace=True,
                optimize_graph=self.optimize_graph,
            )
        else:
            model = horizon.quantization.prepare_qat_fx(
                model,
                hybrid=self.hybrid,
                hybrid_dict=self.hybrid_dict,
            )
        logger.info(
            format_msg(
                "Successfully convert float model to calibration model.",
                MSGColor.GREEN,
            )
        )
        set_value("preparing_qat", False)
        horizon.quantization.set_fake_quantize(
            model, horizon.quantization.FakeQuantState.CALIBRATION
        )
        return model


@OBJECT_REGISTRY.register
class Float2QATV2(BaseConverter):
    """Define the process of convert float model to qat model.

    Args:
        convert_mode: convert mechanism, can be choosen from ('eager',
            'symbolic', 'jit', 'jit-strip').
        hybrid: only used when convert_mode == 'symbolic', please refer to the
            doc of `horizon.quantization.prepare_qat_fx` for more info.
        hybrid_dict: only used when convert_mode == 'symbolic', please refer to
            the doc of `horizon.quantization.prepare_qat_fx` for more info.
        optimize_graph: whether to do some process on origin model for special
            purpose. Currently only support using torch.fx to fix cat input
            scale(only used on Bernoulli).
        qconfig_setter: set qconfig automatically. Value is an qconfig setter
            in horizon_plugin_pytorch.quantization.qconfig_template.
        example_inputs: example inputs for tracing graph. When using
            'jit'/'jit-strip' convert_mode or template qconfig setter, one of
            example_inputs and example_data_loader should be provided.
        example_data_loader: example data loader to get example inputs for
            tracing graph. When using 'jit'/'jit-strip' convert_mode or
            template qconfig setter, one of example_inputs and
            example_data_loader should be provided.
        batch_transforms: batch transforms on example data loader.
        state: model state when tracing.can be choosen from ('train', 'val').
    """

    def __init__(
        self,
        convert_mode="eager",
        hybrid=False,
        hybrid_dict=None,
        optimize_graph=False,
        qconfig_setter=None,
        example_inputs=None,
        example_data_loader=None,
        batch_transforms=None,
        state="val",
    ):
        super(Float2QATV2, self).__init__(convert_mode)
        self.hybrid = hybrid
        self.hybrid_dict = hybrid_dict
        self.optimize_graph = optimize_graph
        self.qconfig_setter = qconfig_setter
        self.example_inputs = example_inputs
        self.example_data_loader = example_data_loader
        self.transforms = batch_transforms
        self.state = state

        assert not (
            example_inputs is not None and example_data_loader is not None
        ), (
            "'example_inputs' and 'example_data_loader' can't be provided at"
            " the same time."
        )
        if self.example_data_loader is not None:
            self.example_inputs = next(iter(self.example_data_loader))
        if self.transforms is not None:
            self.example_inputs = self.transforms(self.example_inputs)

        self.use_unified_prepare = (
            hasattr(horizon.quantization, "prepare")
            and not hybrid
            and hybrid_dict is None
            and not optimize_graph
            and convert_mode in ("jit", "jit-strip")
        )
        self.use_qconfig_template = (
            self.qconfig_setter is not None and self.example_inputs is not None
        )
        self.convert_mode = convert_mode

    def __call__(self, model):
        """Convert float model to QAT-ready model.

        Args:
            model: Input float model.

        Returns:
            Converted QAT model.
        """
        qconfig_manager.set_qconfig_mode(qconfig_manager.QconfigMode.QAT)
        if not self.use_qconfig_template:
            model.qconfig = qconfig_manager.get_default_qconfig()
            if hasattr(model, "set_qconfig"):
                model.set_qconfig()
            else:
                raise RuntimeError(
                    "use qconfig template or set `set_qconfig()` in model"
                )

        set_value("preparing_qat", True)
        if self.use_unified_prepare:
            method = None
            for i in horizon.quantization.PrepareMethod:
                if i.value == self.convert_mode:
                    method = i
            if self.state == "val":
                # set eval before prepare to get eval graph.
                origin_training_state = model.training
                model.eval()
            model = horizon.quantization.prepare(
                model,
                example_inputs=self.example_inputs,
                qconfig_setter=self.qconfig_setter,
                method=method,
            )
            if self.state == "val":
                model.train(origin_training_state)

        elif self.convert_mode == "eager":
            horizon.quantization.prepare_qat(
                model,
                inplace=True,
                optimize_graph=self.optimize_graph,
            )
        else:
            model = horizon.quantization.prepare_qat_fx(
                model,
                hybrid=self.hybrid,
                hybrid_dict=self.hybrid_dict,
            )
        set_value("preparing_qat", False)
        logger.info(
            format_msg(
                "Successfully convert float model to qat model.",
                MSGColor.GREEN,
            )
        )
        horizon.quantization.set_fake_quantize(
            model, horizon.quantization.FakeQuantState.QAT
        )
        return model


@OBJECT_REGISTRY.register
class SetQuantRoundingMode(BaseConverter):
    """Converter that sets global quant rounding mode."""

    def __init__(self):
        """Initialize rounding-mode converter."""
        super(SetQuantRoundingMode, self).__init__()

    def __call__(self, model):
        """Set quant rounding mode and return input model.

        Args:
            model: Input model.

        Returns:
            Unmodified input model.
        """
        from horizon_plugin_pytorch.utils.global_quant_round_mode import (
            QuantRoundMode,
        )

        QuantRoundMode.set(QuantRoundMode.HALF_TO_EVEN)
        return model


@OBJECT_REGISTRY.register
class SetGradientCheckpointing(BaseConverter):
    """Enable/disable gradient checkpointing for supported sub-models.

    This converter is meant to replace driver `__init__` side-effects.
    It enables/disables gradient checkpointing by calling
    `_set_gradient_checkpointing(enable=...)` on supported sub-modules.
    """

    def __init__(
        self,
        module_names: Optional[List[str]] = None,
        enable: Optional[bool] = None,
    ):
        """Initialize the gradient checkpointing converter.

        Args:
            module_names: If set, only apply to these module names.
            enable: if set, force enable/disable. If None, use
                `model.grad_checkpoint` when available, otherwise default True.
        """
        super(SetGradientCheckpointing, self).__init__()
        self._module_names = list(module_names) if module_names else []

    def _apply_to_one(self, m) -> None:
        if hasattr(m, "_set_gradient_checkpointing"):
            m._set_gradient_checkpointing(True)

    def __call__(self, model):
        for name, module in model.named_modules():
            if name in self._module_names:
                self._apply_to_one(module)
        return model


@OBJECT_REGISTRY.register
class CastModulesToTorchDType(BaseConverter):
    """Cast VLM sub-modules into a target torch dtype.

    Designed to replace the side-effect casting in `VLMDriver.__init__`
    (see `vit/scene_aggregator/language_model`).
    """

    def __init__(
        self,
        module_names: Optional[List[str]] = None,
        dtype: Optional[str] = None,
    ):
        """Initialize the dtype casting converter.

        Args:
            module_names: Only apply to these child module names.
            dtype: Optional explicit dtype string
                (e.g. "float16", "bfloat16").
            dtype_attr: attribute name on the target module (usually VLMDriver)
                that stores torch dtype (e.g. `torch_dtype`).
                If `dtype` is None,
                tries to read dtype from this attribute; otherwise infers from
                first submodule parameter.
            module_attr_names: which attributes to cast on the VLM backbone.
        """
        super(CastModulesToTorchDType, self).__init__()
        self._module_names = list(module_names) if module_names else []
        self.dtype = dtype

    def _resolve_dtype(self):
        """Resolve `self.dtype` into a torch dtype.

        Return None to skip casting.
        """
        if self.dtype is None:
            return None

        import torch

        if isinstance(self.dtype, torch.dtype):
            return self.dtype

        if not isinstance(self.dtype, str):
            return None

        dtype_str = self.dtype.strip().lower()
        if dtype_str.startswith("torch."):
            dtype_str = dtype_str.split(".", 1)[1]

        # Tolerate common aliases used in configs.
        alias = {
            "half": "float16",
            "fp16": "float16",
            "bf16": "bfloat16",
            "fp32": "float32",
            "float": "float32",
        }
        dtype_str = alias.get(dtype_str, dtype_str)

        dt = getattr(torch, dtype_str, None)
        return dt

    def _apply_to_one(self, m, dtype) -> None:
        if dtype is None:
            return
        try:
            m.to(dtype)
        except Exception as e:
            logger.warning(
                format_msg(
                    "CastModulesToTorchDType: "
                    "failed to cast module to %s (%s)",
                    dtype,
                    e,
                )
            )

    def __call__(self, model):
        dtype = self._resolve_dtype()
        for name, module in model.named_children():
            if name in self._module_names:
                self._apply_to_one(module, dtype)
        return model


@OBJECT_REGISTRY.register
class FreezeNamedModules(BaseConverter):
    """Freeze submodules by ``named_modules`` name in a convert pipeline.

    Args:
        module_names: Names from ``model.named_modules()``.
        only_batchnorm: If True, only ``eval()``; keep ``requires_grad``.
    """

    def __init__(
        self,
        module_names: Optional[List[str]] = None,
        only_batchnorm: bool = False,
    ):
        super().__init__()
        self._module_names = list(module_names) if module_names else []
        self.only_batchnorm = only_batchnorm

    def _apply_to_one(self, m) -> None:
        m.eval()
        if self.only_batchnorm:
            return
        for param in m.parameters():
            param.requires_grad = False

    def __call__(self, model):
        target = getattr(model, "module", model)
        for name, module in target.named_modules():
            if name in self._module_names:
                self._apply_to_one(module)
        return model


@OBJECT_REGISTRY.register
class ModuleConverterWrapper:
    """Run converters on named child modules, then write them back.

    Mirrors HAT ``ModuleConverterWrapper`` so VLA configs can load
    ``vlm_backbone`` / ``action_head`` checkpoints independently.

    ``module_name`` may be a direct child (``vit``) or a dotted path
    (``vlm_backbone.vit``).
    """

    def __init__(self, module_converter_dict):
        self.module_converter_dict = module_converter_dict

    @staticmethod
    def _set_module(root, module_name: str, module):
        if "." not in module_name:
            setattr(root, module_name, module)
            return
        parent_name, child_name = module_name.rsplit(".", 1)
        named = dict(root.named_modules())
        parent = named.get(parent_name)
        if parent is None:
            raise KeyError(
                f"ModuleConverterWrapper: parent '{parent_name}' "
                f"not found on {type(root).__name__}"
            )
        setattr(parent, child_name, module)

    def __call__(self, model):
        named = dict(model.named_modules())
        for module_name, converters in self.module_converter_dict.items():
            module = named.get(module_name, None)
            if module is None:
                raise KeyError(
                    f"ModuleConverterWrapper: module '{module_name}' "
                    f"not found on {type(model).__name__}"
                )
            for converter in converters:
                module = converter(module)
            self._set_module(model, module_name, module)
        return model


@OBJECT_REGISTRY.register
class LoadHFSafetensors(BaseConverter):
    """Load a HuggingFace safetensors directory into a submodule.

    ``load_func`` maps a checkpoint dir to a state_dict. Typical values:

    * ``_load_vision_weights`` (Qwen3.5 ViT) + ``key_prefix="vision_model."``
    * ``_load_llm_weights`` (InternVL / Qwen3 LLM)
    """

    def __init__(
        self,
        checkpoint_path: str,
        load_func: Callable,
        key_prefix: str = "",
        allow_miss: bool = True,
        ignore_extra: bool = True,
        verbose: bool = True,
    ):
        super().__init__()
        if not callable(load_func):
            raise TypeError(
                "LoadHFSafetensors.load_func must be callable, "
                f"got {type(load_func).__name__}"
            )
        self.checkpoint_path = checkpoint_path
        self.load_func = load_func
        self.key_prefix = key_prefix
        self.allow_miss = allow_miss
        self.ignore_extra = ignore_extra
        self.verbose = verbose

    def __call__(self, model):
        if self.checkpoint_path is None:
            logger.info(
                format_msg(
                    "LoadHFSafetensors: checkpoint_path is None, skip.",
                    MSGColor.RED,
                )
            )
            return model
        raw = self.load_func(self.checkpoint_path)
        if self.key_prefix:
            state_dict = {self.key_prefix + k: v for k, v in raw.items()}
        else:
            state_dict = raw
        model = load_state_dict(
            model,
            path_or_dict={"state_dict": state_dict},
            allow_miss=self.allow_miss,
            ignore_extra=self.ignore_extra,
            verbose=self.verbose,
        )
        logger.info(
            format_msg(
                "LoadHFSafetensors (%s): loaded %d tensors from %s"
                % (
                    getattr(self.load_func, "__name__", repr(self.load_func)),
                    len(state_dict),
                    self.checkpoint_path,
                ),
                MSGColor.GREEN,
            )
        )
        return model


@OBJECT_REGISTRY.register
class LinearMinMaxStat(BaseConverter):
    """Record per-channel min/max stats on every ``nn.Linear``.

    Attach forward hooks to record input/weight channel ranges during
    forward passes. Stats live as buffers on each hooked Linear, so they
    flow into ``state_dict()`` and any saved checkpoint.

    Buffer names on each Linear:

    - ``stat_weight_max_ch``     ``[out_feats]`` per-out-channel max along
      the ``in_feats`` axis of ``weight``
    - ``stat_weight_min_ch``     ``[out_feats]`` per-out-channel min
    - ``stat_input_max_ch``      ``[in_feats]`` running max over all forwards
    - ``stat_input_min_ch``      ``[in_feats]`` running min
    - ``stat_input_global_max``  scalar running max
    - ``stat_input_global_min``  scalar running min
    - ``stat_act_absmax_ch``     ``[in_feats]`` running per-channel abs-max
    - ``stat_act_outlier``       scalar ``amax_ch.max() / amax_ch.median()``
    - ``stat_act_pt_gap``        scalar ``log2(amax_t / amax_ch.median())``

    Hard-quant score (derived)::

        hard_score = outlier * (1 + pt_gap)

    Additionally, a pairing detector watches normalization modules (any Module
    whose class name contains "Norm") and, at Linear hook time, checks whether
    the Linear's input is the most recent Norm's output (compared by
    ``data_ptr``). On a hit, the Linear is tagged with
    ``stat_predecessor_norm = weakref.ref(norm)``, which downstream passes
    (e.g. ``ApplySmoothQuant``) can use to fold the scale into the norm's
    affine parameters instead of wrapping the Linear.
    """

    STAT_BUFFER_NAMES = (
        "stat_weight_max_ch",
        "stat_weight_min_ch",
        "stat_input_max_ch",
        "stat_input_min_ch",
        "stat_input_global_max",
        "stat_input_global_min",
        "stat_act_absmax_ch",
        "stat_act_outlier",
        "stat_act_pt_gap",
    )

    def __init__(
        self,
        module_filter: Optional[List[str]] = None,
    ):
        """Init stat converter.

        Args:
            module_filter: optional list of substrings; only Linear modules
                whose qualified name contains any of these are hooked.
                ``None`` hooks every ``nn.Linear``.
        """
        super(LinearMinMaxStat, self).__init__()
        self.module_filter = module_filter
        self._hooks: List = []
        # LIFO stack of (data_ptr, norm_module) for norms that fired since
        # the last Linear hook consumed them.
        self._recent_norm_ptrs: List = []

    def _match(self, name: str) -> bool:
        if not self.module_filter:
            return True
        return any(sub in name for sub in self.module_filter)

    @staticmethod
    def _register_stat_buffers(mod: nn.Linear) -> None:
        w = mod.weight
        out_feats, in_feats = w.shape
        dtype = torch.float32
        device = w.device
        mod.register_buffer(
            "stat_weight_max_ch",
            torch.full(
                (out_feats,), float("-inf"), dtype=dtype, device=device
            ),
        )
        mod.register_buffer(
            "stat_weight_min_ch",
            torch.full((out_feats,), float("inf"), dtype=dtype, device=device),
        )
        mod.register_buffer(
            "stat_input_max_ch",
            torch.full((in_feats,), float("-inf"), dtype=dtype, device=device),
        )
        mod.register_buffer(
            "stat_input_min_ch",
            torch.full((in_feats,), float("inf"), dtype=dtype, device=device),
        )
        mod.register_buffer(
            "stat_input_global_max",
            torch.tensor(float("-inf"), dtype=dtype, device=device),
        )
        mod.register_buffer(
            "stat_input_global_min",
            torch.tensor(float("inf"), dtype=dtype, device=device),
        )
        mod.register_buffer(
            "stat_act_absmax_ch",
            torch.zeros((in_feats,), dtype=dtype, device=device),
        )
        mod.register_buffer(
            "stat_act_outlier",
            torch.tensor(0.0, dtype=dtype, device=device),
        )
        mod.register_buffer(
            "stat_act_pt_gap",
            torch.tensor(0.0, dtype=dtype, device=device),
        )
        # Snapshot weight stats once; weight is fixed in eval.
        w_float = w.detach().float()
        mod.stat_weight_max_ch.copy_(w_float.amax(dim=1))
        mod.stat_weight_min_ch.copy_(w_float.amin(dim=1))

    @staticmethod
    def _refresh_act_hard_buffers(mod: nn.Linear) -> None:
        """Recompute outlier / pt_gap from running absmax channels."""
        amax_ch = mod.stat_act_absmax_ch.float()
        # Uninitialized (no forward yet): leave zeros.
        if not torch.isfinite(amax_ch).any() or float(amax_ch.max()) <= 0:
            return
        med = amax_ch.median().clamp(min=1e-12)
        mx = amax_ch.max().clamp(min=1e-12)
        amax_t = torch.maximum(
            mod.stat_input_global_max.abs(), mod.stat_input_global_min.abs()
        ).clamp(min=1e-12)
        mod.stat_act_outlier.fill_(float(mx / med))
        mod.stat_act_pt_gap.fill_(float(torch.log2(amax_t / med)))

    @staticmethod
    def hard_score(mod: nn.Module) -> float:
        """Activation hard-quant score: ``outlier * (1 + pt_gap)``."""
        outlier = getattr(mod, "stat_act_outlier", None)
        pt_gap = getattr(mod, "stat_act_pt_gap", None)
        if outlier is None or pt_gap is None:
            return 0.0
        return float(outlier) * (1.0 + max(float(pt_gap), 0.0))

    @classmethod
    def collect_hard_act_rows(cls, model: nn.Module) -> List[dict]:
        """Return per-Linear hard-quant rows sorted by score descending."""
        rows = []
        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            if not hasattr(mod, "stat_act_outlier"):
                continue
            # Prefer absmax buffer; fall back to min/max if absmax empty.
            if (
                hasattr(mod, "stat_act_absmax_ch")
                and float(mod.stat_act_absmax_ch.max()) <= 0
                and hasattr(mod, "stat_input_max_ch")
            ):
                with torch.no_grad():
                    absmax = torch.maximum(
                        mod.stat_input_max_ch.abs(),
                        mod.stat_input_min_ch.abs(),
                    )
                    mod.stat_act_absmax_ch.copy_(absmax)
                    cls._refresh_act_hard_buffers(mod)
            score = cls.hard_score(mod)
            rows.append(
                {
                    "name": name,
                    "hard_score": score,
                    "outlier": float(mod.stat_act_outlier),
                    "pt_gap": float(mod.stat_act_pt_gap),
                    "amax_tensor": float(
                        torch.maximum(
                            mod.stat_input_global_max.abs(),
                            mod.stat_input_global_min.abs(),
                        )
                    ),
                }
            )
        rows.sort(key=lambda r: r["hard_score"], reverse=True)
        return rows

    @classmethod
    def dump_hard_act_report(
        cls,
        model: nn.Module,
        topk: int = 20,
        path: Optional[str] = None,
    ) -> List[dict]:
        """Log / CSV-dump hard-quant ranking; return full sorted rows."""
        rows = cls.collect_hard_act_rows(model)
        if path:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "rank",
                        "name",
                        "hard_score",
                        "outlier",
                        "pt_gap",
                        "amax_tensor",
                    ],
                )
                w.writeheader()
                for i, r in enumerate(rows, 1):
                    w.writerow({"rank": i, **r})
            logger.info(
                format_msg(
                    "LinearMinMaxStat: wrote act hard-quant report "
                    "(%d layers) -> %s" % (len(rows), path),
                    MSGColor.GREEN,
                )
            )
        show = rows[: max(topk, 0)] if topk else []
        if show:
            lines = [
                "LinearMinMaxStat hard-act top-%d (score=outlier*(1+pt_gap)):"
                % len(show)
            ]
            for i, r in enumerate(show, 1):
                lines.append(
                    "  #%d score=%.3f outlier=%.2f pt_gap=%.2f  %s"
                    % (
                        i,
                        r["hard_score"],
                        r["outlier"],
                        r["pt_gap"],
                        r["name"],
                    )
                )
            logger.info(format_msg("\n".join(lines), MSGColor.GREEN))
        return rows

    @classmethod
    def topk_hard_act_names(
        cls, model: nn.Module, topk: int
    ) -> Tuple[List[str], List[dict]]:
        """Return ``(names, rows)`` for the top-``topk`` hardest Linears."""
        rows = cls.collect_hard_act_rows(model)
        if topk <= 0:
            return [], rows
        top = rows[:topk]
        return [r["name"] for r in top], rows

    def _norm_hook(self, mod: nn.Module, _inputs, output) -> None:
        if not isinstance(output, torch.Tensor):
            return
        # Each norm forward produces a fresh log entry. We keep the tie to
        # the actual output **tensor** (not just data_ptr) because the CUDA
        # caching allocator can hand the same data_ptr to an unrelated tensor
        # in a later forward — identity-check via ``is`` is reliable, ptr is
        # not. Tagging adjusts to ptr only when the Linear can also confirm
        # the candidate matches in shape, resolved inside ``_linear_hook``.
        self._recent_norm_ptrs.append(
            (
                output.data_ptr() if output.is_contiguous() else None,
                output.shape[-1],
                weakref.ref(mod),
                weakref.ref(output),
            )
        )

    def _linear_hook(self, mod: nn.Linear, inputs, _output) -> None:
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            return
        x = inputs[0]
        in_feats = mod.in_features

        # Match via tensor identity (``is``), NOT data_ptr — caching allocator
        # reuse can give unrelated tensors the same data_ptr over time. We
        # walk backwards so we prefer the most recent norm whose output is
        # literally this tensor.
        matched_norm = None
        for _ptr, last_dim, norm_ref, out_ref in reversed(
            self._recent_norm_ptrs
        ):
            out = out_ref() if out_ref is not None else None
            if out is not None and out is x and last_dim == in_feats:
                matched_norm = norm_ref
                break
        if matched_norm is not None and not hasattr(
            mod, "stat_predecessor_norm"
        ):
            mod.stat_predecessor_norm = matched_norm

        x_flat = x.float().reshape(-1, x.shape[-1])
        batch_max_ch = x_flat.amax(dim=0)
        batch_min_ch = x_flat.amin(dim=0)
        with torch.no_grad():
            torch.maximum(
                mod.stat_input_max_ch, batch_max_ch, out=mod.stat_input_max_ch
            )
            torch.minimum(
                mod.stat_input_min_ch, batch_min_ch, out=mod.stat_input_min_ch
            )
            mod.stat_input_global_max.clamp_(min=x.amax().item())
            mod.stat_input_global_min.clamp_(max=x.amin().item())
            batch_absmax = torch.maximum(
                batch_max_ch.abs(), batch_min_ch.abs()
            )
            torch.maximum(
                mod.stat_act_absmax_ch,
                batch_absmax,
                out=mod.stat_act_absmax_ch,
            )
            self._refresh_act_hard_buffers(mod)

    @staticmethod
    def _is_norm(mod: nn.Module) -> bool:
        cls_name = type(mod).__name__
        if "Norm" not in cls_name:
            return False
        # LayerNorm(elementwise_affine=False) has no affine; skip it.
        if getattr(mod, "weight", None) is None:
            return False
        return True

    def sync_stats(self, model) -> int:
        """Cross-rank max/min reduce of stat buffers (DDP-safe, no-op single).

        Each buffer is reduced elementwise: ``*_max_ch`` via ``MAX``,
        ``*_min_ch`` via ``MIN``. Weight stats do not need reduction because
        DDP already keeps weights in sync. Returns number of buffers reduced.

        Call after calibration data has been consumed (e.g. in a callback's
        ``on_loop_end`` before ``ApplySmoothQuant``) so every rank computes
        SmoothQuant scales from the same global min/max. Idempotent.
        """
        if not (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            return 0
        num_reduced = 0
        for _name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            for buf_name, op in (
                ("stat_input_max_ch", torch.distributed.ReduceOp.MAX),
                ("stat_input_min_ch", torch.distributed.ReduceOp.MIN),
                ("stat_input_global_max", torch.distributed.ReduceOp.MAX),
                ("stat_input_global_min", torch.distributed.ReduceOp.MIN),
                ("stat_act_absmax_ch", torch.distributed.ReduceOp.MAX),
                ("stat_act_outlier", torch.distributed.ReduceOp.MAX),
                ("stat_act_pt_gap", torch.distributed.ReduceOp.MAX),
            ):
                buf = getattr(mod, buf_name, None)
                if buf is None:
                    continue
                torch.distributed.all_reduce(buf, op=op)
                num_reduced += 1
            # Recompute derived scalars from the globally-synced absmax /
            # global extrema so median-based outlier/pt_gap stay consistent.
            if hasattr(mod, "stat_act_absmax_ch"):
                self._refresh_act_hard_buffers(mod)
        return num_reduced

    def __call__(self, model):
        num_linear = 0
        num_norm = 0
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                if not self._match(name):
                    continue
                if mod.weight is None or mod.weight.ndim != 2:
                    continue
                self._register_stat_buffers(mod)
                h = mod.register_forward_hook(self._linear_hook)
                self._hooks.append(h)
                num_linear += 1
            elif self._is_norm(mod):
                h = mod.register_forward_hook(self._norm_hook)
                self._hooks.append(h)
                num_norm += 1

        logger.info(
            format_msg(
                "LinearMinMaxStat: hooked %d Linear + %d Norm."
                % (num_linear, num_norm),
                MSGColor.GREEN,
            )
        )
        return model

    def remove_hooks(self, model) -> None:
        """Detach all forward hooks; safe to call multiple times."""
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks = []
        self._recent_norm_ptrs = []
