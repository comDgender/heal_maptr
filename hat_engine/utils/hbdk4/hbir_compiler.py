# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import logging
import os
from typing import Dict, List, Optional, Union

from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.logger import MSGColor, format_msg
from hat_engine.utils.package_helper import require_packages

try:
    import hbdk4.compiler as hbdk4_compiler
    from hbdk4.compiler.extra_apis import llm_convert  # noqa: E402
    from horizon_plugin_pytorch.quantization.hbdk4 import (
        get_hbir_input_flattener,
    )
except Exception:
    hbdk4_compiler = None
    llm_convert = None
    get_hbir_input_flattener = None

from hat_engine.utils.hbdk4.statistics_helpers import warn_cpu_ops

__all__ = ["HbirCompiler"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class HbirCompiler:
    """Compile mlir.Module and estimate performance.

    Args:
        model_path: QAT hbir module file path.
        out_path: A filesystem path to save the compiled artifact.
            Performance is only measured when it ends with ".hbm".
        march: BPU march, options are "bayes", "b25", "b25e", "b30", "b30g".
        opt: Optimization level. Defaults to 2.
        jobs: Number of threads launched during compiler optimization.
            Defaults to 4.
        max_time_per_fc:
            Set maximum time constraint (unit:us) for per funccall.
        debug: Set whether to contain debug info in HBM.
        preserve_quant_dequant:
            Whether preserve the quantize node on model input and
            dequantize node on model output.
        input_source:
            A structure of strings has the same structure with model input.
            Supported options are "ddr" and "pyramid".
            Defaults to None.
        transpose_dim:
            Optional transpose permutes for model IO nodes, e.g.
            ``dict(outputs={"global": [0, 2, 3, 1]})`` to convert NCHW
            outputs to NHWC.
    """

    @require_packages("horizon_plugin_pytorch>=1.10.3", "hbdk4")
    def __init__(
        self,
        model_path: str,
        out_path: str,
        march: str,
        opt: int,
        jobs: int,
        max_time_per_fc: float,
        cache_path: Optional[str] = None,
        cache_mode: Optional[str] = "disable",
        preserve_transpose: bool = True,
        preserve_quant_dequant: bool = False,
        remove_nodes_ioname=None,
        remove_io_op_types=("Quantize", "Dequantize"),
        input_source=None,
        transpose_dim: Optional[Dict[str, Dict[str, List[int]]]] = None,
        debug: bool = True,
        enable_vpu: bool = True,
        enable_spu: bool = True,
        balance: int = 100,
        input_no_padding: bool = False,
        output_no_padding: bool = False,
        remote_ip: Optional[str] = None,
        remote_port: int = 22,
        enable_llm_convert: bool = False,
        warn_cpu_ops: bool = False,
        rmsnorm_version: str = None,
        softmax_version: str = None,
        use_f16_quant_dequant_on_vae_always: bool = False,
        core_num: Optional[int] = None,
        enable_hpc: Optional[bool] = None,
        max_l2m_size: Optional[int] = None,
        enable_integer_conv: bool = True,
    ) -> None:
        if preserve_quant_dequant and not preserve_transpose:
            raise ValueError(
                "Cannot preserve quant dequant when dropping transpose"
            )

        self.model = hbdk4_compiler.load(model_path)
        self.out_path = out_path
        self.march = march
        self.opt = opt
        self.jobs = jobs
        self.debug = debug
        self.max_time_per_fc = max_time_per_fc

        self.preserve_transpose = preserve_transpose
        self.preserve_quant_dequant = preserve_quant_dequant
        self.remove_nodes_ioname = remove_nodes_ioname
        self.remove_io_op_types = remove_io_op_types
        self.enable_vpu = enable_vpu
        self.enable_spu = enable_spu
        self.balance = balance
        self.input_no_padding = input_no_padding
        self.output_no_padding = output_no_padding
        self.cache_path = cache_path
        self.cache_mode = cache_mode
        self.remote_ip = remote_ip
        self.remote_port = remote_port
        self.transpose_dim = transpose_dim

        self.enable_llm_convert = enable_llm_convert
        self.enable_integer_conv = enable_integer_conv
        self.warn_cpu_ops = warn_cpu_ops
        self.rmsnorm_version = rmsnorm_version
        self.softmax_version = softmax_version
        self.use_f16_quant_dequant_on_vae_always = (
            use_f16_quant_dequant_on_vae_always
        )
        self.llm_convert_kwargs = dict(
            rmsnorm_version=rmsnorm_version,
            softmax_version=softmax_version,
        )
        self.core_num = core_num
        self.enable_hpc = enable_hpc
        self.max_l2m_size = max_l2m_size

        if input_source is None:
            self.input_source = input_source
        else:
            if isinstance(input_source, str):
                input_source = input_source.split(",")
            self.input_source = get_hbir_input_flattener(self.model)(
                input_source
            )
            self.input_source = [
                source.strip() for source in self.input_source
            ]

        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)

    def set_pyramid_source(self, node, input_source, preprocess):
        mode = preprocess["type"]
        if mode == "rgb":
            mode = "yuvbt601full2rgb"
        elif mode == "bgr":
            mode = "yuvbt601full2bgr"
        elif mode == "yuv":
            mode = None
        divisor = preprocess["divisor"]
        mean = preprocess["mean"]
        std = preprocess["std"]
        node = node.insert_transpose(permutes=[0, 3, 1, 2])
        node = node.insert_image_preprocess(
            mode=mode, divisor=divisor, mean=mean, std=std
        )
        if input_source == "pyramid":
            print(f"insert pyramid for {node.name}")
            node.insert_image_convert("nv12")
        elif input_source == "resizer":
            print(f"insert resizer for {node.name}")
            node.insert_roi_resize("nv12")

    def remove_op_by_ioname(self, func, io_name=None):
        for loc in func.inputs + func.outputs:
            if not loc.is_removable[0]:
                if io_name == loc.name:
                    raise ValueError(
                        f"Failed when deleting {io_name} ,which id unremovable"
                    )
                continue
            attached_op = loc.get_attached_op[0]
            removed = None
            output_name = attached_op.outputs[0].name
            input_name = attached_op.inputs[0].name

            if io_name in [output_name, input_name]:
                removed, diagnostic = loc.remove_attached_op()
            if removed is True:
                print(f"Remove node {io_name} successfully", flush=True)
            if removed is False:
                raise ValueError(
                    f"Failed when deleting {attached_op.name} operator,"
                    f"error: {diagnostic}"
                )

    def hbir_disas(self, func):
        inputs = func.inputs
        for i, input in enumerate(inputs):
            print(
                f"input[{i}]: ",
                input.name,
                " ",
                input.type.shape,
                " ",
                input.type.dtype,
                " ",
                input.quant_info,
                flush=True,
            )
        outputs = func.outputs
        for i, output in enumerate(outputs):
            print(
                f"output[{i}]: ",
                output.name,
                " ",
                output.type.shape,
                " ",
                output.type.dtype,
                " ",
                output.quant_info,
                flush=True,
            )

    def set_input_nodes(self, qat_hbir, input_source):
        preprocess = {
            "type": "yuv",
            "divisor": 1,
            "mean": [128, 128, 128],
            "std": [128, 128, 128],
        }
        if input_source is None or (
            "pyramid" not in input_source and "resizer" not in input_source
        ):
            logger.info("no input need to be changed!")
        elif len(input_source) != len(qat_hbir[0].flatten_inputs):
            logger.info(
                f"length of input_source must be equal to the number of    \
                  model inputs, but got input_source: {len(input_source)}, \
                  model inputs: {len(qat_hbir[0].flatten_inputs)}"
            )
        else:
            func = qat_hbir[0]
            for input, source in zip(
                func.flatten_inputs[::-1], self.input_source[::-1]
            ):
                if source == "ddr":
                    continue
                if source == "pyramid":
                    if input.type.shape[0] > 1:
                        split_inputs = input.insert_split(0)
                        for split_input in reversed(split_inputs):
                            self.set_pyramid_source(
                                split_input, "pyramid", preprocess
                            )
                    else:
                        self.set_pyramid_source(input, "pyramid", preprocess)
                elif source == "resizer":
                    if input.type.shape[0] > 1:
                        split_inputs = input.insert_split(0)
                        for split_input in reversed(split_inputs):
                            self.set_pyramid_source(
                                split_input, "resizer", preprocess
                            )
                    else:
                        self.set_pyramid_source(input, "resizer", preprocess)
                else:
                    raise ValueError(
                        "input source expected `pyramid` or `resizer, but get {}".format(  # noqa
                            source
                        )
                    )

            logger.info("change all source success!")

    @staticmethod
    def _normalize_permutes(permutes: Union[List[int], List[str]]) -> List[int]:
        return [int(p) for p in permutes]

    def _apply_transpose_cfg(self, nodes, cfg: Dict[str, List[int]]):
        if not cfg:
            return
        if "global" in cfg:
            permutes = self._normalize_permutes(cfg["global"])
            for node in nodes:
                node.insert_transpose(permutes)
            return
        for name, permutes in cfg.items():
            permutes = self._normalize_permutes(permutes)
            for node in nodes:
                if node.name == name:
                    node.insert_transpose(permutes)

    def set_transpose_nodes(self, model):
        if not self.transpose_dim:
            return
        func = model[0]
        if "inputs" in self.transpose_dim:
            self._apply_transpose_cfg(func.inputs, self.transpose_dim["inputs"])
        if "outputs" in self.transpose_dim:
            self._apply_transpose_cfg(
                func.outputs, self.transpose_dim["outputs"]
            )

    def __call__(self):
        self.set_input_nodes(self.model, self.input_source)
        self.set_transpose_nodes(self.model)
        if self.enable_integer_conv and self.march in (
            "nash-b",
            "nash-h",
            "nash-p",
        ):
            self.model._integer_conv = True

        if self.enable_llm_convert:
            self.model = llm_convert(
                self.model, self.march, **self.llm_convert_kwargs
            )
        if self.use_f16_quant_dequant_on_vae_always:
            self.model._use_f16_quant_dequant_on_vae_always = True

        quantized_model = hbdk4_compiler.convert(
            self.model,
            self.march,
            enable_vpu=self.enable_vpu,
            enable_spu=self.enable_spu,
        )

        if not self.preserve_quant_dequant:
            quantized_model[0].remove_io_op(
                op_types=list(self.remove_io_op_types)
            )

        if self.remove_nodes_ioname:
            for node in self.remove_nodes_ioname:
                self.remove_op_by_ioname(quantized_model[0], node)

        out_dir = os.path.dirname(self.out_path)

        self.hbir_disas(quantized_model[0])

        hbdk4_compiler.save(
            quantized_model, os.path.join(out_dir, "quantized_deploy.bc")
        )

        warn_cpu_ops(
            hbdk4_compiler.statistics(quantized_model),
            model_name=os.path.basename(self.out_path),
            warn=self.warn_cpu_ops,
            strict=os.environ.get("WARN_CPU_OPS_STRICT", "").lower()
            in ("1", "true"),
        )

        compile_kwargs = dict(
            path=self.out_path,
            march=self.march,
            opt=self.opt,
            jobs=self.jobs,
            max_time_per_fc=self.max_time_per_fc,
            debug=self.debug,
            balance=self.balance,
            input_no_padding=self.input_no_padding,
            output_no_padding=self.output_no_padding,
            cache_path=self.cache_path,
            cache_mode=self.cache_mode,
            progress_bar=True,
        )
        if self.core_num is not None:
            compile_kwargs["core_num"] = self.core_num
        if self.enable_hpc is not None:
            compile_kwargs["enable_hpc"] = self.enable_hpc
        if self.max_l2m_size is not None:
            compile_kwargs["max_l2m_size"] = self.max_l2m_size
        hbdk4_compiler.compile(quantized_model, **compile_kwargs)

        if self.out_path.endswith(".hbm"):
            hbdk4_compiler.hbm_perf(
                self.out_path,
                remote_ip=self.remote_ip,
                remote_port=self.remote_port,
                remote_work_root="/map/hbm_infer",
            )

        logger.info(
            format_msg("Compiled model: %s" % self.out_path, MSGColor.GREEN)
        )
