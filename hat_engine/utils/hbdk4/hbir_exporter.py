# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import logging
import os
from typing import List

try:
    import horizon_plugin_pytorch as horizon
except ImportError:
    horizon = None
import torch
from torch.utils._pytree import tree_flatten

from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.apply_func import _as_list
from hat_engine.utils.logger import MSGColor, format_msg
from hat_engine.utils.package_helper import require_packages

try:
    import hbdk4.compiler as hbdk4_compiler
    from hbdk4.compiler import statistics  # noqa: E402
    from hbdk4.compiler.extra_apis import llm_convert  # noqa: E402
    from horizon_plugin_pytorch.quantization import (
        FakeQuantState,
        set_fake_quantize,
    )
    from horizon_plugin_pytorch.quantization.hbdk4 import export
except Exception:
    hbdk4_compiler = None
    statistics = None
    llm_convert = None
    export = None
    set_fake_quantize = None
    FakeQuantState = None

from hat_engine.utils.hbdk4.statistics_helpers import (
    _recurse_cast_to_fp32,
    warn_cpu_ops,
)

__all__ = ["HbirExporter"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class HbirExporter:
    """Export hbir from qat nn.Module and convert hbir to quantized.

    Args:
        model: Input model.
        model_convert_pipeline:
            Model convert pipeline that converts model to qat mode.
        example_inputs: Model input for exporting.
            **NOTE**: Hbir model do not support dynamic shape currently.
        march: BPU march.
        save_path: Where to save exported hbir files.
        with_check: Whether check the mlir model forward with example input.
        model_name: Name of compiled model.
        enable_vpu: Whether enable vpu for quantized hbir.
        kwargs: other params of export.
    """

    @require_packages("horizon_plugin_pytorch>=1.10.3", "hbdk4")
    def __init__(
        self,
        model: torch.nn.Module,
        model_convert_pipeline: List[callable],
        example_inputs: dict,
        march: str,
        save_path: str,
        with_check: bool,
        model_name: str = "model",
        enable_vpu: bool = True,
        enable_spu: bool = True,
        enable_llm_convert: bool = False,
        warn_cpu_ops: bool = False,
        rmsnorm_version: str = None,
        softmax_version: str = None,
        use_f16_quant_dequant_on_vae_always: bool = False,
        **kwargs
    ) -> None:
        horizon.march.set_march(march)

        self.model = model_convert_pipeline(model)
        self.example_inputs = _as_list(example_inputs)
        self.march = march
        self.save_path = save_path
        self.with_check = with_check
        self.model_name = model_name
        self.enable_vpu = enable_vpu
        self.enable_spu = enable_spu
        self.existing_fp16 = bool(kwargs.pop("existing_fp16", False))
        self.march_modify = kwargs.pop("march_modify", None)
        self.march_modify_list = ("nash-b", "nash-h", "nash-p")
        self.kwargs = kwargs

        self.enable_llm_convert = enable_llm_convert
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

    def get_hbir_input(self, example_inputs):
        flat_inputs, _ = tree_flatten(example_inputs)
        flat_inputs = horizon.utils.misc.pytree_convert(
            flat_inputs, torch.Tensor, lambda x: x.cpu().numpy()
        )

        return flat_inputs

    def save_hbir(self, model, path: str, debug: bool):
        dir_name = os.path.dirname(path)
        if len(dir_name) > 0:
            os.makedirs(dir_name, exist_ok=True)

        if debug:
            if isinstance(model, hbdk4_compiler.Module):
                model = model.module

            if not path.endswith(".mlir"):
                path += ".mlir"
            with open(path, "w") as f:
                f.writelines(
                    str(
                        model.operation.get_asm(
                            enable_debug_info=True, pretty_debug_info=False
                        )
                    )
                )
        else:
            if not path.endswith(".bc"):
                path += ".bc"
            hbdk4_compiler.save(model, path)

    def __call__(self, debug: bool) -> None:
        logger.info(
            "Exporting hbir with input {}".format(
                horizon.utils.misc.tensor_struct_repr(self.example_inputs)
            )
        )

        if self.march_modify in self.march_modify_list:
            from horizon_plugin_pytorch.utils.global_quant_round_mode import (
                QuantRoundMode,
            )

            QuantRoundMode.set(QuantRoundMode.HALF_TO_EVEN)
            self.march = self.march_modify
            horizon.march.set_march(self.march)

        self.model.eval()
        if set_fake_quantize is not None:
            set_fake_quantize(self.model, FakeQuantState.VALIDATION)

        self.model = self.model.float()
        self.example_inputs = _recurse_cast_to_fp32(self.example_inputs)

        qat_hbir = export(
            self.model,
            self.example_inputs,
            name=self.model_name,
            native_pytree=False,
            **self.kwargs
        )
        # int64 批量替换int32
        qat_hbir.replace_index_tensor_type()
        # 删除悬空的输入输出节点
        for arg in qat_hbir.functions[0].flatten_inputs[::-1]:
            if not arg.get_attached_op:
                print(arg.name)
                arg.erase()
        for arg in qat_hbir.functions[0].flatten_outputs[::-1]:
            if not arg.get_attached_op:
                print(arg.name)
                arg.erase()

        logger.info(
            format_msg(
                "Saving qat hbir to {}".format(self.save_path), MSGColor.GREEN
            )
        )
        self.save_hbir(qat_hbir, os.path.join(self.save_path, "qat"), debug)

        if self.with_check:
            logger.info("Checking qat hbir with example_inputs.")
            qat_hbir.functions[0](*self.get_hbir_input(self.example_inputs))
            logger.info(format_msg("Qat hbir check passed.", MSGColor.GREEN))

        if self.enable_llm_convert:
            qat_hbir = llm_convert(
                qat_hbir, self.march, **self.llm_convert_kwargs
            )
        if self.use_f16_quant_dequant_on_vae_always:
            qat_hbir._use_f16_quant_dequant_on_vae_always = True

        logger.info("Converting hbir to quantized.")
        if (
            self.march_modify in self.march_modify_list
            and not self.existing_fp16
        ):
            qat_hbir._integer_conv = True
        quantized_hbir = hbdk4_compiler.convert(
            qat_hbir,
            self.march,
            enable_vpu=self.enable_vpu,
            enable_spu=self.enable_spu,
        )
        logger.info(
            format_msg(
                "Saving quantized hbir to {}".format(self.save_path),
                MSGColor.GREEN,
            )
        )
        self.save_hbir(
            quantized_hbir,
            os.path.join(self.save_path, "quantized_infer"),
            debug,
        )
        warn_cpu_ops(
            statistics(quantized_hbir),
            model_name=self.model_name,
            warn=self.warn_cpu_ops,
            strict=os.environ.get("WARN_CPU_OPS_STRICT", "").lower()
            in ("1", "true"),
        )

        if self.with_check:
            logger.info("Checking quantized hbir with example_inputs.")
            quantized_hbir.functions[0](
                *self.get_hbir_input(self.example_inputs)
            )
            logger.info(
                format_msg("Quantized hbir check passed.", MSGColor.GREEN)
            )
