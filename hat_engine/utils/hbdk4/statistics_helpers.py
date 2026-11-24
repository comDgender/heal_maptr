# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import logging

import torch

from hat_engine.utils.logger import MSGColor, format_msg

__all__ = ["warn_cpu_ops", "_recurse_cast_to_fp32"]

logger = logging.getLogger(__name__)


def warn_cpu_ops(op_stats, model_name="", warn=False, strict=False):
    """Print op stats; when ``warn=True``, flag hbtl.call CPU fallbacks."""
    for i, func_stats in enumerate(op_stats):
        print(f"[{model_name}] function {i} op counts:", func_stats)

    if not warn:
        return

    cpu_ops = {}
    for func_stats in op_stats:
        for op, count in func_stats.items():
            if op.startswith("hbtl.call"):
                cpu_ops[op] = cpu_ops.get(op, 0) + count
    if cpu_ops:
        details = ", ".join(f"{k}:{v}" for k, v in cpu_ops.items())
        msg = (
            f"[{model_name}] Quantized HBIR contains CPU op(s): {details}. "
            "These would fall back to CPU on hbm. Inspect the saved .bc "
            "for details."
        )
        logger.warning(format_msg(msg, MSGColor.RED))
        if strict:
            raise RuntimeError(msg)


def _recurse_cast_to_fp32(inputs):
    """Recursively cast floating tensors to fp32; other leaves untouched."""
    if isinstance(inputs, torch.Tensor):
        if torch.is_floating_point(inputs):
            return inputs.float()
        return inputs
    if isinstance(inputs, (list, tuple)):
        return type(inputs)(_recurse_cast_to_fp32(x) for x in inputs)
    return inputs
