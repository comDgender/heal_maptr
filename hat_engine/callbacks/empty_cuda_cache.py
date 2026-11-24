# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This file has been modified by Beijing Horizon Information Technology Co., Ltd. on 2026/08/06.
# Modifications: Adapted the original implementation to our framework;
#
# This source code is licensed under the BSD-style license found in the
# LICENSE-BSD-style file.

import logging

import torch

from hat_engine.registry import OBJECT_REGISTRY
from .callbacks import CallbackMixin

__all__ = ["EmptyCudaCache"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class EmptyCudaCache(CallbackMixin):
    """
    Callback to empty the CUDA cache.

    A callback that performs periodic emptying of cuda cache using
    `torch.cuda.empty_cache <https://pytorch.org/docs/stable/generated/torch.cuda.empty_cache.html#torch.cuda.empty_cache>_`. # noqa

    On different ranks, reserved memory and fragmentation might
    diverge after several iterations. If different ranks trigger
    de-fragmentation (i.e. cudaFree and redo cudaMalloc later)
    at different times, there will be different stragglers in
    different iterations, which will hurt the performance and will
    get worse with larger clusters. To avoid this, this callback
    calls empty_cache() at the same cadence across all ranks.

    Args:
        step_interval: number of steps to run before emptying cuda cache
    """

    def __init__(self, step_interval: int) -> None:
        self._step_interval = step_interval

    def do_empty_cache(self):
        torch.cuda.empty_cache()

    def on_step_end(self, step_id: int, **kwargs) -> None:
        if step_id % self._step_interval == 0:
            self.do_empty_cache()
            logger.info(f"Emptied CUDA cache at step {step_id}.")
