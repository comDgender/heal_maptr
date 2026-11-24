# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import logging
from typing import List

import torch

from hat_engine.registry import OBJECT_REGISTRY
from .profilers import BaseProfiler, ProfilerAction, schedule

__all__ = ["NVTXProfiler"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class NVTXProfiler(BaseProfiler):
    """
    NVTX Profiler for PyTorch.

    This profiler uses NVTX to mark regions in the code for profiling.
    It is useful for performance analysis and debugging.

    Args:
        active (int): Number of profiling steps to be active.
        warmup (int, optional): Number of warmup steps before profiling
            starts. Defaults to 0.
        repeat (int, optional): Number of times to repeat the profiling.
            Defaults to 0.
        skip_first (int, optional): Number of initial steps to skip before
            starting profiling. Defaults to 0.
        with_sync (bool, optional): Whether to synchronize CUDA operations.
            Defaults to False.
    """

    def __init__(
        self,
        active: int,
        warmup: int = 0,
        repeat: int = 0,
        skip_first: int = 0,
        skip_actions: List[str] = None,
        with_sync: bool = False,
    ):
        schedule_fn = schedule(
            active=active,
            warmup=warmup,
            repeat=repeat,
            skip_first=skip_first,
        )
        super().__init__(schedule=schedule_fn, with_sync=with_sync)
        self.nvtx_context = torch.autograd.profiler.emit_nvtx(
            record_shapes=True
        )
        self._init_flag = False
        self.stack_entered = []

        if skip_actions is None:
            self.skip_actions = []
        else:
            self.skip_actions = skip_actions

    def step(self):
        self._init_flag = True
        if self._prof_current_action == ProfilerAction.RECORD_AND_SAVE:
            torch.cuda.cudart().cudaProfilerStop()
            if self.nvtx_context.entered:
                self.nvtx_context.__exit__(None, None, None)
            self._init_flag = False
            logger.info("NVTX profiling completed.")
        return super().step()

    def start(self, action_name: str):
        if action_name in self.skip_actions:
            return
        self.stack_entered.append(action_name)
        if (
            self._init_flag
            and len(self.stack_entered) <= 1
            and self._prof_current_action == ProfilerAction.RECORD
            and not self.nvtx_context.entered
        ):
            logger.info("NVTX profiling started.")
            torch.cuda.cudart().cudaProfilerStart()
            self.nvtx_context.__enter__()
        torch.cuda.nvtx.range_push(action_name)

    def stop(self, action_name: str):
        if action_name in self.skip_actions:
            return
        torch.cuda.nvtx.range_pop()
        self.stack_entered.pop()

    def reset(self):
        if self._init_flag:
            torch.cuda.cudart().cudaProfilerStop()
            if self.nvtx_context.entered:
                self.nvtx_context.__exit__(None, None, None)
            self._init_flag = False
            logger.info("NVTX profiler reset.")

    def summary(self):
        pass
