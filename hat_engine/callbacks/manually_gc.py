# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import gc
import logging

from hat_engine.registry import OBJECT_REGISTRY
from .callbacks import CallbackMixin

__all__ = ["ManuallyGCCallback"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class ManuallyGCCallback(CallbackMixin):
    """
    Callback to manually trigger garbage collection.

    This callback allows manual garbage collection during the training process.
    It can be useful for managing memory in long-running training sessions.
    """

    def __init__(self, manual_gc_collect_freq: int):
        self.manual_gc_collect_freq = manual_gc_collect_freq
        gc.disable()
        logger.info("Automatic garbage collection is disabled. ")
        logger.info(
            "Manually triggered garbage collection"
            f" by step frequency: {self.manual_gc_collect_freq}."
        )

    def _trigger_gc(self):
        """Trigger garbage collection."""
        gc.collect()
        gc.disable()
        logger.info("Garbage collection triggered.")

    def on_step_end(self, step_id, *args, **kwargs):
        """Trigger garbage collection at the end of each step."""
        if step_id % self.manual_gc_collect_freq == 0:
            self._trigger_gc()

    def on_loop_end(self, **kwargs):
        gc.enable()
        logger.info("Automatic garbage collection is enabled again.")
