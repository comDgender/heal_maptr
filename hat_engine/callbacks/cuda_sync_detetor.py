# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import copy
import logging
import warnings
from warnings import WarningMessage, _formatwarnmsg

import torch

from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.package_helper import require_packages
from .callbacks import CallbackMixin

__all__ = ["CudaSyncDetector"]

logger = logging.getLogger(__name__)


def showwarning(message, category, filename, lineno, file=None, line=None):
    """Write a warning to a file; replace if you like."""
    msg = WarningMessage(message, category, filename, lineno, file, line)
    _showwarnmsg_impl(msg)


def _showwarnmsg_impl(msg):
    text = _formatwarnmsg(msg)
    logger.warning(text)


@OBJECT_REGISTRY.register
class CudaSyncDetector(CallbackMixin):
    """
    Detect CUDA synchronization issues.

    This callback detects CUDA synchronization issues by setting the
    CUDA synchronization debug mode to "warn" or "error".

    Args:
        warn_once: Whether to only show the cuda sync issues in
            first step. Defaults to True. If set to False, the
            cuda sync issues will be shown every step.
        mode: The mode of the CUDA synchronization detector.
            Options are "warn" and "error".
    """

    @require_packages("torch>=2.0")
    def __init__(self, warn_once: bool = True, mode: str = "warn"):
        assert mode in [
            "warn",
            "error",
        ], f"mode must be 'warn' or 'error', got {mode}"
        self.mode = mode
        self.warn_once = warn_once
        self._warn_flag = False

    def _set_warning_filter(self):
        """Set the warning filter to catch CUDA synchronization issues."""
        self.old_filter = copy.deepcopy(warnings.filters[1:])
        warnings.filterwarnings(
            "always",
            message=".*called a synchronizing CUDA operation.*",
            category=UserWarning,
            module="",
            lineno=0,
            append=False,
        )
        self.old_showwarning = warnings.showwarning
        warnings.showwarning = showwarning

    def _rollback_warning_filter(self):
        """Rollback the warning filter to the original state."""
        warnings.resetwarnings()
        for f in self.old_filter:
            warnings.filters.append(f)
        warnings.showwarning = self.old_showwarning
        self.old_showwarning = None
        self.old_filter = []

    def on_forward_begin(self, **kwargs):
        if self.warn_once and self._warn_flag:
            return
        self._set_warning_filter()
        self.debug_mode_restore = torch.cuda.get_sync_debug_mode()
        torch.cuda.set_sync_debug_mode(self.mode)
        logger.info(f"Set CUDA synchronization detector mode to {self.mode}")

    def on_step_end(self, **kwargs):
        if self.warn_once and self._warn_flag:
            return
        torch.cuda.set_sync_debug_mode(self.debug_mode_restore)
        logger.info(
            "Reset CUDA synchronization detector"
            f" mode to {self.debug_mode_restore}"
        )
        self._rollback_warning_filter()
        if self.warn_once:
            self._warn_flag = True
