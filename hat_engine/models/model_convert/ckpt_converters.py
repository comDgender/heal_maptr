# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import logging
from typing import Callable, Optional

try:
    import torch.distributed.checkpoint as dcp
except ImportError:
    dcp = None

from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.checkpoint import load_checkpoint, load_state_dict
from hat_engine.utils.global_var import set_value
from hat_engine.utils.logger import MSGColor, format_msg
from hat_engine.utils.package_helper import require_packages
from .converters import BaseConverter

__all__ = [
    "LoadCheckpoint",
    "LoadDistributedCheckpoint",
    "LoadMeanTeacherCheckpoint",
]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class LoadCheckpoint(BaseConverter):
    """Load the checkpoint from file to model and return the checkpoint.

    LoadCheckpoint usually happens before or after BaseConverter.It means
    the model needs to load parameters before or after BaseConverter.

    Args:
        checkpoint_path: Path of the checkpoint file.
        state_dict_update_func: `state_dict` update function. The input
            of the function is a `state_dict`, The output is a modified
            `state_dict` as you want.
        check_hash: Whether to check the file hash.
        allow_miss: Whether to allow missing while loading state dict.
        ignore_extra: Whether to ignore extra while loading state dict.
        ignore_tensor_shape: Whether to ignore matched key name but
            unmatched shape of tensor while loading state dict.
        verbose: Show unexpect_key and miss_key info.
        enable_tracking: whether enable tracking checkpoint.
        load_ema_model: Whether to load weights from ``ema_model`` entry.
        zero_to_fp32: Rebuild fp32 weights from ZeRO shards. ``None``
            gathers ZeRO-3 only; ``True`` also gathers ZeRO-1/2 from
            optimizer partitions; ``False`` unwraps ``mp_rank_00``.
            ``checkpoint_path`` may be a tag dir, a save dir with
            ``latest``, or ``mp_rank_00_model_states.pt``.
    """

    def __init__(
        self,
        checkpoint_path: str,
        state_dict_update_func: Optional[Callable] = None,
        check_hash: bool = True,
        allow_miss: bool = False,
        ignore_extra: bool = False,
        ignore_tensor_shape: bool = False,
        verbose: bool = False,
        enable_tracking: bool = False,
        load_ema_model: bool = False,
        zero_to_fp32: Optional[bool] = None,
    ):
        super(LoadCheckpoint, self).__init__()
        self.checkpoint_path = checkpoint_path
        self.state_dict_update_func = state_dict_update_func
        self.check_hash = check_hash
        self.allow_miss = allow_miss
        self.ignore_extra = ignore_extra
        self.ignore_tensor_shape = ignore_tensor_shape
        self.verbose = verbose
        self.enable_tracking = enable_tracking
        self.load_ema_model = load_ema_model
        self.zero_to_fp32 = zero_to_fp32

    def __call__(self, model):
        if self.checkpoint_path is None:
            logger.info(
                format_msg(
                    f"Checkpoint path is None, skip.",  # noqa E501
                    MSGColor.RED,
                )
            )
            return model

        model_checkpoint = load_checkpoint(
            path_or_dict=self.checkpoint_path,
            map_location="cpu",
            state_dict_update_func=self.state_dict_update_func,
            check_hash=self.check_hash,
            enable_tracking=self.enable_tracking,
            load_ema_model=self.load_ema_model,
            zero_to_fp32=self.zero_to_fp32,
        )
        if (
            "optimizer" in model_checkpoint
            and isinstance(model_checkpoint["optimizer"], dict)
            and "state" not in model_checkpoint["optimizer"]
        ):
            model_checkpoint["optimizer"]["state"] = {}
        set_value("model_checkpoint", model_checkpoint)
        model = load_state_dict(
            model,
            path_or_dict=model_checkpoint,
            allow_miss=self.allow_miss,
            ignore_extra=self.ignore_extra,
            ignore_tensor_shape=self.ignore_tensor_shape,
            verbose=self.verbose,
        )
        logger.info(
            format_msg(
                f"Load the checkpoint successfully from {self.checkpoint_path}",  # noqa E501
                MSGColor.GREEN,
            )
        )
        return model


@OBJECT_REGISTRY.register
class LoadDistributedCheckpoint(LoadCheckpoint):
    """Load the checkpoint from file to model and return the checkpoint.

    LoadDistributedCheckpoint usually happens before or after BaseConverter.
    It means the model needs to load parameters before or after BaseConverter.
    LoadDistributedCheckpoint is used to load dcp format checkpoint.

    Args:
        checkpoint_path: Path of the dcp checkpoint dir.
        state_dict_update_func: `state_dict` update function. The input
            of the function is a `state_dict`, The output is a modified
            `state_dict` as you want.
        check_hash: Whether to check the file hash.
        allow_miss: Whether to allow missing while loading state dict.
        ignore_extra: Whether to ignore extra while loading state dict.
        ignore_tensor_shape: Whether to ignore matched key name but
            unmatched shape of tensor while loading state dict.
        verbose: Show unexpect_key and miss_key info.
        enable_tracking: whether enable tracking checkpoint.
    """

    @require_packages("torch>=2.3")
    def __init__(
        self,
        checkpoint_path: str,
        state_dict_update_func: Optional[Callable] = None,
        check_hash: bool = True,
        allow_miss: bool = False,
        ignore_extra: bool = False,
        ignore_tensor_shape: bool = False,
        verbose: bool = False,
        enable_tracking: bool = False,
        load_ema_model: bool = False,
    ):
        super().__init__(
            checkpoint_path,
            state_dict_update_func,
            check_hash,
            allow_miss,
            ignore_extra,
            ignore_tensor_shape,
            verbose,
            enable_tracking,
            load_ema_model,
        )
        assert (
            "//" not in checkpoint_path
        ), "Now only support local dcp checkpoint file load!"

    def __call__(self, model):
        if self.checkpoint_path is None:
            logger.info(
                format_msg(
                    f"Checkpoint path is None, skip.",  # noqa E501
                    MSGColor.RED,
                )
            )
            return model
        state_dict = {
            "epoch": 1,
            "step": 1,
            "devices": 0,
            "grad_scaler": {},
            "state_dict": model.state_dict(),
            "optimizer": {},
            "horizon-plugin-version": "",
            "ema_model": {},
        }

        dcp.load(
            state_dict=state_dict,
            checkpoint_id=self.checkpoint_path,
        )
        set_value("model_checkpoint", state_dict)

        weights = state_dict["state_dict"]
        if self.load_ema_model and state_dict.get("ema_model"):
            weights = state_dict["ema_model"]

        model = load_state_dict(
            model,
            path_or_dict=weights,
            allow_miss=self.allow_miss,
            ignore_extra=self.ignore_extra,
            ignore_tensor_shape=self.ignore_tensor_shape,
            verbose=self.verbose,
        )
        logger.info(
            format_msg(
                f"Load the checkpoint successfully from {self.checkpoint_path}",  # noqa E501
                MSGColor.GREEN,
            )
        )
        return model


@OBJECT_REGISTRY.register
class LoadMeanTeacherCheckpoint(BaseConverter):
    """Load the Mean-teacher model checkpoint.

    student and teacher model have same structure.
    LoadMeanTeacherCheckpoint usually happens before or after BaseConverter.
    It means the model needs to load parameters before or after BaseConverter.

    Args:
        checkpoint_path: Path of the checkpoint file.
        state_dict_update_func: `state_dict` update function. The input
            of the function is a `state_dict`, The output is a modified
            `state_dict` as you want.
        check_hash: Whether to check the file hash.
        allow_miss: Whether to allow missing while loading state dict.
        ignore_extra: Whether to ignore extra while loading state dict.
        verbose: Show unexpect_key and miss_key info.
    """

    def __init__(
        self,
        checkpoint_path: str,
        strip_prefix: str = "module.",
        state_dict_update_func: Optional[Callable] = None,
        check_hash: bool = True,
        allow_miss: bool = False,
        ignore_extra: bool = False,
        verbose: bool = False,
    ):
        super(LoadMeanTeacherCheckpoint, self).__init__()
        self.checkpoint_path = checkpoint_path
        self.strip_prefix = strip_prefix
        self.state_dict_update_func = state_dict_update_func
        self.check_hash = check_hash
        self.allow_miss = allow_miss
        self.ignore_extra = ignore_extra
        self.verbose = verbose

    def __call__(self, model):
        if self.checkpoint_path is None:
            logger.info(
                format_msg(
                    f"Checkpoint path is None, skip.",  # noqa E501
                    MSGColor.RED,
                )
            )
            return model

        model_checkpoint = load_checkpoint(
            path_or_dict=self.checkpoint_path,
            map_location="cpu",
            state_dict_update_func=self.state_dict_update_func,
            check_hash=self.check_hash,
        )
        set_value("model_checkpoint", model_checkpoint)
        model = load_state_dict(
            model,
            path_or_dict=model_checkpoint,
            strip_prefix=self.strip_prefix,
            state_dict_update_func=self.state_dict_update_func,
            allow_miss=self.allow_miss,
            ignore_extra=self.ignore_extra,
            verbose=self.verbose,
        )
        logger.info(
            format_msg(
                f"Load the checkpoint successfully from {self.checkpoint_path}",  # noqa E501
                MSGColor.GREEN,
            )
        )
        return model
