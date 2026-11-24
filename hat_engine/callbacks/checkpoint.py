# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
import logging
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from distutils.version import LooseVersion
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from hat_engine.utils.hash import generate_sha256_file, remove_file_by_regex

try:
    import torch.distributed.checkpoint as dcp
    import torch.distributed.fsdp as FSDP
    from torch.distributed.checkpoint import StorageWriter
    from torch.distributed.fsdp import (
        OptimStateDictConfig,
        ShardedStateDictConfig,
        StateDictType,
    )
except ImportError:
    FSDP = None
    dcp = None
    StorageWriter = None
    ShardedStateDictConfig = None
    OptimStateDictConfig = None
    StateDictType = None

try:
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
except ImportError:
    dcp_to_torch_save = None

try:
    from torch.distributed.checkpoint.state_dict import get_model_state_dict
except ImportError:
    get_model_state_dict = None

from hat_engine.metrics.metric import EvalMetric
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.apply_func import _as_list
from hat_engine.utils.checkpoint import (
    get_flash_ckpt_names,
    save_shm_ckpt_to_disk,
)
from hat_engine.utils.checkpoint_saver import FlashCheckpointer
from hat_engine.utils.distributed import (
    all_gather_object,
    get_device_count,
    get_dist_info,
    rank_zero_only,
)
from hat_engine.utils.elastic import ElasticState, use_elastic
from hat_engine.utils.hierarchical_dist import maybe_sync_params
from hat_engine.utils.package_helper import compare_version, require_packages
from .callbacks import CallbackMixin

try:
    import apex
except ImportError:
    apex = None

try:
    import deepspeed
except ImportError:
    deepspeed = None

__all__ = ["Checkpoint", "DistributedCheckpoint"]

logger = logging.getLogger(__name__)

TRAIN_CHECKPOINT_FORMAT = "%scheckpoint-%s.pth.tar"
TRAIN_DCP_CHECKPOINT_FORMAT = "%scheckpoint-%s"


def get_fsdp_state_dict(model: nn.Module) -> Dict[str, Any]:
    """Get state dict of FullyShardedDataParallel model.

    Args:
        model: FullyShardedDataParallel model.

    Returns:
        Dict of param names and values
    """

    def _new_recommend_fsdp_state_dict(model):
        cpu_state = {}
        save_policy = FSDP.FullStateDictConfig(
            offload_to_cpu=True, rank0_only=True
        )
        with FSDP.FullyShardedDataParallel.state_dict_type(
            model, FSDP.StateDictType.FULL_STATE_DICT, save_policy
        ):
            cpu_state = get_model_state_dict(model)
        return cpu_state

    def _old_version_fsdp_state_dict(model):
        FSDP.set_state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
            optim_state_dict_config=OptimStateDictConfig(offload_to_cpu=True),
        )
        cpu_state = model.state_dict()
        return cpu_state

    if compare_version("torch", ">=", "2.3.0"):
        cpu_state = _new_recommend_fsdp_state_dict(model)
    else:
        cpu_state = _old_version_fsdp_state_dict(model)
    return cpu_state


def get_valid_state_dict(model: nn.Module, only_save_ddp=False) -> dict:
    """Remove 'module' prefix in param name if model is ddp or dp model.

    Args:
        model: Module that need to get state dict.
        only_save_ddp: Only save ddp module.

    Returns:
        Dict of param names and values
    """
    if isinstance(
        model, (nn.parallel.DistributedDataParallel, nn.DataParallel)
    ):
        return model.module.state_dict()
    elif deepspeed is not None and isinstance(
        model, deepspeed.DeepSpeedEngine
    ):
        return model.module.state_dict()
    elif apex is not None and isinstance(
        model, apex.parallel.distributed.DistributedDataParallel
    ):
        return model.module.state_dict()
    elif FSDP is not None and isinstance(model, FSDP.FullyShardedDataParallel):
        cpu_state = get_fsdp_state_dict(model)
        return cpu_state

    elif isinstance(model, nn.Module):
        ddp_module_dict = {}
        fsdp_module_dict = {}
        for name, module in model.named_children():
            if isinstance(module, nn.parallel.DistributedDataParallel):
                ddp_module_dict[name] = module
            elif FSDP is not None and isinstance(
                module, FSDP.FullyShardedDataParallel
            ):
                fsdp_module_dict[name] = module

        if ddp_module_dict:
            # only save ddp module
            if only_save_ddp:
                model_bak = {}
                for name, mod in ddp_module_dict.items():
                    for key, values in mod.state_dict().items():
                        model_bak[key.replace("module", name)] = values
                return model_bak

            # support submodule is ddp by remove 'module' prefix
            new_state_dict = OrderedDict()
            state_dict = model.state_dict()
            for key, values in state_dict.items():
                new_state_dict[key.replace("module.", "")] = values
            return new_state_dict
        elif FSDP is not None and fsdp_module_dict:
            # only save fsdp module
            if only_save_ddp:
                model_bak = {}
                for name, mod in fsdp_module_dict.items():
                    cpu_state = get_fsdp_state_dict(mod)
                    for key, values in cpu_state.items():
                        model_bak[key.replace("module", name)] = values
                return model_bak

            # support submodule is fsdp.
            model_state_dict = get_fsdp_state_dict(model)
            return model_state_dict
        else:
            return model.state_dict()
    else:
        raise NotImplementedError("unknown model type: %s" % type(model))


def _get_average_tensor(tensors: List[Tensor]):
    """Get average tensor of a list of tensors.

    When compute average for low precision tensor, using high precision dtype.
    """
    with torch.no_grad():
        origin_type = tensors[0].dtype
        for i, t in enumerate(tensors):
            assert (
                origin_type == t.dtype
            ), f"{i}th tensor type is {t.dtype} not {origin_type}"

        if origin_type == torch.float16:
            tmp = tensors[0].to(torch.float32)
        elif origin_type == torch.uint8:
            tmp = tensors[0].to(torch.int32)
        elif origin_type == torch.int8:
            tmp = tensors[0].to(torch.int32)
        else:
            tmp = tensors[0]

        for t in tensors[1:]:
            tmp = tmp.add(t.to(tmp.device))
        tmp = tmp / len(tensors)
        return tmp.to(origin_type)


@OBJECT_REGISTRY.register
class Checkpoint(CallbackMixin):  # noqa: D205,D400
    """
    Checkpoint Callback is used for saving model after training
    and resume model before training as the same times.

    Args:
        save_dir: Directory to save checkpoints.
        name_prefix: Checkpoint name prefix.
        save_interval: Save checkpoint every `save_interval` epoch or step.
        interval_by: Set `save_interval` unit to step or epoch.
            Default is epoch.
        save_on_train_end: Whether save checkpoint when `on_loop_end` is
            triggered.
        strict_match: Whether to strictly enforce that the keys in
            `model.state_dict()` (train model) match the keys in
            `test_model.state_dict()`. Default: ``False``
        mode: State of monitor for saving model.
        monitor_metric_key: Monitor metric for saving best checkpoint.
        best_refer_metric: Metric that evaluate which epoch is the best.
        save_hash: Whether to save the hash value to the name of the
             Checkpoint file. Default is True.
        only_save_ddp: Only save ddp module. Default is False.
        keep_n: Keep the last n checkpoints. Default is None.
        shm_save_freq_factor: Checkpoint Save frequency factor to shared memory
            during each save interval.Default is 5.Tune down this factor can
            improve training performance, min value is 1.
    """

    SUPPORTED_MODES = ["min", "max"]

    def __init__(
        self,
        save_dir: str,
        name_prefix: Optional[str] = "",
        save_interval: Optional[int] = 1,
        interval_by: Optional[str] = "epoch",
        save_on_train_end: Optional[bool] = True,
        strict_match: Optional[bool] = False,
        mode: Optional[str] = None,
        monitor_metric_key: Optional[str] = None,
        best_refer_metric: Optional[Union[dict, EvalMetric]] = None,
        task_sampler=None,
        save_hash: bool = True,
        only_save_ddp: bool = False,
        keep_n: Optional[int] = None,
        shm_save_freq_factor: Optional[int] = 5,
    ):
        self.save_dir = save_dir
        self.name_prefix = name_prefix
        self.save_interval = save_interval
        self.interval_by = interval_by
        self.save_on_train_end = save_on_train_end
        self.strict_match = strict_match
        self.task_sampler = task_sampler
        self.save_hash = save_hash
        self.only_save_ddp = only_save_ddp
        self.shm_save_freq_factor = shm_save_freq_factor
        (
            self.flash_ckpt_name,
            self.best_flash_ckpt_name,
        ) = get_flash_ckpt_names()
        self.async_ckpter = FlashCheckpointer(self.flash_ckpt_name)
        self.best_async_ckpter = FlashCheckpointer(self.best_flash_ckpt_name)
        self.save_disk_ckpt_thread = None
        self.shm_handles = []
        self._staging_executor = ThreadPoolExecutor(max_workers=1)
        self._staging_future = None
        self._best_staging_executor = ThreadPoolExecutor(max_workers=1)
        self._best_staging_future = None
        assert self.interval_by in ("epoch", "step")
        assert (
            isinstance(shm_save_freq_factor, int) and shm_save_freq_factor > 0
        )

        self.mode = mode
        self.monitor_metric_key = monitor_metric_key
        if best_refer_metric is not None:
            self.best_refer_metric = _as_list(best_refer_metric)
        else:
            self.best_refer_metric = None
        if self.mode is not None:
            if self.mode == "min":
                self.monitor_op = np.less
                self.best = np.Inf
            elif self.mode == "max":
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                raise ValueError(
                    "Supported modes: %s, but get %s"
                    % (self.SUPPORTED_MODES, self.mode)
                )

        self.keep_n = keep_n
        self.epoch_ckpt_files = []

    def _save_ckpt(self, file_ext, ckpt_type, sha, ckpt_file_with_root):
        file_name = TRAIN_CHECKPOINT_FORMAT % (self.name_prefix, ckpt_type)
        last_file = os.path.join(self.save_dir, file_name)
        os.system(f"cp {ckpt_file_with_root} {last_file}")
        if self.save_hash:
            prefix = file_name[: -len(file_ext)]
            regex = "%s-([a-f0-9]*)\\." % prefix
            remove_file_by_regex(self.save_dir, regex)
            file = os.path.join(
                self.save_dir,
                TRAIN_CHECKPOINT_FORMAT
                % (self.name_prefix, f"{ckpt_type}-{sha}"),
            )
            os.system(f"cp {ckpt_file_with_root} {file}")

    def _save_ckpt_shm(
        self,
        state,
        ckpt_dir_and_prefix,
        ckpt_type,
        async_ckpter,
        constant_data=None,
    ):
        async_ckpter.save_state_dict_to_memory(
            state, ckpt_dir_and_prefix, constant_data
        )

    def _save_disk_ckpt(
        self, ckpt_file_with_root, save_last, save_best, staging_future
    ):
        staging_future.result()
        save_shm_ckpt_to_disk(
            ckpt_file_with_root, ckpt_type=self.flash_ckpt_name
        )
        logger.info("Save model checkpoint: %s" % ckpt_file_with_root)
        if self.save_hash:
            ckpt_file_with_root = generate_sha256_file(ckpt_file_with_root)
            file_name = Path(ckpt_file_with_root).name
            sha = file_name.split(".")[0].split("-")[-1]
        else:
            sha = None

        # record checkpoint, should with hash when `save_hash` is True
        if use_elastic():
            ElasticState().update_checkpoint(ckpt_file_with_root)

        self.epoch_ckpt_files.append(ckpt_file_with_root)
        # remove old ckpt files to save the disk space
        if (
            self.keep_n is not None
            and len(self.epoch_ckpt_files) > self.keep_n
        ):
            for f in self.epoch_ckpt_files[: -self.keep_n]:
                try:
                    os.remove(f)
                except Exception as e:
                    logger.warning(
                        f"Failed to remove old checkpoint file: {f},",
                        "please check the reason.",
                    )
                    logger.warning(
                        "1. Make sure there are not two or more jobs, "
                        "which have the same path for saving models; "
                        "2. Make sure you have the permission to "
                        "manipulate the folder where the models are"
                        " saved."
                    )
                    logger.warning(str(e))
            self.epoch_ckpt_files = self.epoch_ckpt_files[-self.keep_n :]
        file_ext = ".pth.tar"
        if save_last:
            self._save_ckpt(file_ext, "last", sha, ckpt_file_with_root)

        if save_best:
            self._save_ckpt(file_ext, "best", sha, ckpt_file_with_root)

    def _save_disk_thread_join(self):
        if (
            self.save_disk_ckpt_thread is not None
            and self.save_disk_ckpt_thread.is_alive()
        ):
            self.save_disk_ckpt_thread.join()

    @rank_zero_only
    def save_to_file(
        self,
        state,
        epoch_id,
        save_best,
        save_epoch_or_step=True,
        save_type="epoch",
        save_last=True,
        step_id=None,
        write_disk_ckpt=False,
    ):
        try:
            os.makedirs(self.save_dir, exist_ok=True)
        except Exception:
            pass

        assert save_type in [
            "epoch",
            "step",
        ], f"save_type should be one of ['epoch', 'step'], but get {save_type}"
        if save_epoch_or_step:
            ckpt_file = os.path.join(
                self.save_dir,
                TRAIN_CHECKPOINT_FORMAT
                % (
                    self.name_prefix,
                    "step-%d" % step_id
                    if save_type == "step"
                    else "epoch-%04d" % epoch_id,
                ),
            )
            self._save_disk_thread_join()
            self._staging_future = self._staging_executor.submit(
                self._save_ckpt_shm,
                state,
                os.path.join(self.save_dir, self.name_prefix),
                self.flash_ckpt_name,
                self.async_ckpter,
                constant_data={
                    "horizon-plugin-version": state["horizon-plugin-version"]
                },
            )
            if save_best:
                if self._best_staging_future is not None:
                    self._best_staging_future.result()
                self._best_staging_future = self._best_staging_executor.submit(
                    self._save_ckpt_shm,
                    state,
                    os.path.join(self.save_dir, self.name_prefix),
                    self.best_flash_ckpt_name,
                    self.best_async_ckpter,
                    constant_data={
                        "horizon-plugin-version": state[
                            "horizon-plugin-version"
                        ]
                    },
                )
            if write_disk_ckpt:
                self.save_disk_ckpt_thread = threading.Thread(
                    target=self._save_disk_ckpt,
                    args=(
                        ckpt_file,
                        save_last,
                        save_best,
                        self._staging_future,
                    ),
                )
                self.save_disk_ckpt_thread.start()
        else:
            if (
                self.save_disk_ckpt_thread is not None
                and self.save_disk_ckpt_thread.is_alive()
            ):
                self.save_disk_ckpt_thread.join()
            if save_best:
                best_file = os.path.join(
                    self.save_dir,
                    TRAIN_CHECKPOINT_FORMAT % (self.name_prefix, "best"),
                )
                torch.save(state, best_file)
                if self.save_hash:
                    best_file = generate_sha256_file(
                        best_file, remove_old=True
                    )
                logger.info("Save last model checkpoint: %s" % best_file)

            if save_last:
                last_file = os.path.join(
                    self.save_dir,
                    TRAIN_CHECKPOINT_FORMAT % (self.name_prefix, "last"),
                )
                torch.save(state, last_file)
                if self.save_hash:
                    last_file = generate_sha256_file(
                        last_file, remove_old=True
                    )
                logger.info("Save last model checkpoint: %s" % last_file)

    def _sync_model(self, model):
        """
        Sync model's buffers that distributed on diff host.

        Gather all buffers from diff host, then merge them together.
        When the buffer of diff host has diff values, compute average of them.
        """
        if self.task_sampler is None or not self.task_sampler.is_parallel():
            return

        if not isinstance(model, nn.parallel.DistributedDataParallel):
            return
        else:
            model = model.module
            if not hasattr(model, "named_parameters_by_outname"):
                return

        with torch.no_grad():
            tasks = self.task_sampler.tasks
            # get current task params and buffers
            states = {}
            for n, b in model.named_buffers_by_outname(tasks):
                states[n] = b

            # gather all state from every ranks
            rank, world_size = get_dist_info()
            global_states = [None for _ in range(world_size)]
            all_gather_object(global_states, states)

            merged_states = {}  # {p_name => [buffer, cnt]}
            for s in global_states:
                for k, v in s.items():
                    if k not in merged_states:
                        merged_states[k] = [v]
                    else:
                        merged_states[k].append(v)

            # check and average same tensor on different ranks
            def _merge_tensor(key, tensor):
                # check all buffer in model state_dict
                if k not in merged_states:
                    logger.warning(f"tensor: {k} not updated on training.")
                    return

                v = merged_states[k]
                if len(v) == 1:
                    # set into model state dict directly
                    v = v[0]
                else:
                    v = _get_average_tensor(v)
                if tensor.shape != v.shape:
                    tensor.resize_(v.shape)
                tensor.copy_(v)

            # check and average same buffer on different ranks
            for k, buff in model.named_buffers():
                _merge_tensor(k, buff)

            # sync params if used hierarchical ddp
            maybe_sync_params(model)

    def save_checkpoint(
        self,
        model,
        optimizer,
        save_best,
        storage,
        ema_model=None,
        save_epoch_or_step=True,
        save_type="epoch",
        save_last=True,
        epoch_id=None,
        step_id=None,
        write_disk_ckpt=False,
    ):
        self._sync_model(model)
        try:
            grad_scaler_state = storage.get("grad_scaler")[0]
        except Exception:
            grad_scaler_state = {}
        try:
            from horizon_plugin_pytorch import __version__
        except ImportError:
            __version__ = "unknown"

        contiguous_model = model.to(memory_format=torch.contiguous_format)

        state = {
            "epoch": epoch_id,
            "step": step_id,
            "devices": get_device_count(),
            "grad_scaler": grad_scaler_state,
            "state_dict": get_valid_state_dict(
                contiguous_model, self.only_save_ddp
            ),
            "horizon-plugin-version": __version__,
        }
        if ema_model is not None:
            ema_c_model = ema_model.to(memory_format=torch.contiguous_format)
            state["ema_model"] = get_valid_state_dict(
                ema_c_model, self.only_save_ddp
            )

        if deepspeed is not None and isinstance(
            contiguous_model, deepspeed.DeepSpeedEngine
        ):
            tag = "%s-ds-ckpt-%s" % (
                self.name_prefix,
                "step-%d" % step_id
                if save_type == "step"
                else "epoch-%04d" % epoch_id,
            )
            contiguous_model.save_checkpoint(
                save_dir=self.save_dir,
                tag=tag,
                client_state=state,
            )
            if save_last:
                tag = "ds-last-ckpt"
                contiguous_model.save_checkpoint(
                    save_dir=self.save_dir,
                    tag=tag,
                    client_state=state,
                )
            if save_best:
                tag = "ds-best-ckpt"
                contiguous_model.save_checkpoint(
                    save_dir=self.save_dir,
                    tag=tag,
                    client_state=state,
                )
        else:
            if optimizer is not None:
                state["optimizer"] = optimizer.state_dict()
            else:
                state["optimizer"] = None

            self.save_to_file(
                state,
                epoch_id,
                save_best,
                save_epoch_or_step,
                save_type,
                save_last,
                step_id,
                write_disk_ckpt,
            )

    def _get_ckp_model(self, model, ema_model):
        ckp_model = model
        if ema_model is not None:
            ckp_model = ema_model
        return ckp_model

    def on_loop_begin(self, loop, **kwargs):
        if self.best_refer_metric is not None:
            for m in self.best_refer_metric:
                m.to(loop.device)

    def on_step_end(
        self,
        epoch_id,
        global_step_id,
        val_metrics,
        storage=None,
        model=None,
        ema_model=None,
        optimizer=None,
        **kwargs,
    ):
        if self.interval_by == "step" and (
            (global_step_id + 1)
            % int(np.ceil(self.save_interval / self.shm_save_freq_factor))
            == 0
        ):
            self.do_checkpoint(
                model,
                optimizer,
                epoch_id,
                ema_model=ema_model,
                save_type="step",
                step_id=global_step_id,
                val_metrics=val_metrics,
                storage=storage,
                write_disk_ckpt=(global_step_id + 1) % self.save_interval == 0,
            )

    def on_epoch_end(
        self,
        epoch_id,
        global_step_id,
        model,
        optimizer,
        val_metrics,
        storage=None,
        ema_model=None,
        **kwargs,
    ):
        if self.interval_by == "epoch" and (
            (epoch_id + 1) % self.save_interval == 0
        ):
            self.do_checkpoint(
                model,
                optimizer,
                epoch_id,
                ema_model=ema_model,
                save_type="epoch",
                step_id=global_step_id,
                val_metrics=val_metrics,
                storage=storage,
                write_disk_ckpt=True,
            )

    def do_checkpoint(
        self,
        model,
        optimizer,
        epoch_id,
        storage,
        ema_model=None,
        save_type="epoch",
        step_id=None,
        val_metrics=None,
        write_disk_ckpt=False,
    ):
        if self.mode is not None:
            if self.best_refer_metric is not None:
                metrics = self.best_refer_metric
            else:
                assert val_metrics is not None
                metrics = val_metrics
            value = None
            for val_metric in metrics:
                if hasattr(val_metric, "fast_get"):
                    names, values = val_metric.fast_get()
                else:
                    names, values = val_metric.get()
                if self.monitor_metric_key is None:
                    values = _as_list(values)
                    if len(values) != 1:
                        raise KeyError(
                            "Cannot resolve more than one metric values"
                            + "while monitor_metric_key is None."
                        )
                    value = values[0]
                    break
                else:
                    if self.monitor_metric_key in names:
                        index = _as_list(names).index(self.monitor_metric_key)
                        value = _as_list(values)[index]
                        break
            assert value is not None

            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            is_best = self.monitor_op(value, self.best)
            if is_best:
                logger.info("former best:%f, current:%f" % (self.best, value))
                self.best = value
            # DeepSpeed save_checkpoint is collective: every rank must agree
            # on save_best, otherwise ranks that skip ds-best-ckpt hang forever.
            rank, world_size = get_dist_info()
            if world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
                flag = torch.tensor(
                    [1 if is_best else 0],
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                )
                torch.distributed.broadcast(flag, src=0)
                is_best = bool(flag.item())
                if is_best:
                    best_t = torch.tensor(
                        [float(self.best)],
                        device=flag.device,
                    )
                    torch.distributed.broadcast(best_t, src=0)
                    self.best = float(best_t.item())
        else:
            is_best = False

        # save train model
        self.save_checkpoint(
            model,
            ema_model=ema_model,
            epoch_id=epoch_id,
            optimizer=optimizer,
            save_best=is_best,
            save_type=save_type,
            step_id=step_id,
            storage=storage,
            write_disk_ckpt=write_disk_ckpt,
        )

    def on_loop_end(
        self,
        model,
        optimizer,
        epoch_id,
        global_step_id,
        storage=None,
        ema_model=None,
        **kwargs,
    ):
        if not self.save_on_train_end:
            return

        ckp_model = self._get_ckp_model(model, ema_model)

        self.save_checkpoint(
            ckp_model,
            epoch_id=epoch_id,
            step_id=global_step_id,
            optimizer=optimizer,
            save_best=False,
            save_epoch_or_step=False,
            save_last=True,
            storage=storage,
        )
        self.async_ckpter.close()
        self.best_async_ckpter.close()

    def on_optimizer_step_begin(self, **kwargs):
        if self._staging_future is not None:
            self._staging_future.result()

    def __del__(self):
        self._save_disk_thread_join()


@OBJECT_REGISTRY.register
class DistributedCheckpoint(Checkpoint):
    """
    Checkpoint Callback is used for saving model after training.

    DistributedCheckpoint Callback is used for saving model after
    training and resume model before training as the same times
    in distributed environment.

    Args:
        save_dir: Directory to save checkpoints.
        name_prefix: Checkpoint name prefix.
        save_interval: Save checkpoint every `save_interval` epoch or step.
        interval_by: Set `save_interval` unit to step or epoch.
            Default is epoch.
        save_on_train_end: Whether save checkpoint when `on_loop_end` is
            triggered.
        strict_match: Whether to strictly enforce that the keys in
            `model.state_dict()` (train model) match the keys in
            `test_model.state_dict()`. Default: ``False``
        mode: State of monitor for saving model.
        monitor_metric_key: Monitor metric for saving best checkpoint.
        best_refer_metric: Metric that evaluate which epoch is the best.
        save_hash: Whether to save the hash value to the name of the
             Checkpoint file. Default is True.
        only_save_ddp: Only save ddp module. Default is False.
        keep_n: Keep the last n checkpoints. Default is None.
        enable_async_save: Whether to enable async save. Default is False.
        pinned_memory_buffer: Whether to use pinned memory buffer
            to speed up checkpointing. Default is False.
        convert_to_torch_save: Whether to convert dcp checkpoint to
            `torch.save` checkpoint format. Default is True.
    """

    @require_packages("torch>=2.3")
    def __init__(
        self,
        save_dir: str,
        name_prefix: Optional[str] = "",
        save_interval: Optional[int] = 1,
        interval_by: Optional[str] = "epoch",
        save_on_train_end: Optional[bool] = True,
        strict_match: Optional[bool] = False,
        mode: Optional[str] = None,
        monitor_metric_key: Optional[str] = None,
        best_refer_metric: Optional[Union[dict, EvalMetric]] = None,
        task_sampler=None,
        save_hash: bool = True,
        only_save_ddp: bool = False,
        keep_n: Optional[int] = None,
        enable_async_save: bool = False,
        pinned_memory_buffer: bool = False,
        convert_to_torch_save: bool = True,
    ):
        super().__init__(
            save_dir,
            name_prefix,
            save_interval,
            interval_by,
            save_on_train_end,
            strict_match,
            mode,
            monitor_metric_key,
            best_refer_metric,
            task_sampler,
            save_hash,
            only_save_ddp,
            keep_n,
        )
        self.enable_async_save = enable_async_save
        if self.enable_async_save:
            assert LooseVersion(torch.__version__) >= LooseVersion(
                "2.4.0"
            ), "Async save is only supported in torch >= 2.4.0"
        self.pinned_memory_buffer = pinned_memory_buffer
        self.convert_to_torch_save = convert_to_torch_save
        self.checkpoint_future = None
        if self.enable_async_save and self.pinned_memory_buffer:
            self.writer = StorageWriter(cached_state_dict=True)
        self.dcp_to_torch_save_thread_pool = ThreadPoolExecutor(max_workers=1)

    @rank_zero_only
    @require_packages("torch>=2.3")
    def _dcp_to_torch_save(
        self, dcp_ckpt_dir: str, ckpt_file: str, remove_old: bool = False
    ):
        def dcp_to_torch_save_threading(
            dcp_ckpt_dir, ckpt_file, save_hash, remove_old
        ):
            dcp_to_torch_save(dcp_ckpt_dir, ckpt_file)
            if save_hash:
                ckpt_file = generate_sha256_file(
                    ckpt_file, remove_old=remove_old
                )

        self.dcp_to_torch_save_thread_pool.submit(
            dcp_to_torch_save_threading,
            dcp_ckpt_dir,
            ckpt_file,
            self.save_hash,
            remove_old,
        )
        logger.info(
            "Sumit threading convert DCP checkpoint to torch.save"
            " checkpoint file: %s" % ckpt_file
        )
        return ckpt_file

    def _save_handle(
        self,
        state_dict: Dict[str, Any],
        step_id: int,
        epoch_id: int,
        save_type: str,
        remove_old: bool = False,
    ):
        save_type_str = ""
        if save_type == "step":
            save_type_str = "step-%d" % step_id
        elif save_type == "epoch":
            save_type_str = "epoch-%04d" % epoch_id
        else:
            save_type_str = save_type

        dcp_ckpt_dir = os.path.join(
            self.save_dir,
            TRAIN_DCP_CHECKPOINT_FORMAT
            % (
                self.name_prefix,
                save_type_str,
            ),
        )

        if self.enable_async_save:
            if self.checkpoint_future is not None:
                # waits for checkpointing to finish, avoiding queuing more then
                # one checkpoint request at a time
                self.checkpoint_future.result()
            dcp.async_save(
                state_dict,
                storage_writer=self.writer
                if self.pinned_memory_buffer
                else None,
                checkpoint_id=dcp_ckpt_dir,
            )
        else:
            dcp.save(state_dict, checkpoint_id=dcp_ckpt_dir)
        if self.convert_to_torch_save:
            ckpt_file = os.path.join(
                self.save_dir,
                TRAIN_CHECKPOINT_FORMAT
                % (
                    self.name_prefix,
                    save_type_str,
                ),
            )
            ckpt_file = self._dcp_to_torch_save(
                dcp_ckpt_dir, ckpt_file, remove_old
            )
        return dcp_ckpt_dir

    def save_checkpoint(
        self,
        model,
        optimizer,
        save_best,
        storage,
        ema_model=None,
        save_epoch_or_step=True,
        save_type="epoch",
        save_last=True,
        epoch_id=None,
        step_id=None,
        write_disk_ckpt=False,
    ):
        try:
            grad_scaler_state = storage.get("grad_scaler")[0]
        except Exception:
            grad_scaler_state = {}
        try:
            from horizon_plugin_pytorch import __version__
        except ImportError:
            __version__ = "unknown"

        contiguous_model = model.to(memory_format=torch.contiguous_format)

        state_dict = {
            "epoch": epoch_id,
            "step": step_id,
            "devices": get_device_count(),
            "grad_scaler": grad_scaler_state,
            "state_dict": get_valid_state_dict(
                contiguous_model, self.only_save_ddp
            ),
            "horizon-plugin-version": __version__,
        }
        if ema_model is not None:
            ema_c_model = ema_model.to(memory_format=torch.contiguous_format)
            state_dict["ema_model"] = get_valid_state_dict(
                ema_c_model, self.only_save_ddp
            )
        if optimizer is not None:
            state_dict["optimizer"] = optimizer.state_dict()
        else:
            state_dict["optimizer"] = None

        try:
            os.makedirs(self.save_dir, exist_ok=True)
        except Exception:
            pass

        assert save_type in [
            "epoch",
            "step",
        ], f"save_type should be one of ['epoch', 'step'], but get {save_type}"
        if save_epoch_or_step:
            ckpt_file = self._save_handle(
                state_dict, step_id, epoch_id, save_type
            )
            logger.info("Save model checkpoint: %s" % ckpt_file)

            if self.convert_to_torch_save:
                self.epoch_ckpt_files.append(ckpt_file)
                # remove old ckpt files to save the disk space
                if (
                    self.keep_n is not None
                    and len(self.epoch_ckpt_files) > self.keep_n
                ):
                    for f in self.epoch_ckpt_files[: -self.keep_n]:
                        try:
                            os.remove(f)
                        except Exception as e:
                            logger.warning(
                                f"Failed to remove old checkpoint file: {f},",
                                "please check the reason.",
                            )
                            logger.warning(
                                "1. Make sure there are not two or more jobs, "
                                "which have the same path for saving models; "
                                "2. Make sure you have the permission to "
                                "manipulate the folder where the models"
                                " are saved."
                            )
                            logger.warning(str(e))
                    self.epoch_ckpt_files = self.epoch_ckpt_files[
                        -self.keep_n :
                    ]

        if save_last:
            last_file = self._save_handle(
                state_dict, step_id, epoch_id, "last", remove_old=True
            )
            logger.info("Save last model checkpoint: %s" % last_file)

        if save_best:
            best_file = self._save_handle(
                state_dict, step_id, epoch_id, "best", remove_old=True
            )
            logger.info(
                "DCP do not support only save EMA model. Please check load"
                " the correct state_dict by yourself. "
                "Save best model checkpoint: %s" % best_file
            )

    def on_loop_end(
        self,
        model,
        optimizer,
        epoch_id,
        global_step_id,
        storage=None,
        ema_model=None,
        **kwargs,
    ):
        res = super().on_loop_end(
            model,
            optimizer,
            epoch_id,
            global_step_id,
            storage,
            ema_model,
            **kwargs,
        )
        self.dcp_to_torch_save_thread_pool.shutdown()
        return res
