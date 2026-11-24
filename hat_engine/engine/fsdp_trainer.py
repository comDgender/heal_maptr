# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
# type: ignore

import logging
import os
from distutils.version import LooseVersion
from itertools import chain
from typing import Any, Dict, Iterable, List, Optional, Sequence, Type, Union

import torch
import torch.nn as nn

try:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.api import (
        BackwardPrefetch,
        CPUOffload,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy
except ImportError:
    FSDP = None
    dcp = None
    BackwardPrefetch = None
    CPUOffload = None
    MixedPrecision = None
    ShardingStrategy = None
    ShardedGradScaler = None
    ModuleWrapPolicy = None

try:
    from torch.distributed._tensor import DeviceMesh
except ImportError:
    DeviceMesh = None

try:
    from torch.distributed.fsdp._init_utils import ProcessGroupType
except ImportError:
    ProcessGroupType = None

from hat_engine.callbacks import CallbackMixin
from hat_engine.data.samplers.task_sampler import TaskSampler
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.apply_func import _as_list
from hat_engine.utils.distributed import get_dist_info
from hat_engine.utils.elastic import ElasticState, elastic_need_resume
from hat_engine.utils.event import EventStorage
from hat_engine.utils.global_var import get_value, set_value
from hat_engine.utils.package_helper import require_packages
from hat_engine.utils.parallel_dims import ParallelDims

# isort: off
from hat_engine.utils.distributed import (
    get_local_process_group,
)
from hat_engine.utils.saved_tensor import (
    checkpoint_with_saved_tensor,
    support_saved_tensor,
)
from .launcher import register_launcher
from .processors import BatchProcessorMixin
from .loop_base import PipeBase, LoopBase
from .ddp_trainer import (
    launch,
    DistributedDataParallelTrainer,
    convert_sync_bn,
)

# isort: on

__all__ = ["FullyShardedDataParallelTrainer"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

torch_version = torch.__version__


@OBJECT_REGISTRY.register
@OBJECT_REGISTRY.alias("full_sharded_data_parallel_trainer")
class FullyShardedDataParallelTrainer(
    DistributedDataParallelTrainer, PipeBase
):
    """FullyShardedDataParallelTrainer tool.

    FullyShardedDataParallelTrainer is a tool function to new a `Trainer`
    instance, which training with `DistributedDataParallel` method,
    and running on one of the GPU devices.

    It can be launched by launch function below, which spawns multiple
    processes and each of it owns an independent Trainer.

    By setting `stop_by`, you are able to stop training by counting epoch
    (default) or step.

    Args:
        model: Model config or a `nn.Module` instance.
        data_loader: Training data loader config or a instantiated data loader.
        optimizer: Optimizer config or a optimizer instance.
        batch_processor: Batch processor config or a `BatchProcessorMixin`
            instance.
        device: GPU id.
        model_convert_pipeline: Define the process of model convert.
            e.g. convert float model to qat model, convert qat model
            to quantize model.
        resume_optimizer: whether load optimizer dict when resume checkpoint.
        resume_epoch_or_step: whether need to resume epoch_or_step
            when resume checkpoint.
        resume_dataloader: whether to resume dataloader index.
            Only effective when `stop_by=='step'`.
        checkpoint_dir: The directory to load checkpoint.
        stop_by: Stop training by counting epoch or step.
            If equal to 'epoch', stop training when
            `epoch_id == num_epochs - 1`.
            If equal to 'step', stop training when
            `global_step_id == num_steps - 1`.
            Default 'epoch'.
        num_epochs: Num of training epochs, should be non-negative integer.
            If stop_by != 'epoch', no-op.
            Set 0 to skip training and run `self.on_loop_begin/end` only.
        start_epoch: Training start epoch, should be non-negative integer.
        num_steps: Num of training steps, should be non-negative integer.
            If stop_by != 'step', no-op.
            Set 0 to skip training and run `self.on_loop_begin/end` only.
        start_step: Training start step, should be non-negative integer.
        callbacks: Callback configs or instances.
        sync_bn: Whether to convert bn to sync_bn.
        sync_bn_by_host: Whether sync bn within host node
        train_metrics: Metrics on training data.
        val_metrics: Metrics on validation data.
        profiler: To profile individual steps during training and
            assist in identifying bottlenecks.
        task_sampler: TaskSampler config for multitask training.
        find_unused_parameters: Args of DistributedDataParallel module.
        compiler: Converter of `torch.compile`.
        sharding_strategy: This configures the sharding strategy,
            which may trade off memory saving and communication overhead.
            See ShardingStrategy for details. (Default: "full_shard")
        cpu_offload: This configures CPU offloading. If this is set to False,
            then no CPU offloading happens. (Default: False)

        Other params reference to apis of PyTorch's FSDP.
    """

    @require_packages("torch>=2.0")
    def __init__(
        self,
        model: nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        batch_processor: BatchProcessorMixin,
        device: int,
        model_convert_pipeline: Optional[Union[Dict, List]] = None,
        resume_optimizer: bool = False,
        resume_epoch_or_step: bool = False,
        resume_dataloader: bool = False,
        checkpoint_dir: Optional[str] = None,
        stop_by: Optional[str] = "epoch",
        num_epochs: Optional[int] = None,
        start_epoch: Optional[int] = 0,
        num_steps: Optional[int] = None,
        start_step: Optional[int] = 0,
        callbacks: Optional[Sequence[CallbackMixin]] = None,
        sync_bn: Optional[bool] = False,
        sync_bn_by_host: Optional[bool] = False,
        train_metrics: Optional[dict] = None,
        val_metrics: Optional[dict] = None,
        profiler: Optional[dict] = None,
        log_interval: int = 0,
        task_sampler: Optional[TaskSampler] = None,
        find_unused_parameters: Optional[bool] = True,
        compiler: Optional[Dict] = None,
        model_convert_pipeline_position: Optional[str] = "after_fsdp",
        sharding_strategy: str = "full_shard",
        cpu_offload: bool = False,
        auto_wrap_policy: Optional[ModuleWrapPolicy] = None,
        backward_prefetch: str = "backward_pre",
        mixed_precision_cfg: Optional[Dict[str, Any]] = None,
        param_init_fn: Optional[nn.Module] = None,
        sync_module_states: bool = False,
        forward_prefetch: bool = False,
        limit_all_gathers: bool = True,
        use_orig_params: bool = False,
        ignored_states: Optional[Iterable[nn.Parameter]] = None,
        device_mesh_cfg: Optional[Dict[str, Any]] = None,
    ):  # noqa: D205,D400
        assert torch.cuda.is_available(), (
            f"Make sure install the GPU version of torch, "
            f"but you get {torch.__version__}"
        )
        super(LoopBase, self).__init__(callbacks=callbacks, profiler=profiler)
        assert isinstance(device, int) or device is None, (
            "device should be int (gpu id) or None (run on cpu), but get %s"
            % type(device)
        )

        assert start_epoch >= 0, (
            f"{self.name} loop start epoch should be "
            f"non-negative integer, but get {start_epoch}"
        )
        assert start_step >= 0, (
            f"{self.name} loop start step should be "
            f"non-negative integer, but get {start_step}"
        )
        assert model_convert_pipeline_position in [
            "before_fsdp",
            "after_fsdp",
        ], (
            f"model_convert_pipeline_position should be 'before_fsdp' "
            f"or 'after_fsdp', but get {model_convert_pipeline_position}"
        )

        self.model = model
        self.ema_model = None
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.batch_processor = batch_processor
        self.train_metrics = _as_list(train_metrics)
        self.val_metrics = _as_list(val_metrics)
        self.device = device
        self.num_epochs = num_epochs
        self.start_epoch = start_epoch
        self.num_steps = num_steps
        self.start_step = start_step
        self.log_interval = log_interval
        self.cpu_offload = cpu_offload
        self.sharding_strategy = sharding_strategy
        self.sync_module_states = sync_module_states
        self.forward_prefetch = forward_prefetch
        self.limit_all_gathers = limit_all_gathers
        self.backward_prefetch = backward_prefetch
        self.model_convert_pipeline_position = model_convert_pipeline_position

        if model_convert_pipeline is not None:
            self.model = model_convert_pipeline(self.model)

        if elastic_need_resume():
            self.model = ElasticState.load_checkpoint(self.model)
            resume_epoch_or_step = True
            resume_optimizer = True
            resume_dataloader = True

        stop_by = stop_by.lower()
        self._stop_by_epoch = stop_by == "epoch"
        self._stop_by_step = stop_by == "step"

        self.storage = EventStorage()

        if hasattr(self.profiler, "set_model"):
            self.profiler.set_model(self.model)

        assert isinstance(self.device, int), (
            "%s, run `FullyShardedDataParallelTrainer` model"
            " only on one gpu" % type(self.device)
        )

        current_device = torch.cuda.current_device()
        assert current_device == self.device, "%d vs. %d" % (
            current_device,
            self.device,
        )
        self.sync_bn = sync_bn
        self.sync_bn_by_host = sync_bn_by_host

        assert not isinstance(
            self.model, torch.distributed.fsdp.FullyShardedDataParallel
        ), "is already a `FullyShardedDataParallel` instance"
        if sync_bn:
            self.model = convert_sync_bn(
                self.model,
                process_group=get_local_process_group(),
                local_sync=sync_bn_by_host,
            )
        if support_saved_tensor():
            checkpoint_with_saved_tensor(self.model)

        if bool(int(os.environ.get("HAT_USE_CHECKPOINT", "0"))) or (
            bool(int(os.environ.get("HAT_USE_CUDAGRAPH", "0")))
        ):
            # Set `False` when using checkpoint to enlarge batch size.
            # Or would raise bug.
            assert (
                not find_unused_parameters
            ), "Cannot find unused parameter while checkpoint or cuda-graph."

        if task_sampler is not None and task_sampler.is_parallel():
            # task parallel must set find unused parameters to True
            find_unused_parameters = True

        if batch_processor.enable_amp:
            logger.warning(
                "It is recommended to use the AMP of deepspeed in config"
                "instead of hat_engine trainer"
            )

        if batch_processor.enable_channels_last:
            raise ValueError(
                "Channels last memory format is not supported in FSDP."
                "Please set `enable_channels_last` to False."
            )

        # convert gradscaler to shared_grad_scaler
        if not isinstance(batch_processor.grad_scaler, ShardedGradScaler):
            raw_scaler = batch_processor.grad_scaler
            if LooseVersion(torch_version) < LooseVersion("2.3"):
                if batch_processor.enable_amp:
                    batch_processor.grad_scaler = ShardedGradScaler(
                        init_scale=raw_scaler._init_scale,
                        growth_factor=raw_scaler._growth_factor,
                        backoff_factor=raw_scaler._backoff_factor,
                        growth_interval=raw_scaler._growth_interval,
                        enabled=raw_scaler._enabled,
                    )
                else:
                    batch_processor.grad_scaler = ShardedGradScaler(
                        enabled=raw_scaler._enabled,
                    )
            else:
                if batch_processor.enable_amp:
                    batch_processor.grad_scaler = ShardedGradScaler(
                        device=raw_scaler._device,
                        init_scale=raw_scaler._init_scale,
                        growth_factor=raw_scaler._growth_factor,
                        backoff_factor=raw_scaler._backoff_factor,
                        growth_interval=raw_scaler._growth_interval,
                        enabled=raw_scaler._enabled,
                    )
                else:
                    batch_processor.grad_scaler = ShardedGradScaler(
                        device=raw_scaler._device,
                        enabled=raw_scaler._enabled,
                    )

        self.model = self.apply_fsdp(
            module=self.model,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload,
            auto_wrap_policy=auto_wrap_policy,
            backward_prefetch=backward_prefetch,
            mixed_precision_cfg=mixed_precision_cfg,
            param_init_fn=param_init_fn,
            sync_module_states=sync_module_states,
            forward_prefetch=forward_prefetch,
            limit_all_gathers=limit_all_gathers,
            use_orig_params=use_orig_params,
            ignored_states=ignored_states,
            device_mesh_cfg=device_mesh_cfg,
        )

        if self.device is not None:
            with self.profiler.profile("set_device"):
                self.set_device(self.device)
        self.model.train()

        if callable(self.optimizer):
            self.optimizer = self.optimizer(self.model)

        if checkpoint_dir is not None:
            if checkpoint_dir.endswith(".pth") or checkpoint_dir.endswith(
                ".pth.tar"
            ):
                logger.info(
                    "Resume from torch.save fmt checkpoint"
                    f" from {checkpoint_dir}"
                )
                self.checkpoint = get_value("model_checkpoint")
                self._resume_from_checkpoint(
                    resume_epoch_or_step,
                    resume_optimizer,
                    resume_dataloader,
                )
            else:
                logger.info(
                    "Resume from dcp fmt checkpoint" f" from {checkpoint_dir}"
                )
                state_dict = {
                    "epoch": 0,
                    "step": 0,
                    "devices": self.device,
                    "grad_scaler": {},
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "horizon-plugin-version": "",
                    "ema_model": {},
                }
                dcp.load(
                    state_dict=state_dict,
                    checkpoint_id=checkpoint_dir,
                )
                set_value("model_checkpoint", state_dict)
                self.checkpoint = state_dict
                self._resume_from_checkpoint(
                    resume_epoch_or_step,
                    False,
                    resume_dataloader,
                )
        else:
            self.checkpoint = get_value("model_checkpoint")
            self._resume_from_checkpoint(
                resume_epoch_or_step,
                resume_optimizer,
                resume_dataloader,
            )

        _skip_loop = False

        if stop_by == "epoch":
            assert num_epochs is not None and self.num_epochs >= 0, (
                f"if stop {self.name} loop by counting epoch, num_epochs "
                f"should be non-negative integer, but get {num_epochs}"
            )
            if self.start_epoch >= num_epochs:
                logger.warning(
                    f"Start epoch {self.start_epoch} larger "
                    f"than num epochs {num_epochs}"
                )
                _skip_loop = True

        elif stop_by == "step":
            assert num_steps is not None and self.num_steps >= 0, (
                f"if stop {self.name} loop by counting step, num_steps "
                f"should be non-negative integer, but get {self.num_steps}"
            )
            if self.start_step >= num_steps:
                logger.warning(
                    f"Start step {self.start_step} larger than "
                    f"num steps {num_steps}"
                )
                _skip_loop = True

        else:
            raise ValueError(
                f"stop_by should be `epoch` or `step`, but get {stop_by}"
            )

        self._skip_loop = _skip_loop

        if compiler:
            assert use_orig_params, (
                "Argument `use_orig_params` must be `True`, "
                "when using `torch.compile`"
            )
            self.model = compiler(self.model)

        # save training strategy
        self.dump_strategy()
        self.dump_yaml()
        self.upload_strategy()

        self.profiler.setup(stage="train", local_rank=device)

    def set_device(self, device):
        if self.cpu_offload:
            init_device = "cpu"
        else:
            init_device = "cuda"
        self.device = device
        self.model.to_empty(device=init_device)

        for m in chain(self.train_metrics, self.val_metrics):
            if m is not None:
                m.to(device)

    def _get_sharding_strategy(
        self, sharding_strategy: str
    ) -> ShardingStrategy:
        if sharding_strategy is None:
            return None
        elif sharding_strategy == "full_shard":
            return ShardingStrategy.FULL_SHARD
        elif sharding_strategy == "shard_grad_op":
            return ShardingStrategy.SHARD_GRAD_OP
        elif sharding_strategy == "no_shard":
            return ShardingStrategy.NO_SHARD
        elif sharding_strategy == "hybrid_shard":
            return ShardingStrategy.HYBRID_SHARD
        elif sharding_strategy == "hybrid_shard_zero2":
            return ShardingStrategy._HYBRID_SHARD_ZERO2
        else:
            raise ValueError(f"Unknown sharding strategy: {sharding_strategy}")

    def _construct_process_group(
        self,
        sharding_strategy: str,
        device_mesh_cfg: Optional[Dict[str, Any]] = None,
    ) -> ProcessGroupType:

        # Todo(wujun.wen): support construct process group
        return None

    @require_packages("torch>=2.2")
    def _construct_device_mesh(
        self,
        dp_shard: int = -1,
        dp_replicate: int = 1,
        cp: int = 1,
        tp: int = 1,
        pp: int = 1,
        enable_loss_parallel: bool = False,
    ) -> DeviceMesh:
        """
        Initialize a `DeviceMesh` from ParallelDims.

        Args:
            Reference to ParallelDims in PyTorch.

        """
        _, world_size = get_dist_info()
        parallel_dims = ParallelDims(
            dp_replicate=dp_replicate,
            dp_shard=dp_shard,
            cp=cp,
            tp=tp,
            pp=pp,
            world_size=world_size,
            enable_loss_parallel=enable_loss_parallel,
        )
        device_mesh = parallel_dims.build_mesh(device_type="cuda")
        return device_mesh

    def _get_cpu_offloading_strategy(self, cpu_offload: bool) -> CPUOffload:
        return CPUOffload(offload_params=cpu_offload)

    def _get_backward_prefetch(
        self, backward_prefetch: str
    ) -> BackwardPrefetch:
        if backward_prefetch is None:
            return None
        elif backward_prefetch == "backward_pre":
            return BackwardPrefetch.BACKWARD_PRE
        elif backward_prefetch == "backward_post":
            return BackwardPrefetch.BACKWARD_POST
        else:
            raise ValueError(
                f"Unknown backward prefetch strategy: {backward_prefetch}"
            )

    def _get_mixed_precision(
        self,
        param_dtype: Optional[torch.dtype] = None,
        reduce_dtype: Optional[torch.dtype] = None,
        buffer_dtype: Optional[torch.dtype] = None,
        keep_low_precision_grads: bool = False,
        cast_forward_inputs: bool = False,
        cast_root_forward_inputs: bool = True,
        _module_classes_to_ignore: Sequence[
            Type[torch.nn.modules.module.Module]
        ] = (torch.nn.modules.batchnorm._BatchNorm,),
    ) -> MixedPrecision:
        """
        Construct configures FSDP-native mixed precision training.

        Args:
            Reference to MixedPrecision in PyTorch.
        """
        return MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
            keep_low_precision_grads=keep_low_precision_grads,
            cast_forward_inputs=cast_forward_inputs,
            cast_root_forward_inputs=cast_root_forward_inputs,
            _module_classes_to_ignore=_module_classes_to_ignore,
        )

    def apply_fsdp(
        self,
        module: nn.Module,
        sharding_strategy: str = None,
        cpu_offload: bool = False,
        auto_wrap_policy: Optional[ModuleWrapPolicy] = None,
        backward_prefetch: str = "backward_pre",
        mixed_precision_cfg: Optional[Dict[str, Any]] = None,
        param_init_fn: Optional[nn.Module] = None,
        sync_module_states: bool = False,
        forward_prefetch: bool = False,
        limit_all_gathers: bool = True,
        use_orig_params: bool = False,
        ignored_states: Optional[Iterable[nn.Parameter]] = None,
        device_mesh_cfg: Optional[Dict[str, Any]] = None,
    ):
        # construct FSDP model parameters
        sharding_strategy = self._get_sharding_strategy(sharding_strategy)
        process_group = self._construct_process_group(
            sharding_strategy, device_mesh_cfg
        )
        cpu_offload = self._get_cpu_offloading_strategy(cpu_offload)
        backward_prefetch = self._get_backward_prefetch(backward_prefetch)
        if mixed_precision_cfg is not None:
            mixed_precision = self._get_mixed_precision(**mixed_precision_cfg)
        else:
            mixed_precision = None

        if LooseVersion(torch_version) < LooseVersion("2.1"):
            model = FSDP(
                module=module,
                process_group=process_group,
                sharding_strategy=sharding_strategy,
                cpu_offload=cpu_offload,
                auto_wrap_policy=auto_wrap_policy,
                backward_prefetch=backward_prefetch,
                mixed_precision=mixed_precision,
                param_init_fn=param_init_fn,
                device_id=self.device,
                sync_module_states=sync_module_states,
                forward_prefetch=forward_prefetch,
                limit_all_gathers=limit_all_gathers,
                use_orig_params=use_orig_params,
                ignored_parameters=ignored_states,
            )
        elif LooseVersion(torch_version) < LooseVersion("2.2"):
            model = FSDP(
                module=module,
                process_group=process_group,
                sharding_strategy=sharding_strategy,
                cpu_offload=cpu_offload,
                auto_wrap_policy=auto_wrap_policy,
                backward_prefetch=backward_prefetch,
                mixed_precision=mixed_precision,
                param_init_fn=param_init_fn,
                device_id=self.device,
                sync_module_states=sync_module_states,
                forward_prefetch=forward_prefetch,
                limit_all_gathers=limit_all_gathers,
                use_orig_params=use_orig_params,
                ignored_states=ignored_states,
            )
        else:
            assert not (
                (device_mesh_cfg is not None) and (process_group is not None)
            ), "device_mesh and process_group should be mutually exclusive."
            if device_mesh_cfg is not None:
                device_mesh = self._construct_device_mesh(**device_mesh_cfg)
            else:
                device_mesh = None
            model = FSDP(
                module=module,
                process_group=process_group,
                sharding_strategy=sharding_strategy,
                cpu_offload=cpu_offload,
                auto_wrap_policy=auto_wrap_policy,
                backward_prefetch=backward_prefetch,
                mixed_precision=mixed_precision,
                param_init_fn=param_init_fn,
                device_id=self.device,
                sync_module_states=sync_module_states,
                forward_prefetch=forward_prefetch,
                limit_all_gathers=limit_all_gathers,
                use_orig_params=use_orig_params,
                ignored_states=ignored_states,
                device_mesh=device_mesh,
            )
        return model

    def dump_strategy(self):
        super().dump_strategy()
        strategy = {
            "cpu_offload": self.cpu_offload,
            "sharding_strategy": self.sharding_strategy,
            "sync_module_states": self.sync_module_states,
            "forward_prefetch": self.forward_prefetch,
            "limit_all_gathers": self.limit_all_gathers,
            "backward_prefetch": self.backward_prefetch,
        }
        for key, value in strategy.items():
            self.strategy.append(
                {"name": str(key), "usage_detail": str(value)}
            )


register_launcher("FullyShardedDataParallelTrainer", launch)
register_launcher("full_sharded_data_parallel_trainer", launch)
