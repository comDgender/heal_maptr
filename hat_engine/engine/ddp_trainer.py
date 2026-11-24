# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
# type: ignore

import logging
import os
import signal
import time
import traceback
from datetime import timedelta
from decimal import localcontext
from distutils.version import LooseVersion
from functools import wraps
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from hat_engine.callbacks import CallbackMixin
from hat_engine.data.samplers.task_sampler import TaskSampler
from hat_engine.models.base_modules.sync_bn import SyncBatchNorm
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.apply_func import _as_list, to_cuda
from hat_engine.utils.deprecate import deprecated_warning
from hat_engine.utils.logger import rank_zero_info
from hat_engine.utils.traceback_monitor import init_traceback_mp_monitor

# isort: off
from hat_engine.utils.distributed import (
    find_free_port,
    get_local_host,
    used_split_subprocess_group,
    collect_host_rank_map,
    dist_initialized,
)
from hat_engine.utils.saved_tensor import (
    checkpoint_with_saved_tensor,
    support_saved_tensor,
)
from hat_engine.utils.hierarchical_dist import HierarchicalPlugin
from .launcher import register_launcher
from .processors import BatchProcessorMixin
from .trainer import Trainer
from .loop_base import LoopType

try:
    from horizon_plugin_pytorch.nn.qat.batchnorm import (
        SyncBatchNorm as HorizonSyncBatchNorm,
    )
except (RuntimeError, ImportError):
    HorizonSyncBatchNorm = None


# isort: on

__all__ = ["DistributedDataParallelTrainer", "launch"]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

torch_version = torch.__version__


def convert_sync_bn(model, process_group=None, local_sync=True):

    if LooseVersion(torch_version) >= LooseVersion("1.13.0"):
        if HorizonSyncBatchNorm is not None and hasattr(
            HorizonSyncBatchNorm, "convert_sync_batchnorm"
        ):
            logger.warning(
                f"Detected that `torch=={torch_version}`, "
                f"and `horizon_plugin_pytorch.nn.batchnorm.SyncBatchNorm` "
                f"will be used to replace `torch.nn.SyncBatchNorm`, which "
                f"will be faster during training."
            )
            return HorizonSyncBatchNorm.convert_sync_batchnorm(
                model, process_group
            )
        logger.warning(
            f"Detected that `torch=={torch_version}`, "
            f"and `hat_engine.models.SyncBatchNorm `will be used "
            f"to replace `torch.nn.SyncBatchNorm`, which will be "
            f"faster during training."
        )
        return SyncBatchNorm.convert_sync_batchnorm(model, process_group)
    else:
        return nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group)


class CustomDistributedDataParallel(DDP):
    """Implements custom distributed data parallelism that is based on \
    torch.nn.parallel.DistributedDataParallel.

    When buffers are reassigned by user module like this or not,
    https://gist.github.com/rohan-varma/5d599284b632f4abdcf633e720aae9d7#file-ddp_issue-py-L23,
    User can decide whether to use assign_module_buffers.
    See:
    1. https://github.com/pytorch/pytorch/issues/63916
    2. https://github.com/pytorch/pytorch/pull/64472.
    """

    def __init__(
        self,
        module: nn.Module,
        assign_module_buffers: bool = True,
        ignore_qat_buffers: bool = False,
        splited_subprocess_group: Optional[dist.ProcessGroup] = None,
        *args,
        **kwargs,
    ):
        self.assign_module_buffers = assign_module_buffers
        self.splited_subprocess_group = splited_subprocess_group
        self.ignore_qat_buffers = ignore_qat_buffers

        if (
            self.splited_subprocess_group is not None
            and used_split_subprocess_group()
        ):
            self._reset_process_group = True
        else:
            self._reset_process_group = False
        self.qat_buffers_ignore = set()
        if self.ignore_qat_buffers:
            for name, _ in module.named_buffers():
                if (
                    "activation_post_process.eps" in name
                    or "fake_quant_enabled" in name
                    or "observer_enabled" in name
                    or ".scale" in name
                    or ".zero_point" in name
                    or ".qdtype_id" in name
                ):
                    self.qat_buffers_ignore.add(name)
        super(CustomDistributedDataParallel, self).__init__(
            module, *args, **kwargs
        )

    def forward(self, *inputs, **kwargs):
        if self._reset_process_group:
            self.process_group = self.splited_subprocess_group
        return super().forward(*inputs, **kwargs)

    # override torch1.10.2
    def _sync_params(self):
        with torch.no_grad():
            if self.will_sync_module_buffers():
                if self._join_config.enable:
                    authoritative_rank = self._find_common_rank(
                        self._distributed_rank, True
                    )
                else:
                    authoritative_rank = 0
                # assign module buffers is optional
                if self.assign_module_buffers:
                    self._assign_modules_buffers()
                self._distributed_broadcast_coalesced(
                    self.modules_buffers,
                    self.broadcast_bucket_size,
                    authoritative_rank,
                )

    # override torch1.13.0
    def _sync_buffers(self):
        with torch.no_grad():
            if self._join_config.enable:
                authoritative_rank = self._find_common_rank(
                    self._distributed_rank, True
                )
            else:
                authoritative_rank = 0

            # assign module buffers is optional
            if self.assign_module_buffers:
                self._assign_modules_buffers()
            self._sync_module_buffers(authoritative_rank)

    def _assign_modules_buffers(self):
        if not self.ignore_qat_buffers:
            return super()._assign_modules_buffers()

        # Collect buffers for modules, filtering out buffers that
        # should be ignored.
        named_module_buffers = [
            (buffer, buffer_name)
            for buffer_name, buffer in self.module.named_buffers()
            if (
                buffer_name not in self.parameters_to_ignore
                and buffer_name not in self.qat_buffers_ignore
            )
        ]
        self.modules_buffers = [
            buffer for (buffer, buffer_name) in named_module_buffers
        ]
        # Dict[str, tensor] representing module buffers not ignored by DDP.
        self.named_module_buffers = {
            buffer_name: buffer
            for (buffer, buffer_name) in named_module_buffers
        }


@OBJECT_REGISTRY.register
@OBJECT_REGISTRY.alias("distributed_data_parallel_trainer")
class DistributedDataParallelTrainer(Trainer):
    """DistributedDataParallelTrainer tool.

    DistributedDataParallelTrainer is a tool function to new a `Trainer`
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
        hierarchical_plugin: HierarchicalPlugin config for hierarchical sync,
            see `hat_engine.utils.hierarchical_dist.HierarchicalPlugin`.
        train_metrics: Metrics on training data.
        val_metrics: Metrics on validation data.
        profiler: To profile individual steps during training and
            assist in identifying bottlenecks.
        task_sampler: TaskSampler config for multitask training.
        convert_submodule_list: List of submodule for converting DDP.
        assign_module_buffers: Whether reassign modules buffers. For details,
            see: https://horizonrobotics.feishu.cn/wiki/wikcn5j6CW9MVemYwnl1n349JMf#GAyJdS  # noqa: E501
        ignore_qat_buffers: Whether ignore qat buffers to speed up.
        find_unused_parameters: Args of DistributedDataParallel module.
        compiler: Converter of `torch.compile`.
    """

    def __init__(
        self,
        model: nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        batch_processor: BatchProcessorMixin,
        device: int,
        stop_by: Optional[str] = "epoch",
        num_epochs: Optional[int] = None,
        start_epoch: Optional[int] = 0,
        num_steps: Optional[int] = None,
        start_step: Optional[int] = 0,
        callbacks: Optional[Sequence[CallbackMixin]] = None,
        sync_bn: Optional[bool] = False,
        sync_bn_by_host: Optional[bool] = False,
        hierarchical_plugin: Optional[HierarchicalPlugin] = None,
        train_metrics: Optional[dict] = None,
        val_metrics: Optional[dict] = None,
        profiler: Optional[dict] = None,
        task_sampler: Optional[TaskSampler] = None,
        convert_submodule_list: Optional[List[str]] = None,
        find_unused_parameters: Optional[bool] = True,
        assign_module_buffers: Optional[bool] = True,
        ignore_qat_buffers: Optional[bool] = False,
        compiler: Optional[Dict] = None,
        **kwargs,
    ):  # noqa: D205,D400
        super(DistributedDataParallelTrainer, self).__init__(
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            batch_processor=batch_processor,
            device=device,
            stop_by=stop_by,
            num_epochs=num_epochs,
            start_epoch=start_epoch,
            num_steps=num_steps,
            start_step=start_step,
            callbacks=callbacks,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            profiler=profiler,
            compiler=None,
            hierarchical_plugin=hierarchical_plugin,
            loop_type=LoopType.TRAIN,
            **kwargs,
        )
        assert isinstance(self.device, int), (
            "%s, run `DistributedDataParallel` model"
            " only on one gpu" % type(self.device)
        )
        current_device = torch.cuda.current_device()
        assert current_device == self.device, "%d vs. %d" % (
            current_device,
            self.device,
        )
        self.sync_bn = sync_bn
        self.sync_bn_by_host = sync_bn_by_host

        self.model.cuda(self.device)
        # move optimizer to cuda
        if isinstance(self.optimizer, torch.optim.Optimizer):
            to_cuda(self.optimizer, self.device, inplace=True)
        assert not isinstance(
            self.model, nn.parallel.DistributedDataParallel
        ), "is already a `DistributedDataParallel` instance"

        if sync_bn and sync_bn_by_host:
            deprecated_warning(
                author="mengyang.duan",
                deprecation_version="2.3.0",
                removal_version="2.6.0",
                old_name="sync_bn_by_host",
                new_name="hierarchical_plugin",
            )
            if hierarchical_plugin is None:
                hierarchical_plugin = HierarchicalPlugin(
                    num_host_per_group=1,
                    group_sync_strategy="syncbn_only",
                )

        subprocess_group = None
        if hierarchical_plugin is not None:
            subprocess_group = hierarchical_plugin.create_subgroups()

        if sync_bn:
            self.model = convert_sync_bn(
                self.model,
                process_group=subprocess_group,
            )
        if support_saved_tensor():
            checkpoint_with_saved_tensor(self.model)

        if bool(int(os.environ.get("HAT_USE_CUDAGRAPH", "0"))):
            assert (
                not find_unused_parameters
            ), "Cannot find unused parameter while cuda-graph."

        broadcast_buffers = True
        if task_sampler is not None and task_sampler.is_parallel():
            # task parallel must set find unused parameters to True
            find_unused_parameters = True
            broadcast_buffers = False

        if batch_processor.enable_channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        if convert_submodule_list:
            # support to convert DDP with submodule
            convert_submodule_list = _as_list(convert_submodule_list)
            for submodule in convert_submodule_list:
                module = getattr(self.model, submodule)
                if LooseVersion(torch_version) >= LooseVersion("1.10.2"):
                    module = CustomDistributedDataParallel(
                        assign_module_buffers=assign_module_buffers,
                        ignore_qat_buffers=ignore_qat_buffers,
                        module=module,
                        find_unused_parameters=find_unused_parameters,
                        broadcast_buffers=broadcast_buffers,
                        device_ids=[device],
                        splited_subprocess_group=subprocess_group,
                    )
                else:
                    module = nn.parallel.DistributedDataParallel(
                        module=module,
                        find_unused_parameters=find_unused_parameters,
                        broadcast_buffers=broadcast_buffers,
                        device_ids=[device],
                    )
                setattr(self.model, submodule, module)
        else:
            cuda_graph = bool(int(os.environ.get("HAT_USE_CUDAGRAPH", "0")))
            stream_context = (
                torch.cuda.stream(torch.cuda.Stream())
                if cuda_graph
                else localcontext()
            )
            with stream_context:
                if LooseVersion(torch_version) >= LooseVersion("1.10.2"):
                    self.model = CustomDistributedDataParallel(
                        assign_module_buffers=assign_module_buffers,
                        ignore_qat_buffers=ignore_qat_buffers,
                        module=self.model,
                        find_unused_parameters=find_unused_parameters,
                        broadcast_buffers=broadcast_buffers,
                        device_ids=[self.device],
                        splited_subprocess_group=subprocess_group,
                    )
                else:
                    self.model = nn.parallel.DistributedDataParallel(
                        module=self.model,
                        find_unused_parameters=find_unused_parameters,
                        broadcast_buffers=broadcast_buffers,
                        device_ids=[device],
                    )

        if hierarchical_plugin is not None:
            self.model = hierarchical_plugin.register_comm_hook_to_model(
                model=self.model,
                process_group=subprocess_group,
            )

        if compiler:
            self.model = compiler(self.model)

        # save training strategy
        self.dump_strategy()
        self.dump_yaml()
        self.upload_strategy()

    def dump_strategy(self):
        # dump train strategy
        strategy = {}
        strategy["sync_bn"] = self.sync_bn
        strategy["sync_bn_by_host"] = self.sync_bn_by_host
        strategy["checkpoint"] = bool(
            int(os.environ.get("HAT_USE_CHECKPOINT", "0"))
        )
        strategy["saved_tensor"] = bool(
            int(os.environ.get("HAT_USE_SAVEDTENSOR", "0"))
        )
        strategy["amp"] = self.batch_processor.enable_amp
        strategy["amp_dtype"] = str(self.batch_processor.enable_amp_dtype)
        strategy["channels_last"] = str(
            self.batch_processor.enable_channels_last
        )
        strategy["grad_accumulation"] = self.batch_processor.ga_step

        for key, value in strategy.items():
            self.strategy.append(
                {"name": str(key), "usage_detail": str(value)}
            )


def launch(
    main_func,
    device_ids,
    dist_url="auto",
    dist_launcher=None,
    num_processes=None,
    backend="NCCL",
    args=(),
):
    if device_ids is not None:
        device_ids = _as_list(device_ids)
        num_devices = len(device_ids)
        assert num_devices > 0

        num_processes = num_processes if num_processes else num_devices
        assert num_processes > 0
        if num_processes == num_devices and backend != "NCCL":
            logger.warning(
                "NCCL is the best choice in case of single "
                "process on single gpu."
            )

        # Note: if device_ids=[1, 3], then after setting
        # `CUDA_VISIBLE_DEVICES`, new device_ids=[0, 1].
        str_ids = list(map(str, device_ids))
        if torch.version.cuda is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str_ids)
        elif torch.version.hip is not None:
            os.environ["HIP_VISIBLE_DEVICES"] = ",".join(str_ids)
    else:
        num_devices = None
        num_processes = num_processes if num_processes else 1

    if dist_url == "auto":
        port = find_free_port()
        dist_url = "tcp://localhost:%s" % port

    if dist_launcher is not None:
        if dist_launcher == "mpi":
            deprecated_warning(
                author="mengyang.duan",
                deprecation_version="1.4.1",
                removal_version="1.4.3",
                old_name="mpi",
                new_name="torch",
            )
            _main_mpi(
                main_func,
                dist_url,
                backend,
                num_devices,
                num_processes,
                args,
            )
        elif dist_launcher == "torch":
            master_port = os.getenv("MASTER_PORT", None)
            if master_port:
                logger.warning(f"DDP master port will be set to {master_port}")
            init_traceback_mp_monitor(
                ranks=[int(os.getenv("RANK"))], pids=[os.getpid()]
            )
            try:
                _main_func(
                    -1,
                    main_func,
                    None,
                    backend,
                    num_devices,
                    -1,
                    args,
                )
            except Exception as e:
                raise e
        else:
            raise TypeError("unknown dist_launcher: %s" % dist_launcher)
    else:
        try:
            context = mp.spawn(
                _main_func,
                nprocs=num_processes,
                join=False,
                args=(
                    main_func,
                    dist_url,
                    backend,
                    num_devices,
                    num_processes,
                    args,
                ),
            )
            init_traceback_mp_monitor(
                ranks=range(len(context.processes)),
                pids=[p.pid for p in context.processes],
            )
            while not context.join():
                pass
        # when press Ctrl+c, all sub processes will exits too.
        except KeyboardInterrupt as exception:
            logger.exception(str(exception))
            os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)


register_launcher("DistributedDataParallelTrainer", launch)
register_launcher("distributed_data_parallel_trainer", launch)


def collect_host_rank_info():

    try:
        glob_host_rank_map = collect_host_rank_map(
            process_group=None,
            init_local_rank_env=True,
        )
        msg = ""
        if glob_host_rank_map is not None:
            for host, ranks in glob_host_rank_map.items():
                msg += f"{host}: {list(ranks)}\n"
            rank_zero_info(
                "Collect host-rank map:\n"
                + "=" * 70
                + "\n"
                + f"{msg}"
                + "=" * 70
                + "\n"
            )
    except Exception as e:
        logger.error(f"collect host rank map get error: {str(e)}")


def _wrap(fn):
    """Different with torch.multiprocess.spawn._wrap.

    error_queue may carsh when some exceptions happens.
    """

    collect_host_rank_info()

    @wraps(fn)
    def _with_exception(*args):
        try:
            fn(*args)
            # for exit safely
            time.sleep(5)
        except KeyboardInterrupt:
            pass  # SIGINT; Killed by parent, do nothing
        except Exception as e:
            rank = os.getenv("RANK", None)
            host_name = get_local_host()
            logger.error(
                f"Host[{host_name}] Rank[{rank}] error: {traceback.format_exc()}"  # noqa E501
            )
            raise e

    return _with_exception


def _main_func(
    local_rank, main_func, dist_url, backend, num_devices, num_processes, args
):

    host_name = get_local_host()
    logger.info(
        f"Launch with rank: {os.getenv('RANK') if local_rank==-1 else local_rank} "  # noqa E501
        f"world_size: {os.getenv('WORLD_SIZE', None)} "
        f"hostname: {host_name} "
        f"dist_url: {dist_url} "
        f"num_devices: {num_devices} "
        f"num_processes: {num_processes} "
    )
    local_rank = (
        int(os.environ["LOCAL_RANK"]) if local_rank == -1 else local_rank
    )
    torch.cuda.set_device(
        local_rank % num_devices if num_devices else local_rank
    )
    dist_init_kwargs = {}
    if LooseVersion(torch_version) >= LooseVersion("2.8.0"):
        dist_init_kwargs["device_id"] = (
            local_rank % num_devices if num_devices else local_rank
        )
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    elif local_rank != -1:
        logger.info(
            "Environment 'RANK' not found, using local_rank as global rank."
        )
        rank = local_rank
    else:
        rank = None

    try:
        dist.init_process_group(
            backend=backend,
            init_method=dist_url,
            world_size=num_processes,
            rank=rank,
            timeout=timedelta(
                seconds=int(
                    os.environ.get("HAT_PROCESS_GROUP_TIMEOUT", "1800")
                )
            ),
            **dist_init_kwargs,
        )
        # Trigger NCCL initialization Immediately
        dist.barrier()
    except Exception as e:
        logger.error(
            f"init process group({os.getenv('RANK') if local_rank==-1 else local_rank}:{dist_url})"  # noqa E501
            f"error on host[{host_name}]!" + str(e)
        )
        raise e

    if num_devices is not None:
        _wrap(main_func)(local_rank % num_devices, *args)
    else:
        _wrap(main_func)(None, *args)

    # destroy_process_group when torch version >=2.6.0
    if (
        LooseVersion(torch_version) >= LooseVersion("2.6.0")
        and dist_initialized()
    ):
        dist.destroy_process_group()


def _main_mpi(main_func, dist_url, backend, num_devices, num_processes, args):
    import mpi4py.MPI as MPI

    comm = MPI.COMM_WORLD
    local_rank = comm.Get_rank()
    world_size = comm.Get_size()
    host_ip = get_local_host()  # noqa: F841
    logger.info(
        f"MPI launch with rank: {local_rank} world_size: {world_size} "
        f"ip: {host_ip} dist_url: {dist_url} "
        f"num_devices: {num_devices} num_processes: {num_processes} "
    )
    dist_init_kwargs = {}
    if LooseVersion(torch_version) >= LooseVersion("2.8.0"):
        dist_init_kwargs["device_id"] = (
            local_rank % num_devices if num_devices else local_rank
        )
    try:
        dist.init_process_group(
            backend=backend,
            init_method=dist_url,
            world_size=world_size,
            rank=local_rank,
            timeout=timedelta(
                seconds=int(
                    os.environ.get("HAT_PROCESS_GROUP_TIMEOUT", "1800")
                )
            ),
            **dist_init_kwargs,
        )
    except Exception as e:
        logger.error(
            f"init process group({local_rank}) error on host[{host_ip}]!"
            + str(e)
        )
        raise e

    current_device = local_rank % num_devices
    torch.cuda.set_device(current_device)
    _wrap(main_func)(current_device, *args)

    # destroy_process_group when torch version >=2.6.0
    if (
        LooseVersion(torch_version) >= LooseVersion("2.6.0")
        and dist_initialized()
    ):
        dist.destroy_process_group()
