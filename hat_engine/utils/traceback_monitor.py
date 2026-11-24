# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import functools
import io
import logging
import multiprocessing
import os
import subprocess
import threading
import time
from collections import deque
from pprint import PrettyPrinter
from typing import List

import torch
import torch.distributed as dist
import torch.utils._pytree as pytree
from torch.distributed.elastic.multiprocessing.redirects import (
    redirect_stderr,
    redirect_stdout,
)

from hat_engine.utils.distributed import get_dist_info, get_local_host
from hat_engine.utils.logger import LOG_DIR, get_monitor_logger

try:
    import pystack
except ImportError:
    pystack = None

try:
    from torch.utils._python_dispatch import TorchDispatchMode
except ImportError:
    TorchDispatchMode = object

__all__ = [
    "dataloader_traceback",
    "train_traceback",
    "FLIGHT_RECORDER",
    "flight_record",
    "HATNCCLHookDispatchMode",
    "setup_hat_flight_record",
]


common_logger = logging.getLogger(__name__)


def get_pystack_cmd():
    user_pystack = os.path.join(os.path.expanduser("~"), ".local/bin/pystack")
    if os.path.exists(user_pystack):
        return user_pystack
    elif pystack is not None:
        return "pystack"
    else:
        raise ImportError(
            "Please install pystack to get traceback for stuck debug"
        )


def _get_monitor_logger_file(prefix_key: str):
    hostid = get_local_host()
    stage = os.getenv("HAT_TRAINING_STEP", "undefined")
    log_file = os.path.join(
        LOG_DIR, f"{prefix_key}/{stage}/{prefix_key}-{hostid}"
    )
    return log_file


def dataloader_monitor(pids, dataloader_traceback_interval):
    if pystack is None:
        common_logger.warning(
            "Please install pystack to get traceback for stuck debug"
        )
        return
    log_file = _get_monitor_logger_file(prefix_key="data-traceback")
    logger = get_monitor_logger(
        "data-traceback",
        log_file,
        logging.INFO,
        force_init_logger=True,
        stderr=True,
        stdout=True,
    )

    pystack_cmd = get_pystack_cmd()
    while True:
        time.sleep(dataloader_traceback_interval)
        for pid in pids:
            rank, _ = get_dist_info()
            log_file_to_write = log_file + "-rank%d.log" % rank
            try:
                _log_stack_info(
                    rank=rank,
                    pid=pid,
                    pystack_cmd=pystack_cmd,
                    log_file=log_file_to_write,
                    logger=logger,
                )
                time.sleep(0.1)
            except Exception as e:
                common_logger.warning(f"data-traceback failed:{e}")
                pass


def dataloader_traceback(dataloader_iter, dataloader_traceback_interval):

    log_file = _get_monitor_logger_file(prefix_key="data-traceback")
    logger = get_monitor_logger(
        "data-traceback",
        log_file,
        logging.INFO,
        force_init_logger=True,
        stderr=True,
        stdout=True,
    )
    workers = []
    iter_name = dataloader_iter.__class__.__name__
    if iter_name == "_SingleProcessDataLoaderIter":
        logger.info(
            "No need to activate dataloader traceback for 0 dataloader worker"
        )
        return
    if iter_name == "_MultiProcessingDataLoaderIter":
        workers = dataloader_iter._workers
    elif iter_name == "MultitaskLoaderIterator":
        loaders = dataloader_iter.loaders
        for v in loaders.values():
            if (
                v._loader_iter.__class__.__name__
                == "_MultiProcessingDataLoaderIter"
            ):
                workers = workers + v._loader_iter._workers
    elif iter_name == "MultitaskInfLoader":
        for v in dataloader_iter._loader_iters.values():
            if v.__class__.__name__ == "_MultiProcessingDataLoaderIter":
                workers = workers + v._workers
    elif iter_name in ["ChainDataloader", "MultiBatchChainDataloader"]:
        for iter in dataloader_iter.data_loader_iters:
            if iter.__class__.__name__ == "_MultiProcessingDataLoaderIter":
                workers = workers + iter._workers
    else:
        logger.warning(
            f"Unsupported iterator type {type(dataloader_iter)} "
            "for dataloader traceback use"
        )
        return
    if len(workers) == 0:
        logger.warning("Get no dataloader workers")
        return
    pids = [worker.pid for worker in workers]

    trace_method = os.environ.get(
        "HAT_TRAIN_TRACEBACK_METHOD", "multiprocessing"
    )
    if trace_method == "threading":
        threading.Thread(
            target=dataloader_monitor,
            daemon=True,
            name="dataloader_traceback_thread",
            args=(pids, dataloader_traceback_interval),
        ).start()
    elif trace_method == "multiprocessing":
        monitor = multiprocessing.Process(
            target=dataloader_monitor,
            args=(pids, dataloader_traceback_interval),
        )
        monitor.daemon = True
        monitor.start()
    else:
        raise ValueError(
            f"Unsupported traceback method: {trace_method}. "
            "Please set HAT_TRAIN_TRACEBACK_METHOD to "
            "'threading' or 'multiprocessing'."
        )


def _log_stack_info(rank, pid, pystack_cmd, log_file, logger):

    with redirect_stdout(to_file=log_file), redirect_stderr(to_file=log_file):
        rank_info = (
            "=" * 20
            + f" rank[{rank}] "
            + f"processid[{pid}] "
            + "train traceback "
            + "=" * 20
        )
        logger.info(rank_info)
        command = f"{pystack_cmd} remote {pid}"
        result = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        logger.info(result.stdout)


def train_traceback(ranks, pids, traceback_monitor_interval):
    if pystack is None:
        common_logger.warning(
            "Please install pystack to get traceback for stuck debug"
        )
        return
    assert len(ranks) == len(pids)
    log_file = _get_monitor_logger_file(prefix_key="train-traceback")
    logger = get_monitor_logger(
        "train-traceback",
        log_file,
        logging.INFO,
        ranks[0],
        force_init_logger=True,
        stderr=True,
        stdout=True,
    )

    pystack_cmd = get_pystack_cmd()

    while True:
        time.sleep(traceback_monitor_interval)
        for rank, pid in zip(ranks, pids):
            try:
                log_file_to_write = log_file + "-rank%d.log" % rank
                _log_stack_info(
                    rank=rank,
                    pid=pid,
                    pystack_cmd=pystack_cmd,
                    log_file=log_file_to_write,
                    logger=logger,
                )
                time.sleep(0.1)
            except ProcessLookupError:
                break
            except Exception as e:
                common_logger.warning(f"train-traceback failed:{e}")
                pass


def init_traceback_mp_monitor(ranks: List[int], pids: List[int]):
    traceback_monitor_interval = int(
        os.environ.get("HAT_TRAIN_TRACEBACK_INTERVAL", 600)
    )

    if traceback_monitor_interval > 0:
        trace_method = os.environ.get(
            "HAT_TRAIN_TRACEBACK_METHOD", "multiprocessing"
        )
        if trace_method == "threading":
            threading.Thread(
                target=train_traceback,
                daemon=True,
                name="train_traceback_thread",
                args=(
                    ranks,
                    pids,
                    traceback_monitor_interval,
                ),
            ).start()
        elif trace_method == "multiprocessing":
            monitor_process = multiprocessing.Process(
                target=train_traceback,
                args=(
                    ranks,
                    pids,
                    traceback_monitor_interval,
                ),
            )
            monitor_process.daemon = True
            monitor_process.start()
        else:
            raise ValueError(
                f"Unsupported traceback method: {trace_method}. "
                "Please set HAT_TRAIN_TRACEBACK_METHOD to "
                "'threading' or 'multiprocessing'."
            )
        common_logger.info(
            f"Train traceback monitor process started with interval "
            f"{traceback_monitor_interval} seconds with "
            f"trace_method {trace_method}"
        )


class HATFlightRecorder:
    """
    HAT Flight Recorder for communication operation traceback.

    Record the communication operation and its arguments for communication
    traceback. The records are stored in a deque with a maximum length
    specified by the environment variable HAT_COMMUNICATION_TRACE_BUFFER_SIZE.
    """

    rank: int = None
    _init_flag: bool = False
    _instance: "HATFlightRecorder" = None
    _instance_lock = threading.Lock()
    __slots__ = ("records", "_initialized")

    def __init__(self):
        if not hasattr(self, "_initialized"):
            if os.environ.get("HAT_COMMUNICATION_TRACEBACK", "0") == "1":
                self.records = deque(
                    maxlen=os.getenv("HAT_COMMUNICATION_TRACE_BUFFER_SIZE", 20)
                )
            else:
                self.records = None
            self._initialized = True

    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super(HATFlightRecorder, cls).__new__(cls)
        return cls._instance

    @staticmethod
    def get_tensor_brief_info(x: torch.Tensor):
        return {
            "shape": list(x.size()),
            "dtype": x.dtype,
            "device": str(x.device),
        }

    @staticmethod
    def set_rank(rank):
        HATFlightRecorder.rank = rank

    def record(self, op_name, args=None, kwargs=None):
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        record_dict = {
            "rank": HATFlightRecorder.rank,
            "op_name": op_name,
            "args": args,
            "kwargs": kwargs,
        }
        self.records.append(record_dict)

    def dump(self) -> str:
        str_io = io.StringIO()
        pp = PrettyPrinter(indent=4, width=100, stream=str_io)
        for i, data in enumerate(self.records):
            pp.pprint(f"Record {i}")
            pp.pprint(data)
        return str_io.getvalue()

    def reset(self):
        self.records = deque(
            maxlen=os.getenv("HAT_COMMUNICATION_TRACE_BUFFER_SIZE", 20)
        )


FLIGHT_RECORDER = HATFlightRecorder()


def flight_record(func=None, communication: bool = True):
    """
    Mark a function for flight recording.

    This decorator will record the function name, its arguments,
    and keyword arguments. The recorded information will be stored
    in the HATFlightRecorder instance. If the function is called with
    communication=True, it will record the tensor information (shape,
    dtype, device) of the arguments and keyword arguments. Otherwise,
    it will record the string representation of the arguments and
    keyword arguments.

    Args:
        func (callable, optional): The function to be decorated.
            If not provided, the decorator will return a decorator.
        communication (bool): If True, record tensor information.
            If False, record string representation.
    """

    def _inner_warpper(_func, communication):
        if communication:

            @functools.wraps(_func)
            def wrapper(*args, **kwargs):
                record_args = pytree.tree_map_only(
                    torch.Tensor, HATFlightRecorder.get_tensor_brief_info, args
                )
                record_kwargs = pytree.tree_map_only(
                    torch.Tensor,
                    HATFlightRecorder.get_tensor_brief_info,
                    kwargs,
                )
                FLIGHT_RECORDER.record(
                    _func.__name__, record_args, record_kwargs
                )
                return _func(*args, **kwargs)

            return wrapper
        else:

            @functools.wraps(_func)
            def wrapper(*args, **kwargs):
                record_args = pytree.tree_map(lambda x: str(x), args)
                record_kwargs = pytree.tree_map(lambda x: str(x), kwargs)
                FLIGHT_RECORDER.record(
                    _func.__name__, record_args, record_kwargs
                )
                return _func(*args, **kwargs)

            return wrapper

    if callable(func):
        return _inner_warpper(func, communication)
    else:

        def decorator(func):
            return _inner_warpper(func, communication)

        return decorator


class HATNCCLHookDispatchMode(TorchDispatchMode):
    """
    Trcae the NCCL communication operations in PyTorch.

    This class is a TorchDispatchMode that intercepts the specified NCCL
    operations and records their arguments and keyword arguments.
    The recorded information is stored in the HATFlightRecorder
    instance. The operations to be traced can be specified
    when creating an instance of this class. If no operations
    are specified, a default set of NCCL operations will be traced.
    The default operations include allgather, allreduce, alltoall,
    broadcast, gather, scatter, reduce, and reduce_scatter.

    Args:
        hook_op_name_list (list, optional): A list of NCCL operations
            to be traced. If None, a default set of operations will be used.
    """

    NCCL_OPS = [
        "c10d._allgather_base_",
        "c10d._reduce_scatter_base_",
        "c10d.allgather_",
        "c10d.allgather_coalesced_",
        "c10d.allgather_into_tensor_coalesced_",
        "c10d.allreduce_",
        "c10d.allreduce_coalesced_",
        "c10d.alltoall_",
        "c10d.alltoall_base_",
        "c10d.broadcast_",
        "c10d.gather_",
        "c10d.scatter_",
        "c10d.reduce_",
        "c10d.reduce_scatter_",
        "c10d.reduce_scatter_tensor_coalesced_",
    ]
    __slots__ = ("hook_op_name_list",)

    def __init__(self, hook_op_name_list=None):
        super().__init__()
        if hook_op_name_list is None:
            hook_op_name_list = HATNCCLHookDispatchMode.NCCL_OPS
        else:
            hook_op_name_list.extend(HATNCCLHookDispatchMode.NCCL_OPS)
        self.hook_op_name_list = hook_op_name_list

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        func_packet = func._overloadpacket
        func_packet_name = str(func_packet)
        if func_packet_name in self.hook_op_name_list:
            record_args = pytree.tree_map_only(
                torch.Tensor, HATFlightRecorder.get_tensor_brief_info, args
            )
            record_kwargs = pytree.tree_map_only(
                torch.Tensor, HATFlightRecorder.get_tensor_brief_info, kwargs
            )
            FLIGHT_RECORDER.record(func.__name__, record_args, record_kwargs)
        return func(*args, **kwargs)


if os.environ.get("HAT_COMMUNICATION_TRACEBACK", "0") == "1":
    dist.all_gather = flight_record(dist.all_gather)
    dist.all_reduce = flight_record(dist.all_reduce)
    dist.all_to_all = flight_record(dist.all_to_all)
    dist.barrier = flight_record(dist.barrier, communication=False)
    dist.broadcast = flight_record(dist.broadcast)
    dist.gather = flight_record(dist.gather)
    dist.reduce = flight_record(dist.reduce)
    dist.scatter = flight_record(dist.scatter)
    dist.reduce_scatter = flight_record(dist.reduce_scatter)
    dist.gather_object = flight_record(dist.gather_object)
    dist.all_gather_object = flight_record(dist.all_gather_object)
    dist.all_gather_into_tensor = flight_record(dist.all_gather_into_tensor)
    dist.all_gather_coalesced = flight_record(dist.all_gather_coalesced)
    dist.all_to_all = flight_record(dist.all_to_all)
    dist.all_to_all_single = flight_record(dist.all_to_all_single)
    dist.all_reduce_coalesced = flight_record(dist.all_reduce_coalesced)
    dist.broadcast_object_list = flight_record(dist.broadcast_object_list)
    dist.reduce_scatter_tensor = flight_record(dist.reduce_scatter_tensor)
    dist.scatter_object_list = flight_record(dist.scatter_object_list)


def log_flight_record_heartbeat_thread():
    def log_flight_record(flight_record_logger):
        rank_info = (
            "=" * 20
            + f" rank[{HATFlightRecorder.rank}] "
            + "HAT Flight Recorder "
            + "=" * 20
        )
        flight_record_logger.info(rank_info)
        flight_record_logger.info(f"\n{FLIGHT_RECORDER.dump()}\n")
        time.sleep(0.1)

    log_file = _get_monitor_logger_file(prefix_key="train-traceback")
    logger = get_monitor_logger(
        "train-traceback",
        log_file,
        level=logging.INFO,
        rank=HATFlightRecorder.rank,
        force_init_logger=True,
        stderr=True,
        stdout=False,
        log_in_any_rank=True,
    )
    interval = int(os.getenv("HAT_COMMUNICATION_TRACEBACK_INTERVAL", 600))

    while True:
        time.sleep(interval)
        log_flight_record(logger)


def setup_hat_flight_record(rank: int = 0):
    """Set up the HAT Flight Recorder for communication traceback."""
    if os.environ.get("HAT_COMMUNICATION_TRACEBACK", "0") == "1":
        if dist.is_initialized():
            HATFlightRecorder.set_rank(rank)
            if not HATFlightRecorder._init_flag:
                threading.Thread(
                    target=log_flight_record_heartbeat_thread,
                    daemon=True,
                    name="HATFlightRecorderThread",
                ).start()
                HATFlightRecorder._init_flag = True
                common_logger.info(
                    "HAT Flight Recorder is initialized, "
                    "and the periodic print record is activated."
                )
        else:
            common_logger.info(
                "HAT_COMMUNICATION_TRACEBACK is set to 1, but "
                "distributed is not initialized. Please "
                "initialize distributed "
                "before using HAT Flight Recorder."
            )
    else:
        common_logger.warning(
            "Please set HAT_COMMUNICATION_TRACEBACK=1 to enable "
            "the periodic print record"
        )
