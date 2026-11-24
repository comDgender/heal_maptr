# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501


import logging
import os
import random
import socket
from collections import defaultdict
from datetime import timedelta
from enum import Enum
from functools import wraps
from itertools import chain
from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.utils.data as data
from torch.distributed import ProcessGroup

try:
    from torch.distributed.utils import _sync_module_states
except ImportError:
    _sync_module_states = None

__all__ = [
    "find_free_port",
    "get_dist_info",
    "rank_zero_only",
    "get_device_count",
    "get_global_process_group",
    "get_local_host",
    "set_local_process_group",
    "get_local_process_group",
    "split_process_group_by_host",
    "reduce_mean",
    "get_comm_backend_name",
    "sync_module_param_states",
]

logger = logging.getLogger(__name__)

_GLOBAL_PROCESS_GROUP = None
_LOCAL_PROCESS_GROUP = _GLOBAL_PROCESS_GROUP
_HOST_PROCESS_GROUP_CACHED = {}
_RANK_PROCESS_GROUP_CACHED = {}


class ProcessGroupTag(Enum):
    DEFAULT = "default"
    SYNC_PARAM = "sync_param"


def set_split_subprocess_group_flag(flag: bool = False):
    os.environ["HAT_SPLIT_SUBPROCESS_GROUP"] = str(int(flag))


def used_split_subprocess_group():
    return os.getenv("HAT_SPLIT_SUBPROCESS_GROUP", "0") == "1"


def find_free_port(start_port: int = 10001, end_port: int = 19999) -> int:
    """Find free port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    free_port = -1
    max_retry = 300
    for _i in range(0, max_retry):
        port = random.randint(start_port, end_port)
        try:
            sock.bind(("", port))
            free_port = port
            sock.close()
            break
        except socket.error:
            continue
    if free_port == -1:
        raise Exception("can not found a free port")
    return free_port


def rank_zero_only(fn):
    """Migrated from pytorch-lightning."""

    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            return fn(*args, **kwargs)

    return wrapped_fn


def dist_initialized():
    initialized = False
    if torch.__version__ < "1.0":
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
    return initialized


def get_dist_info(process_group=None) -> Tuple[int, int]:
    """Get distributed information from process group."""
    if dist_initialized():
        rank = dist.get_rank(process_group)
        world_size = dist.get_world_size(process_group)
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def get_worker_info() -> Tuple[int, int]:
    worker_info = data.get_worker_info()
    if worker_info is not None:
        worker_idx = worker_info.id
        num_workers = worker_info.num_workers
    else:
        num_workers = 1
        worker_idx = 0
    return worker_idx, num_workers


def get_device_count(process_group=None):
    """Return the number of GPUs available."""
    initialized = dist_initialized()
    if initialized:
        _, world_size = get_dist_info(process_group)
        return world_size
    else:
        return torch.cuda.device_count()


def get_node_count(process_group=None):
    """Return the number of Nodes/Machines."""
    if dist_initialized():
        _, world_size = get_dist_info(process_group)
        device_count = torch.cuda.device_count()
        return world_size // device_count
    else:
        return 1


def create_process_group(rank_list: Tuple[str]) -> Any:
    """Create a new process group by a list of rank id."""
    group = None
    if dist_initialized():
        group = dist.new_group(
            rank_list,
            timeout=timedelta(
                seconds=int(
                    os.environ.get("HAT_PROCESS_GROUP_TIMEOUT", "1800")
                )
            ),
        )
    return group


def get_global_process_group():
    global _GLOBAL_PROCESS_GROUP
    return _GLOBAL_PROCESS_GROUP


def set_local_process_group(process_group):
    global _LOCAL_PROCESS_GROUP
    _LOCAL_PROCESS_GROUP = process_group


def get_local_process_group():
    global _LOCAL_PROCESS_GROUP
    return _LOCAL_PROCESS_GROUP


def set_rank_process_group(
    rank: int,
    process_group: ProcessGroup,
    pg_tag: str = ProcessGroupTag.DEFAULT,
):
    global _RANK_PROCESS_GROUP_CACHED
    _RANK_PROCESS_GROUP_CACHED[(rank, pg_tag)] = process_group


def get_rank_process_group(
    rank: int = None, pg_tag: str = ProcessGroupTag.DEFAULT
):
    global _RANK_PROCESS_GROUP_CACHED

    if rank is None:
        rank, _ = get_dist_info(None)
    return _RANK_PROCESS_GROUP_CACHED.get((rank, pg_tag), None)


def get_local_host():
    """Get local host ip."""
    try:
        hostid = socket.gethostname()
        if hostid is None:
            hostid = socket.getfqdn()
            if hostid == "localhost":
                hostid = socket.gethostbyname(hostid)
                if hostid == "0.0.0.0":
                    raise Exception("get host name failed")
        return hostid
    except Exception as e:
        logger.error(str(e))
        return None


def collect_host_rank_map(
    process_group: Optional[ProcessGroup] = None,
    init_local_rank_env: bool = False,
):
    """Get host-rank map information.

    Args:
        process_group: The process_group which contains local rank.
        init_local_rank_env: Whether to initialize the local rank
            environment variable.
    """

    if not dist_initialized():
        return None

    hostid = get_local_host()

    if hostid is None:
        return None

    current_rank, _ = get_dist_info(None)
    _, local_world_size = get_dist_info(process_group)
    # get all rank and hostid within process_group
    local_data = [current_rank, hostid]
    glob_data = [None for _ in range(local_world_size)]
    all_gather_object(glob_data, local_data, process_group)

    # aggregate ranks by hostid
    glob_host_dict = defaultdict(set)
    for gd in glob_data:
        r, host = gd  # gd is [rank, hostid]
        glob_host_dict[host].add(r)

    if init_local_rank_env:
        for _, ranks in glob_host_dict.items():
            if current_rank in ranks:
                host_local_rank = list(ranks).index(current_rank)
                os.environ["HAT_HOST_LOCAL_RANK"] = str(host_local_rank)

    return glob_host_dict


def split_process_group_by_host(
    process_group: Optional[ProcessGroup] = None,
    hosts_per_group: int = 1,
    pg_tag: str = ProcessGroupTag.DEFAULT,
) -> Tuple[ProcessGroup, bool]:
    """Get process group that only contains localhost rank within process group.

    Args:
        process_group: a process_group which contains local rank.
        hosts_per_group: number of hosts per group.
        pg_tag: tag of process group.
    """
    if not dist_initialized():
        return process_group, False

    global _HOST_PROCESS_GROUP_CACHED
    cache_key = (process_group, hosts_per_group, pg_tag)
    if cache_key in _HOST_PROCESS_GROUP_CACHED:
        # use cached result for process group
        return _HOST_PROCESS_GROUP_CACHED[cache_key], True

    # first get host name/ipaddr for current rank
    hostid = get_local_host()
    if hostid is None:
        # get host failed, fallback to origin process group
        return process_group, False

    # 1. first split process group by host
    current_rank, _ = get_dist_info(None)
    glob_host_dict = collect_host_rank_map(process_group)

    # new_group_ranks is used to create new groups
    # it is [[r1,r2],[r3,r4]] format, which is grouped by host,
    # and each element will be used to create group.
    # for example:
    #    total 4 hosts, 2 ranks per host:
    #       host1: [0, 1], host2: [2, 3], host3: [4, 5], host4: [6, 7]
    #    if hosts_per_group is 2, then
    #        new_group_hostid: [[host1, host2], [host3, host4]]
    #        new_group_ranks: [[0, 1, 2, 3], [4, 5, 6, 7]]
    sorted_hosts = sorted(glob_host_dict.keys())
    nums_hosts = len(sorted_hosts)
    new_group_ranks, new_group_hostid = [], []
    for i in range(0, nums_hosts, hosts_per_group):
        group_hosts = sorted_hosts[i : i + hosts_per_group]
        group_ranks = list(
            chain.from_iterable(glob_host_dict[host] for host in group_hosts)
        )

        new_group_ranks.append(group_ranks)
        new_group_hostid.append(group_hosts)

    # handle the case where hosts_per_group does not
    # divide len(sorted_hosts) evenly.
    if nums_hosts % hosts_per_group != 0:
        remaining_hosts = sorted_hosts[
            nums_hosts - (nums_hosts % hosts_per_group) :
        ]
        remaining_ranks = list(
            chain.from_iterable(
                glob_host_dict[host] for host in remaining_hosts
            )
        )

        new_group_hostid.append(remaining_hosts)
        new_group_ranks.append(remaining_ranks)

    if len(new_group_ranks) == 1 and len(new_group_hostid) == 1:
        # the process group all in same host
        # so not need to split
        new_group_ranks = []
        new_group_hostid = []

    # 2. if process group is not global group, then exchange all split info
    if process_group is not None and process_group != dist.group.WORLD:
        # exchange group info when process group is not default
        _, world_size = get_dist_info(None)
        glob_data = [None for _ in range(world_size)]

        new_group_data = [new_group_hostid, new_group_ranks]
        all_gather_object(glob_data, new_group_data, None)
        new_group_ranks, new_group_hostid = [], []
        for h, r in glob_data:
            # r is list of rank-list that on same host
            new_group_ranks.extend(r)
            new_group_hostid.extend(h)

    if len(new_group_ranks) == 0:
        # all process group on one host, not need to create new group
        logger.info(
            f"rank {current_rank} same host {hostid} not need to split"
        )
        return process_group, True

    # create new groups
    result_pg = process_group
    for host_group, ranks in zip(new_group_hostid, new_group_ranks):
        npg = create_process_group(ranks)
        if current_rank in ranks:
            logger.info(
                f"host: {host_group} create new group for ranks: {ranks}"
            )
            result_pg = npg
            set_rank_process_group(current_rank, result_pg, pg_tag=pg_tag)
    _HOST_PROCESS_GROUP_CACHED[cache_key] = result_pg
    return result_pg, True


def all_gather_object(obj_list: List[Any], obj: Any, group=None):
    """Gather object from every ranks in group."""
    if dist_initialized():
        dist.all_gather_object(obj_list, obj, group)
    else:
        assert isinstance(obj_list, list) and len(obj_list) == 1
        obj_list[0] = obj


def get_global_out(output, default_value=None):
    global_rank, global_world_size = get_dist_info()
    global_output = [default_value for _ in range(global_world_size)]
    all_gather_object(global_output, output)
    return global_rank, global_output


def reduce_mean(
    tensor: torch.Tensor,
    process_group: Optional[ProcessGroup] = None,
) -> torch.Tensor:
    """Obtain the mean of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor

    if process_group is None and used_split_subprocess_group():
        process_group = get_rank_process_group(pg_tag=ProcessGroupTag.DEFAULT)

    tensor = tensor.clone()
    dist.all_reduce(
        tensor.div_(dist.get_world_size(group=process_group)),
        group=process_group,
        op=dist.ReduceOp.SUM,
    )
    return tensor


def get_comm_backend_name():
    """Get the communication backend name."""
    if dist_initialized():
        return dist.get_backend()
    return ""


def reduce_max(
    tensor: torch.Tensor,
    process_group: Optional[ProcessGroup] = None,
) -> torch.Tensor:
    """Obtain the max value of tensor on different GPUs."""
    if not (dist.is_available() and dist.is_initialized()):
        return tensor

    if process_group is None and used_split_subprocess_group():
        process_group = get_rank_process_group(pg_tag=ProcessGroupTag.DEFAULT)

    dist.all_reduce(tensor, group=process_group, op=dist.ReduceOp.MAX)
    return tensor


def sync_module_param_states(module):
    """Sync the module parameter states across different GPUs."""
    if (
        not (dist.is_available() and dist.is_initialized())
        and _sync_module_states is None
    ):
        logger.warning(
            "current torch version does not support sync_module_states, "
            "so the module parameter states will not be synced."
        )
        return
    _sync_module_states(
        module=module,
        process_group=module.process_group,
        broadcast_bucket_size=module.broadcast_bucket_size,
        src=0,
        params_and_buffers_to_ignore=module.parameters_to_ignore,
        broadcast_buffers=module.broadcast_buffers,
    )
