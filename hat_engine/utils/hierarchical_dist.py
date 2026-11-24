# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import logging
from enum import Enum, auto
from typing import Dict, Iterable, List, Optional, OrderedDict, Union

import torch

try:
    from torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook import (  # noqa
        post_localSGD_hook,
    )
    from torch.distributed.algorithms.model_averaging import (
        utils as average_utils,
    )
    from torch.distributed.algorithms.model_averaging.averagers import (
        ModelAverager,
    )
    from torch.distributed.optim import PostLocalSGDOptimizer
except ImportError:
    ModelAverager = None
    average_utils = None
    PostLocalSGDOptimizer = None
    post_localSGD_hook = None
from torch.distributed import ProcessGroup

from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.distributed import (
    ProcessGroupTag,
    get_local_process_group,
    get_node_count,
    set_split_subprocess_group_flag,
    split_process_group_by_host,
    used_split_subprocess_group,
)

logger = logging.getLogger(__name__)

__all__ = ["HierarchicalPlugin"]


_GLOBAL_OPTIMIZER_STEP_COUNTER = 0


class GroupSyncStrategy(Enum):
    NO_GROUPING = auto()
    SYNCBN_ONLY = auto()
    GRAD_ONLY = auto()
    FULL_GROUPING = auto()


def increace_global_optimizer_step_counter(increacement=1):
    global _GLOBAL_OPTIMIZER_STEP_COUNTER
    _GLOBAL_OPTIMIZER_STEP_COUNTER += increacement


def get_global_optimizer_step_counter():
    global _GLOBAL_OPTIMIZER_STEP_COUNTER
    return _GLOBAL_OPTIMIZER_STEP_COUNTER


def get_all_params_to_average(
    params: Union[
        Iterable[torch.nn.Parameter], Iterable[Dict[str, torch.nn.Parameter]]
    ],
):
    """Return a list of parameters that need to average.

    Args:
        params: The parameters of a model or parameter groups of an optimizer.
    """
    filtered_params = []
    for param in params:
        if isinstance(param, torch.nn.Parameter):
            # model.parameters() input
            param_data = param
            filtered_params.append(param_data)
        elif isinstance(param, dict):
            # optimizer.param_groups input
            for param_data in param["params"]:
                filtered_params.append(param_data)
        else:
            raise NotImplementedError(
                f"Parameter input of type {type(param)} is not supported"
            )

    return filtered_params


class PostLocalState:
    """Stores the state for all-reducing gradients globally or grouplly."""

    __slots__ = [
        "process_group",
        "subgroup",
        "start_localSGD_iter",
    ]

    def __init__(
        self,
        process_group: Optional[ProcessGroup] = None,
        subgroup: Optional[ProcessGroup] = None,
        start_localSGD_iter: int = 0,
    ):

        # used for all-reducing gradients globally.
        self.process_group = process_group
        # used for all-reducing gradients grouplly.
        self.subgroup = subgroup
        self.start_localSGD_iter = start_localSGD_iter


class PostLocalOptimizer(PostLocalSGDOptimizer):
    """Wrapper of `torch.optim.Optimizer`.

    This optimizer runs local optimizer at every step.
    After the warm-up stage, it averages parameters periodically afer the
    local optimizer is applied.

    Args:
        optim: The local optimizer.
        averager: A model averager instance to run post-localSGD algorithm.
    """

    def __init__(
        self,
        optim: torch.optim.Optimizer,
        averager: ModelAverager,
    ):
        super().__init__(optim, averager)

    @property
    def state(self):
        return self.optim.state

    def __repr__(self):
        return self.optim.__repr__()

    def step(self):
        self.optim.step()
        self.averager.average_parameters(params=self.param_groups)
        increace_global_optimizer_step_counter(increacement=1)


class HierarchicalModelAverager(ModelAverager):
    """Model averager for hierarchical model averaging.

    Args:
        period_group_size_dict: An ordered mapping of model averaging period.
        warmup_steps: The number of warm-up steps.
        process_group: The process group to average parameters.
    """

    def __init__(
        self,
        period_group_size_dict: OrderedDict[int, int],
        warmup_steps: int = 0,
        process_group: Optional[ProcessGroup] = None,
    ):
        super().__init__(process_group)

        self._periods = list(period_group_size_dict.keys())
        if self._periods[0] <= 0:
            raise ValueError(
                "The minimum period in arg ``period_group_size_dict`` must be a positive value."  # noqa E501
            )
        elif self._periods[-1] == 1:
            logger.warning(
                "When the maximum period in arg ``period_group_size_dict`` is "
                "1, no need to use model averaging because the communication "
                "cost of all-reducing parameters will be no less than the cost"
                " of all-reducing gradients by DistributedDataParallel in the "
                "backward pass. Therefore, only DistributedDataParallel should"
                " be used for this case."
            )

        self.period_process_group_dict = OrderedDict()
        overall_node_size = get_node_count(self.process_group)

        msg = "Model averaging hierarchy:\n"

        for period, node_size in period_group_size_dict.items():
            msg += f"\tEach group that has {node_size} nodes to average model"
            f"parameters every {period} iterations, if no higher-level averaging.\n"  # noqa E501

            if node_size != overall_node_size:
                (
                    self.period_process_group_dict[period],
                    _,
                ) = split_process_group_by_host(
                    self.process_group,
                    hosts_per_group=node_size,
                    pg_tag=ProcessGroupTag.SYNC_PARAM,
                )

            else:
                self.period_process_group_dict[period] = self.process_group

        logger.info(msg)

        if warmup_steps < 0:
            raise ValueError(
                "Arg ``warmup_steps`` must be a non-negative number."
            )
        self.warmup_steps = warmup_steps

    def _find_process_group(self):
        for period in reversed(self._periods):
            if self.step % period == 0:
                return self.period_process_group_dict[period]
        return None

    def average_parameters(
        self,
        params: Union[
            Iterable[torch.nn.Parameter],
            Iterable[Dict[str, torch.nn.Parameter]],
        ],
    ):
        if self.step >= self.warmup_steps:
            group = self._find_process_group()
            if group is not None:
                average_utils.average_parameters_or_parameter_groups(
                    params, group
                )
        self.step += 1


@OBJECT_REGISTRY.register
class HierarchicalPlugin:
    """Hierarchical and group sync plugin.

    Args:
        num_host_per_group: Number of host per group.
        period_group_size: An ordered mapping keys of model averaging period to
            process group size, used for initializing process groups of
            different sizes in a hierarchy to average parameters concurrently.

            For example:
                period_group_size = [(2, 16), (4, 32), (8, 64)]
                the mapping has three keys: 2, 4, and 8, which means:
                totally three process groups will be created to average
                parameters every 2, 4, and 8 iterations, respectively.
                every 2 iterations, 16 rank will run averaging,
                every 4 iterations, 32 rank will run averaging,
                every 8 iterations, 64 rank will run averaging.
        warmpup_steps: The number of warm-up steps. During this stage,
            model averaging is skipped..
        group_sync_strategy: The strategy of group synchronization.
    """

    _sync_strategy_map = {
        "full_grouped": GroupSyncStrategy.FULL_GROUPING,
        "syncbn_only": GroupSyncStrategy.SYNCBN_ONLY,
        "grad_only": GroupSyncStrategy.GRAD_ONLY,
    }

    def __init__(
        self,
        num_host_per_group: int,
        period_group_size: Optional[Union[OrderedDict, List]] = None,
        warmup_steps: int = 1,
        group_sync_strategy: Optional[str] = None,
    ) -> None:

        self._default_pg = None
        if group_sync_strategy is not None:
            assert (
                group_sync_strategy in self._sync_strategy_map.keys()
            ), f"`group_sync_strategy` should be one of {self._sync_strategy_map.keys()}, but get {group_sync_strategy}"  # noqa E501

            self.group_sync_strategy = self._sync_strategy_map.get(
                group_sync_strategy
            )
        else:
            self.group_sync_strategy = GroupSyncStrategy.NO_GROUPING

        if period_group_size is not None:
            if isinstance(period_group_size, (List, OrderedDict)):
                period_group_size = OrderedDict(period_group_size)
            else:
                raise ValueError(
                    f"period_group_size should be a List or OrderedDict, "
                    f"but got {type(period_group_size)}"
                )
        else:
            if self.group_sync_strategy in [
                GroupSyncStrategy.FULL_GROUPING,
                GroupSyncStrategy.GRAD_ONLY,
            ]:
                logger.warning(
                    "recommend to set `period_group_size` for hierarchical model averaging"  # noqa E501
                )

        self.num_host_per_group = num_host_per_group
        self.period_group_size_dict = period_group_size
        self.warmup_steps = warmup_steps

        if self.group_sync_strategy == GroupSyncStrategy.FULL_GROUPING:
            set_split_subprocess_group_flag(True)

    def create_subgroups(
        self,
        process_group: Optional[ProcessGroup] = None,
    ) -> ProcessGroup:

        if process_group is None:
            process_group = get_local_process_group()

        subgroup_process, change = split_process_group_by_host(
            process_group=(
                process_group if process_group is not None else self.default_pg
            ),
            hosts_per_group=self.num_host_per_group,
            pg_tag=ProcessGroupTag.DEFAULT,
        )

        if change:
            logger.info(
                f"Split process group by each {self.num_host_per_group} host"
            )

        return subgroup_process

    def register_comm_hook_to_model(
        self,
        model: torch.nn.Module,
        process_group: Optional[ProcessGroup] = None,
        subgroup: Optional[ProcessGroup] = None,
    ) -> torch.nn.Module:

        if self.group_sync_strategy in [
            GroupSyncStrategy.FULL_GROUPING,
            GroupSyncStrategy.GRAD_ONLY,
        ]:
            state = PostLocalState(
                process_group=(
                    process_group
                    if process_group is not None
                    else self.default_pg
                ),
                subgroup=subgroup,
                start_localSGD_iter=self.warmup_steps,
            )

            model.register_comm_hook(state, post_localSGD_hook)

        return model

    def wrap_optimizer(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.Optimizer:
        if self.group_sync_strategy in [
            GroupSyncStrategy.FULL_GROUPING,
            GroupSyncStrategy.GRAD_ONLY,
        ]:
            if self.period_group_size_dict is not None:
                optimizer = PostLocalOptimizer(
                    optim=optimizer,
                    averager=HierarchicalModelAverager(
                        period_group_size_dict=self.period_group_size_dict,
                        warmup_steps=self.warmup_steps,
                        process_group=self.default_pg,
                    ),
                )
        return optimizer

    @staticmethod
    def average_model_params_on_global(
        model: torch.nn.Module,
        process_group: Optional[ProcessGroup] = None,
    ) -> torch.nn.Module:
        with torch.no_grad():
            average_utils.average_parameters(
                iter(get_all_params_to_average(model.parameters())),
                process_group=process_group,
            )

        return model

    @property
    def default_pg(self):
        return self._default_pg

    @default_pg.setter
    def default_pg(self, process_group):
        self._default_pg = process_group


def maybe_sync_params(
    model: torch.nn.Module, process_group: Optional[ProcessGroup] = None
):
    """Average model parameters on the `process_group`.

    Args:
        model: The model to average parameters.
        process_group: The process_group to average parameters.
    """
    if used_split_subprocess_group():
        model = HierarchicalPlugin.average_model_params_on_global(
            model=model,
            process_group=process_group,
        )
    return model
