# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import functools
import inspect
import logging
from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import torch
from torch import Tensor
from torch.distributed import ProcessGroup
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import _flatten, dim_zero_cat
from torchmetrics.utilities.distributed import gather_all_tensors

# torchmetrics>=1.x moved this helper out of torchmetrics.utilities.
try:
    from torchmetrics.utilities import apply_to_collection
except ImportError:  # pragma: no cover
    try:
        from lightning_utilities.core.apply_func import apply_to_collection
    except ImportError:
        from hat_engine.utils.apply_func import apply_to_collection

__all__ = ["EvalMetric"]

logger = logging.getLogger(__name__)


class EvalMetric(Metric, ABC):
    """Base class for all evaluation metrics.

    Built on top of `torchmetrics.metric.Metric`, this base class introduces
    the name attribute and a name-value format output (the get method). It
    also makes possible to syncnronize state tensors of different shapes in
    each device to support AP-like metrics.


    .. note::

        This is a base class that provides common metric interfaces.
        One should not use this class directly, but inherit it to create new
        metric classes instead.

    Args:
        name: Name of this metric instance for display.
        process_group:  Specify the process group on which synchronization is
            called. Default: None (which selects the entire world)
        warn_without_compute: Whether to output warning log if `self.compute`
            is not called in `self.get`. Since synchronization among devices
            is executed in `self.compute`, this value reflects if the metric
            will support distributed computation.
    """

    # Mapping torch.dtype -> small integer tag for cross-rank negotiation.
    # Tag -1 is reserved as 'no local dtype / empty list'.
    _DTYPE_TO_TAG_RAW: ClassVar[Dict[torch.dtype, int]] = {
        torch.float32: 0,
        torch.float64: 1,
        torch.int64: 2,
        torch.int32: 3,
        torch.int16: 4,
        torch.int8: 5,
        torch.bool: 6,
        torch.bfloat16: 7,
        torch.float16: 8,
    }
    # Exposed as read-only mapping.
    DTYPE_TO_TAG: ClassVar[Mapping[torch.dtype, int]] = MappingProxyType(
        _DTYPE_TO_TAG_RAW
    )
    TAG_TO_DTYPE: ClassVar[Mapping[int, torch.dtype]] = MappingProxyType(
        {v: k for k, v in _DTYPE_TO_TAG_RAW.items()}
    )

    def __init__(
        self,
        name: Union[List[str], str],
        process_group: Optional[ProcessGroup] = None,
        warn_without_compute: bool = True,
    ):

        self.name = name
        self._attr_dtype_caches: Dict[str, Union[torch.dtype, None]] = {}

        super().__init__(
            compute_on_step=False,
            dist_sync_on_step=False,
            process_group=process_group,
        )
        self._warn_without_compute = warn_without_compute
        self.get = self._wrap_get(self.get)
        self._init_states()
        self.eps = 1e-6
        self._device = torch.device("cpu")
        self._dtype = torch.get_default_dtype()

    @property
    def device(self) -> "torch.device":
        """Return the device of the metric."""
        return self._device

    @property
    def dtype(self) -> "torch.dtype":
        """Return the default dtype of the metric."""
        return self._dtype

    def _init_states(self):
        """Initialize state variables.

        It is generally recommended to create state variables with
        add_state method, since in that case synchronization and reset
        can be handled automatically.

        State variables manually added with `self.xxx = yyy` cannot be
        synchronized and thus cannot be used in distributed case. Besides,
        they need to be manually reset by extending the reset method.
        """

        self.add_state(
            "sum_metric",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "num_inst",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

    def add_state(
        self,
        name: str,
        default: Union[list, Tensor],
        dist_reduce_fx: Optional[Union[str, Callable]] = None,
        persistent: bool = False,
        elem_dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Add a state variable to the metric.

        This method extends the base method to support list type state
        variables.

        Args:
            name: Name of the state variable.
            default: Default value of the state variable.
            dist_reduce_fx: Reduction function to be used when
                synchronizing the state variable across devices.
            persistent: Whether the state variable should be part of
                the metric's state dict.
            elem_dtype: Data type of elements inside the state variable,
                used for list case.
        """
        if isinstance(default, torch.Tensor):
            self._attr_dtype_caches[name] = default.dtype
        elif isinstance(default, list):
            if len(default) > 0 and isinstance(default[0], torch.Tensor):
                self._attr_dtype_caches[name] = default[0].dtype
            else:
                self._attr_dtype_caches[name] = elem_dtype

        super().add_state(
            name,
            default,
            dist_reduce_fx=dist_reduce_fx,
            persistent=persistent,
        )

    def reset(self) -> None:
        """Reset the metric state variables to their default value.

        If （and only if） there are state variables that are not registered
        with `self.add_state` need to be regularly set to default values,
        please extend this method in subclasses.
        """
        super().reset()

    def _sync_dist(
        self,
        dist_sync_fn: Callable = gather_all_tensors,
        process_group: Optional[Any] = None,
    ) -> None:
        input_dict = {
            attr: getattr(self, attr) for attr in self._reductions.keys()
        }

        for attr, reduction_fn in self._reductions.items():
            # pre-concatenate metric states that are lists
            # to reduce number of all_gather operations
            if reduction_fn == dim_zero_cat and isinstance(
                input_dict[attr], list
            ):
                # gather dtype info for list type states,
                # if attr not in _attr_dtype_caches
                attr_dtype = self._attr_dtype_caches.get(attr, None)
                if attr_dtype is None:
                    attr_dtype = self._sync_attr_dtype(input_dict[attr])
                    self._attr_dtype_caches[attr] = attr_dtype

                if len(input_dict[attr]) > 1:
                    _first_elem = input_dict[attr][0]
                    input_dict[attr] = [
                        item for item in input_dict[attr] if item.numel() > 0
                    ]

                    if len(input_dict[attr]) > 0:
                        input_dict[attr] = [dim_zero_cat(input_dict[attr])]
                    else:
                        input_dict[attr] = [torch.empty_like(_first_elem)]

                elif len(input_dict[attr]) == 0:
                    input_dict[attr] = [
                        torch.tensor(
                            [],
                            device=self.device,
                            dtype=self._attr_dtype_caches.get(
                                attr, self.dtype
                            ),
                        )
                    ]

        output_dict = apply_to_collection(
            input_dict,
            torch.Tensor,
            dist_sync_fn,
            group=process_group or self.process_group,
        )

        for attr, reduction_fn in self._reductions.items():
            # pre-processing ops (stack or flatten for inputs)
            if not output_dict[attr]:
                # empty result
                setattr(self, attr, output_dict[attr])
                continue
            elif isinstance(output_dict[attr][0], torch.Tensor):
                output_dict[attr] = torch.stack(output_dict[attr])
            elif isinstance(output_dict[attr][0], list):
                output_dict[attr] = _flatten(output_dict[attr])

            if isinstance(output_dict[attr], list) and isinstance(
                output_dict[attr][0], torch.Tensor
            ):
                all_empty = False
                for item in output_dict[attr]:
                    if item.numel() == 0:
                        all_empty = True
                    else:
                        all_empty = False
                        break
                if all_empty:
                    device = output_dict[attr][0].device
                    dtype = output_dict[attr][0].dtype
                    setattr(
                        self,
                        attr,
                        torch.tensor([], device=device, dtype=dtype),
                    )
                    logger.warning(
                        f"All ranks metric for state '{attr}' are empty."
                    )
                    continue

            assert isinstance(reduction_fn, Callable) or reduction_fn is None
            reduced = (
                reduction_fn(output_dict[attr])
                if reduction_fn is not None
                else output_dict[attr]
            )
            setattr(self, attr, reduced)

    @abstractmethod
    def update(self, *_: Any, **__: Any) -> None:
        """Override this method to update the state variables."""

    def compute(self) -> Union[float, List[float]]:
        """Override this method to compute final results from metric states.

        All states variables registered with `self.add_state` are synchronized
        across devices before the execution of this method.
        """
        val = self.sum_metric / (self.num_inst + self.eps)

        # scalar case
        if val.numel() == 1:
            val = val.item()
        else:
            val = val.cpu().numpy().tolist()
        return val

    def __getstate__(self) -> Dict[str, Any]:
        # ignore update and compute functions for pickling
        return {
            k: v
            for k, v in self.__dict__.items()
            if k not in ["update", "compute", "get", "_update_signature"]
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        # manually restore update and compute functions for pickling
        self.__dict__.update(state)
        self._update_signature = inspect.signature(self.update)
        self.update: Callable = self._wrap_update(self.update)
        self.compute: Callable = self._wrap_compute(self.compute)
        self.get: Callable = self._wrap_get(self.get)

    def get(self) -> Tuple[Union[str, List[str]], Union[float, List[float]]]:
        """Get current evaluation result.

        To skip the synchronization among devices, please override this method
        and calculate results without calling `self.compute()`.

        Returns:
           names: Name of the metrics.
           values: Value of the evaluations.
        """

        values = self.compute()
        if isinstance(values, list):
            assert isinstance(self.name, list) and len(self.name) == len(
                values
            )

        return self.name, values

    def _wrap_get(self, get: Callable):
        @functools.wraps(get)
        def wrapped_func(*args: Any, **kwargs: Any) -> Optional[Any]:
            res = get(*args, **kwargs)
            if self._warn_without_compute and self._computed is None:
                logger.warning(
                    f"{self.__class__} not ready for distributed environment,"
                    + " should not be used together with DistributedSampler."
                    + "Might be slow in validation due to resource competition"
                )
            return res

        return wrapped_func

    def get_name_value(self):
        """
        Return zipped name and value pairs.

        Returns:
            List(tuples): A (name, value) tuple list.
        """

        name, value = self.get()
        if not isinstance(name, list):
            name = [name]
        if not isinstance(value, list):
            value = [value]
        return list(zip(name, value))

    def _apply(self, fn):
        _dummy_tensor = fn(torch.zeros(1, device=self.device))
        self._device = _dummy_tensor.device
        self._dtype = _dummy_tensor.dtype
        return super()._apply(fn)

    def _dtype_to_tag(self, dtype: torch.dtype) -> int:
        """Return integer tag for given dtype or -1 if unknown."""
        return self.DTYPE_TO_TAG.get(dtype, -1)

    def _tag_to_dtype(self, tag: int) -> torch.dtype:
        """Return dtype for given integer tag or None if unknown."""
        return self.TAG_TO_DTYPE.get(tag, None)

    def _sync_attr_dtype(self, attr_data: List[Tensor]):
        """Synchronize dtype across all distributed ranks.

        This method ensures that all ranks in a distributed setting agree on
        the dtype of a list-based attribute.

        The process works as follows:
            1. Each rank inspects its local `attr_data` and extracts the dtype
                of the first element, or uses -1 if empty.
            2. All ranks share dtype tags using a collective gather operation.
            3. Collects all non-empty dtype tags and checks for consistency.
            4. If all ranks are empty, returns None. Otherwise, returns the
                agreed-upon `torch.dtype`.

        Args:
            attr_data: A list of tensors.

        Returns:
            torch.dtype or None: The common dtype agreed upon by all ranks,
                or None if all ranks have empty state.
        """

        if len(attr_data) > 0:
            local_dtype_tag = self._dtype_to_tag(attr_data[0].dtype)
        else:
            local_dtype_tag = -1

        local_dtype_tag_tensor = torch.tensor(
            [local_dtype_tag], device=self.device
        )
        dtype_tags = gather_all_tensors(local_dtype_tag_tensor)

        non_empty_tags = {c.item() for c in dtype_tags if c.item() != -1}
        if not non_empty_tags:
            return None
        if len(non_empty_tags) > 1:
            raise RuntimeError(
                f"Inconsistent dtypes detected across ranks for list state: {non_empty_tags}"  # noqa: E501
            )
        dtype_tag = next(iter(non_empty_tags))
        return self._tag_to_dtype(dtype_tag)
