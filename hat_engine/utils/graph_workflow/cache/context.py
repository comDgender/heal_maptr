# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

from hat_engine.utils.graph_workflow.cache.storager import (
    BaseStorager,
    CacheStrategy,
    Serializer,
    build_serializer,
    build_storager,
)
from hat_engine.utils.graph_workflow.exception import OpSkipFlag
from hat_engine.utils.graph_workflow.symbol import get_operator_name

__all__ = [
    "OpCacheContext",
    "CacheArgs",
    "CacheContext",
    "DisableCacheContext",
]


@dataclass
class OpCacheContext:
    """Context for computing cache key fn.

    Args:
        fn: Operator.
        args: Operator positional inputs.
        kwargs: Operator key-value inputs
        node_name: Node name, by default None
        fn_name: Operator name
    """

    fn: Callable
    args: Tuple
    kwargs: Dict
    node_name: Optional[Union[str, None]] = None

    def __post_init__(self):
        self.fn_name = get_operator_name(self.fn, with_module_name=True)


@dataclass
class CacheArgs:
    """Cache arguments.

    Args:
    cache_key_fn: The way to compute
        cache key, return None if you do not want to use cache
    serializer: Serialization method, by default bypass
    ignore_cache_failure: Whether or not raise exception when
        compute cache failed, by default True
    """  # noqa

    cache_key_fn: Callable[[OpCacheContext], Union[str, None]]
    serializer: Optional[Union[Serializer, str]] = "bypass"
    ignore_cache_failure: Optional[bool] = True

    def __post_init__(self):
        self.serializer_impl = build_serializer(self.serializer)


@dataclass
class SkipFlagCache:
    """A specitial class to hold OpSkipFlag and its reason."""

    data: Any
    reason: Tuple[str] = ()


class CacheContext:
    """Using this context to control the cache mechanism."""

    _current = None

    __OP_SKIP_FLAG__ = "__OP_SKIP_FLAG__"

    def __init__(
        self,
        cache_storager: Optional[Union[str, BaseStorager]] = "memory://",
        cache_strategy: Optional[Union[None, CacheStrategy]] = None,
        cache_args: Optional[Union[None, CacheArgs]] = None,
        key_prefix: Optional[str] = "",
    ):
        """
        Construct a new CacheContext object.

        Args:
            cache_storager: Cache storager,
                by default "memory://"
            cache_strategy: Cache strategy,
                by default None
            cache_args: Cache function
                arguments, by default None
            key_prefix: Cache key prefix, by default ""
        """
        self._old = None
        self._set_cache_storager(cache_storager, cache_strategy)
        self.cache_args = cache_args
        self.prefix = key_prefix

    def _set_cache_storager(self, cache_storager, cache_strategy):
        if isinstance(cache_storager, str):
            if cache_strategy is None:
                cache_strategy = CacheStrategy()
            cache_storager = build_storager(cache_storager, cache_strategy)
        else:
            if not isinstance(cache_storager, BaseStorager):
                raise TypeError(
                    "Expected type `str` or `BaseStorager`, "
                    "but get type {}".format(type(cache_storager))
                )
            if cache_strategy is not None:
                warnings.warn(
                    "cache_storager is already created, "
                    "ignore the `cache_strategy` argument"
                )
        self.cache_storager = cache_storager

    def __enter__(self) -> "CacheContext":
        self._old = CacheContext._current
        CacheContext._current = self
        return self

    def __exit__(self, ptype, value, trace):
        CacheContext._current = self._old
        self._old = None

    @classmethod
    def current(cls):
        """Get Current CacheContext."""
        return cls._current

    @classmethod
    def is_active(cls):
        """Whether or not the CacheContext is active."""
        return cls._current is not None

    def get(self, key: str, serializer: Serializer) -> Any:
        """Get cache key from storager and deserialize."""
        ret = serializer.loads(self.cache_storager.get(self.prefix + key))
        if (
            isinstance(ret, tuple)
            and ret
            and ret[0] == CacheContext.__OP_SKIP_FLAG__
        ):
            ret = SkipFlagCache(data=OpSkipFlag, reason=ret[1])
        return ret

    def set(self, key: str, value: Any, serializer: Serializer) -> bool:
        """Serialize data and store in cache storager."""
        if isinstance(value, SkipFlagCache):
            value = (CacheContext.__OP_SKIP_FLAG__, value.reason)
        return self.cache_storager.set(
            self.prefix + key, serializer.dumps(value)
        )

    def delete(self, key: str) -> bool:
        """Delete key from storager."""
        return self.cache_storager.delete(self.prefix + key)


class DisableCacheContext:
    """An context to disable the cache mechanism."""

    def __init__(self):
        self._old = None

    def __enter__(self) -> "DisableCacheContext":
        self._old = CacheContext._current
        CacheContext._current = None
        return self

    def __exit__(self, ptype, value, trace):
        CacheContext._current = self._old
        self._old = None
