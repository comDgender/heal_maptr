# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501


import os
import warnings
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

__all__ = [
    "CacheNotExistsError",
    "CacheOverwriteError",
    "CacheStrategy",
    # serializer
    "Serializer",
    "CustomSerializer",
    "build_serializer",
    # storager backend
    "BaseStorager",
    "build_storager",
    "InMemoryStorager",
    "FileStorager",
]


def identity(x):
    """
    Identity pass input.

    Args:
        x : object
    """
    return x


class CacheNotExistsError(Exception):
    pass


class CacheOverwriteError(Exception):
    pass


@dataclass
class CacheStrategy:
    timedelta: Optional[None] = None
    strategy: Optional[str] = "access_time"
    overwrite: Optional[bool] = True


class Serializer(metaclass=ABCMeta):
    @abstractmethod
    def dumps(self, data) -> Any:
        pass

    @abstractmethod
    def loads(self, data) -> Any:
        pass


class CustomSerializer(Serializer):
    def __init__(
        self, dumps_fn: Callable[[Any], Any], loads_fn: Callable[[Any], Any]
    ):
        self.dumps_fn = dumps_fn
        self.loads_fn = loads_fn

    def dumps(self, data) -> Any:
        return self.dumps_fn(data)

    def loads(self, data) -> Any:
        return self.loads_fn(data)


def build_serializer(
    serializer: Optional[Union[str, Serializer]] = "bypass"
) -> Serializer:
    if isinstance(serializer, Serializer):
        return serializer
    if not isinstance(serializer, str):
        raise TypeError(
            "Expected type `Serializer` or `str`, "
            "but get type {}".format(type(serializer))
        )
    if serializer == "bypass":
        return CustomSerializer(identity, identity)
    if serializer == "pickle":
        import pickle

        return CustomSerializer(pickle.dumps, pickle.loads)
    if serializer == "json":
        import json

        return CustomSerializer(json.dumps, json.loads)
    raise NotImplementedError(
        "Only support `bypass`, `pickle`, `json`, but get `{}`".format(
            serializer
        )  # noqa
    )


class BaseStorager(metaclass=ABCMeta):
    """Base storager."""

    def __init__(
        self,
        protocol: Optional[Union[str, None]] = None,
        cache_strategy: Optional[Union[CacheStrategy, None]] = None,
    ):
        """Construct a new BaseStorager object.

        Args:
            protocol: Cache protocol
            cache_strategy: Cache strategy
        """
        self.protocol = protocol
        self.cache_strategy = cache_strategy

    @abstractmethod
    def get(self, key: str) -> Any:
        """Get cache by key."""

    @abstractmethod
    def set(self, key: str, value: Any) -> bool:
        """Set cache."""

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete cache."""


def build_storager(protocol: str, *args, **kwargs) -> BaseStorager:
    """Build cache storager by protocol."""
    if not isinstance(protocol, str):
        raise TypeError(
            "Expected type str, but get type {}".format(type(protocol))
        )
    if protocol.startswith("file://"):
        storager = FileStorager(protocol, *args, **kwargs)
        return storager
    elif protocol.startswith("memory://"):
        storager = InMemoryStorager(protocol, *args, **kwargs)
        return storager
    else:
        raise ValueError("does not support protocol: {}".format(protocol))


class InMemoryStorager(BaseStorager):
    """Store cache in memory."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache_data = {}
        self._post_check()

    def _post_check(self):
        if self.cache_strategy.timedelta is not None:
            warnings.warn("Ignore timedelta for InMemoryStorager")

    def get(self, key: str) -> Any:
        if key not in self._cache_data:
            raise CacheNotExistsError
        return self._cache_data[key]

    def set(self, key: str, value: Any) -> bool:
        if key in self._cache_data and not self.cache_strategy.overwrite:
            raise CacheOverwriteError("Overwrite cache {}".format(key))
        self._cache_data[key] = value
        return True

    def delete(self, key: str) -> bool:
        self._cache_data.pop(key)


class FileStorager(BaseStorager):
    """Store cache in file system."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.cache_strategy.timedelta is not None:
            warnings.warn("Ignore timedelta for InMemoryStorager")

        url = self.protocol[len("file://") :]
        if os.path.exists(url):
            if not os.path.isdir(url):
                raise NotADirectoryError(f"{self.protocol} is not a directory")

        self._url = os.path.join(url, ".hat_workflow_cache")
        os.makedirs(self._url, exist_ok=True)

    def _get_cache_url(self, key) -> str:
        return os.path.join(self._url, key)

    def get(self, key: str) -> bytes:
        url = self._get_cache_url(key)
        if not os.path.exists(url):
            raise CacheNotExistsError
        with open(url, "rb") as fread:
            return fread.read()

    def set(self, key: str, value: bytes) -> bool:
        url = self._get_cache_url(key)
        if os.path.exists(url) and not self.cache_strategy.overwrite:
            raise CacheOverwriteError("Overwrite cache {}".format(key))
        with open(url, "wb") as fwrite:
            fwrite.write(value)
        return True

    def delete(self, key: str) -> bool:
        url = self._get_cache_url(key)
        if os.path.exists(url):
            os.remove(url)
        return True
