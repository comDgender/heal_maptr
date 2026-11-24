# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import copy
from typing import Any, Dict, Hashable, Optional

__all__ = [
    "ContextResourceManager",
    "get_current_resource_manager",
    "is_active",
    "get_resource",
]


class ContextResourceManager(object):
    """
    Manage resources in a global context.

    Global context resource manager, using this class to
    register all resources.

    Args:
        resources: A dict mapping resource key to resource object.
    """

    _current = None

    def __init__(self, resources: Optional[Dict[Hashable, Any]] = None):
        self._old_scope = None
        if resources is None:
            resources = {}
        self._resources = resources

    def add_resource(self, key, resource, overwrite=False):
        if key in self._resources:
            assert (
                overwrite
            ), f"{key} already exists, please set overwrite=True if you want to overwrite it"  # noqa
        self._resources[key] = resource

    @property
    def resources(self):
        return self._resources

    @resources.setter
    def resources(self, value: Optional[Dict] = None):
        if value is None:
            value = {}
        assert isinstance(value, dict)
        self._resources = value

    def __enter__(self):
        self._old_scope = ContextResourceManager._current
        if self._old_scope is not None:
            resources = copy.copy(self._old_scope.resources)
            resources.update(self.resources)
            self.resources = resources
        ContextResourceManager._current = self
        return self

    def __exit__(self, ptype, value, trace):
        ContextResourceManager._current = self._old_scope

    def get(self, key, allow_missing=False, default=None):
        if key in self.resources:
            return self.resources[key]
        elif allow_missing:
            return default
        else:
            raise KeyError(f"Missing resource {key}")

    def __contains__(self, item):
        return item in self._resources


def get_current_resource_manager():
    return ContextResourceManager._current


def is_active():
    return get_current_resource_manager() is not None


def get_resource(key: Hashable, allow_missing=False, default=None):
    """
    Get resource object.

    Get resource object from :py:class:`ContextResourceManager`
    with giving resource key.

    Args:
        key: Key of resource object.
    """
    manager = get_current_resource_manager()
    assert (
        manager is not None
    ), "Please running the code within the `ContextResourceManager` scope"
    return manager.get(key, allow_missing=allow_missing, default=default)
