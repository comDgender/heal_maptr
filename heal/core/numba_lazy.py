# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from __future__ import annotations
import functools
import sys
import types
import warnings
from typing import Any, Callable, Optional, Set

# Wrappers currently inside ``get_compiled`` — breaks A↔B materialize cycles.
_compiling: Set[int] = set()


def _import_numba():
    import numba

    try:
        from numba.core.errors import (
            NumbaDeprecationWarning,
            NumbaPerformanceWarning,
            NumbaWarning,
        )

        warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
        warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
        warnings.filterwarnings("ignore", category=NumbaWarning)
    except ImportError:
        pass
    return numba


def _patch_module_attr(fn: Callable, compiled: Any, wrapper: Callable) -> None:
    mod = sys.modules.get(fn.__module__)
    if mod is None:
        return
    name = fn.__name__
    if getattr(mod, name, None) is wrapper:
        setattr(mod, name, compiled)


def _materialize_lazy_in_obj(obj: Any) -> None:
    get_compiled = getattr(obj, "get_compiled", None)
    if not callable(get_compiled):
        return
    if id(obj) in _compiling:
        return
    get_compiled()


def _materialize_lazy_callees(fn: Callable) -> None:
    """Compile lazy helpers visible via ``fn`` globals."""
    for obj in fn.__globals__.values():
        if isinstance(obj, types.ModuleType):
            # Only heal.* — avoid scanning huge third-party modules.
            if not (obj.__name__ or "").startswith("heal."):
                continue
            for val in list(vars(obj).values()):
                _materialize_lazy_in_obj(val)
        else:
            _materialize_lazy_in_obj(obj)


def lazy_njit(func: Optional[Callable] = None, **njit_kwargs):
    """Drop-in replacement for ``@numba.njit`` / ``@numba.njit(...)``."""

    def decorator(fn: Callable) -> Callable:
        compiled: Any = None

        def get_compiled() -> Any:
            nonlocal compiled
            if compiled is None:
                compiled = _import_numba().njit(**njit_kwargs)(fn)
                _patch_module_attr(fn, compiled, wrapper)
            return compiled

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return get_compiled()(*args, **kwargs)

        wrapper.get_compiled = get_compiled  # type: ignore[attr-defined]
        wrapper.py_func = fn  # type: ignore[attr-defined]
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator


def lazy_jit(func: Optional[Callable] = None, **jit_kwargs):
    """Drop-in replacement for ``@numba.jit`` / ``@numba.jit(...)``."""

    def decorator(fn: Callable) -> Callable:
        compiled: Any = None

        def get_compiled() -> Any:
            nonlocal compiled
            if compiled is not None:
                return compiled
            wid = id(wrapper)
            if wid in _compiling:
                # Cycle: a nested materialize re-hit this wrapper.
                return wrapper
            _compiling.add(wid)
            try:
                numba = _import_numba()
                # Materialize callees so njit→njit typing sees Dispatchers.
                _materialize_lazy_callees(fn)
                compiled = numba.jit(**jit_kwargs)(fn)
                _patch_module_attr(fn, compiled, wrapper)
            finally:
                _compiling.discard(wid)
            return compiled

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return get_compiled()(*args, **kwargs)

        wrapper.get_compiled = get_compiled  # type: ignore[attr-defined]
        wrapper.py_func = fn  # type: ignore[attr-defined]
        return wrapper

    if func is not None:
        return decorator(func)
    return decorator
