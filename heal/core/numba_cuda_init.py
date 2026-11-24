# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

"""Configure numba CUDA for CUDA 12 and older drivers."""

import importlib.util
import os

# Set before numba.cuda import (numba 0.60 MVC).
os.environ.setdefault("NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY", "1")

_patched = False


def configure_numba_cuda() -> None:
    """Enable CUDA Minor Version Compatibility / nvjitlink for numba.

    Call before any numba.cuda JIT compile.
    """
    global _patched
    if _patched:
        return

    os.environ["NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY"] = "1"

    from numba import config

    config.CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY = True

    if importlib.util.find_spec("numba_cuda") is not None:
        os.environ.setdefault("NUMBA_CUDA_ENABLE_PYNVJITLINK", "1")
        config.CUDA_ENABLE_PYNVJITLINK = True
    else:
        try:
            from pynvjitlink.patch import patch_numba_linker
        except ImportError:
            from pynvjitlink import patch_numba_linker  # older releases

        patch_numba_linker()
    _patched = True


__all__ = ["configure_numba_cuda"]
