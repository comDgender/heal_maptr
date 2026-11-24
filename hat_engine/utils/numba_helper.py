# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

__all__ = ["numba_jit"]


def numba_jit(*args, **kwargs):  # noqa D205, D209, D400, D401
    """Wrapper of numba jit, will automatically choose to use
    `numba.cuda.jit` or `numba.jit` according to the machine environment."""
    import numba
    from numba import cuda

    if cuda.is_available():
        wrapper = numba.cuda.jit(*args, **kwargs)
    else:
        wrapper = numba.jit(nopython=True)

    def wrap_func(func):
        return wrapper(func)

    return wrap_func
