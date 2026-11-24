# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

try:
    import cloudpickle
    import dill

    dill.extend(False)
except ImportError:
    cloudpickle = None
    dill = None

from hat_engine.utils.package_helper import require_packages

__all__ = ["set_serializer", "get_serializer"]


_serializer = "dill"


@require_packages("dill", "cloudpickle")
def set_serializer(serializer):
    """Set serializer."""
    global _serializer
    assert serializer in [
        "dill",
        "cloudpickle",
        dill,
        cloudpickle,
    ], "Currently, only support dill and cloudpickle"
    _serializer = serializer


@require_packages("dill", "cloudpickle")
def get_serializer(serializer=None, return_name=False):
    """Get serializer.

    Args:
        serializer: serializer, if None, using preset serializer,
            by default None
        return_name: If true, return the serializer module name,
            otherwise return the
            serializer module, by default False

    Returns:
        serializer module or serializer module name
    """
    if serializer is None:
        global _serializer
        serializer = _serializer

    obj2serializer = {
        "dill": dill,
        "cloudpickle": cloudpickle,
        dill: dill,
        cloudpickle: cloudpickle,
    }

    serializer = obj2serializer[serializer]

    if return_name:
        return serializer.__name__
    else:
        return serializer
