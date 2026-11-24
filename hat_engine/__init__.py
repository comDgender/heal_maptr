# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from . import registry
from .version import get_tmp_version

try:
    from ._version import __version__
except Exception:
    __version__ = get_tmp_version()
