# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import Any

__all__ = ["MockFrame", "visualize_stage2_gif"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import vis_help

        return getattr(vis_help, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
