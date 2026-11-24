# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

__all__ = [
    "NavsimMetric",
    "NavsimPerceptionMetric",
    "NavsimPerceptionTrackMetric",
    "PerceptionInferCacheProcess",
]


def __getattr__(name: str):
    if name == "NavsimMetric":
        from .navsim_metric import NavsimMetric

        return NavsimMetric
    if name in (
        "NavsimPerceptionMetric",
        "NavsimPerceptionTrackMetric",
        "PerceptionInferCacheProcess",
    ):
        from . import navsim_perception_metric as _npm

        return getattr(_npm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
