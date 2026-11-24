# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

"""Sparse MapTR task modules: head, decoder, and anchor geometry blocks."""

from hat_engine.registry import OBJECT_REGISTRY

from .blocks import (
    SparsePoint3DEncoder,
    SparsePoint3DKeyPointsGenerator,
    SparsePoint3DRefinementModule,
)
from .decoder import SparseMapPerceptionDecoder
from .head import SparseMapHead

__all__ = [
    "SparsePoint3DEncoder",
    "SparsePoint3DKeyPointsGenerator",
    "SparsePoint3DRefinementModule",
    "SparseMapHead",
    "SparseMapPerceptionDecoder",
]

OBJECT_REGISTRY.alias("SparseOMOEHead")(SparseMapHead)
OBJECT_REGISTRY.alias("SparseOEPoint3DEncoder")(SparsePoint3DEncoder)
