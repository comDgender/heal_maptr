# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from hat_engine.registry import OBJECT_REGISTRY

from .blocks import (
    AsymmetricFFN,
    DeformableFeatureAggregation,
    DeformableFeatureAggregationv2,
    DenseDepthNet,
)
from .decoder import SparseBEVDecoder
from .det_blocks import (
    SparseBEVEncoder,
    SparseBEVKeyPointsGenerator,
    SparseBEVRefinementModule,
)
from .head import SparseBEVHead
from .instance_bank import InstanceBank
from .memory_bank import MemoryBank
from .target import SparseBEVTarget

__all__ = [
    "SparseBEVHead",
    "SparseBEVTarget",
    "SparseBEVDecoder",
    "DeformableFeatureAggregation",
    "DeformableFeatureAggregationv2",
    "AsymmetricFFN",
    "DenseDepthNet",
    "SparseBEVKeyPointsGenerator",
    "SparseBEVRefinementModule",
    "SparseBEVEncoder",
    "InstanceBank",
    "MemoryBank",
]

# Backward-compatible aliases for sparse MapTR / sparsebevoe configs.
OBJECT_REGISTRY.alias("InstanceBankOE")(InstanceBank)
OBJECT_REGISTRY.alias("MemoryBankOE")(MemoryBank)
OBJECT_REGISTRY.alias("AsymmetricFFNOE")(AsymmetricFFN)
OBJECT_REGISTRY.alias("DeformableFeatureAggregationOE")(
    DeformableFeatureAggregation
)
OBJECT_REGISTRY.alias("DeformableFeatureAggregationOEv2")(
    DeformableFeatureAggregationv2
)
OBJECT_REGISTRY.alias("DenseDepthNetOE")(DenseDepthNet)
OBJECT_REGISTRY.alias("SparseBEVOEHead")(SparseBEVHead)
OBJECT_REGISTRY.alias("SparseBEVOEDecoder")(SparseBEVDecoder)
OBJECT_REGISTRY.alias("SparseBEVOEEncoder")(SparseBEVEncoder)
OBJECT_REGISTRY.alias("SparseBEVOEKeyPointsGenerator")(
    SparseBEVKeyPointsGenerator
)
OBJECT_REGISTRY.alias("SparseBEVOERefinementModule")(SparseBEVRefinementModule)
OBJECT_REGISTRY.alias("SparseBEVOETarget")(SparseBEVTarget)
