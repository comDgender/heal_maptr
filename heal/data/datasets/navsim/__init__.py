# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

# Perception / planner-cache registry types (no hydra).
from heal.data.datasets.navsim.navsim_e2e_dataset import (  # noqa: F401
    NavsimE2EDataset,
    NavsimE2EJointDataset,
)
from heal.data.datasets.navsim.navsim_feature_builder import (  # noqa: F401
    DiffusionPlannerFeatureBuilder,
    DiffusionPlannerLaneBoundaryFeatureBuilder,
)
from heal.data.datasets.navsim.navsim_perception_cache_wrapper import (  # noqa: F401
    NavsimPerceptionCacheWrapper,
)
from heal.data.datasets.navsim.navsim_perception_dataset import (  # noqa: F401
    NavsimPerceptionMultitaskDataset,
)
from heal.data.datasets.navsim.navsim_planner_cache_dataset import (  # noqa: F401
    NavsimPerceptionFeatureCacheDataset,
    NavsimPlannerCacheDataset,
)
from heal.data.datasets.navsim.trajectory_target_builder import (  # noqa: F401
    TrajectoryTargetBuilder,
)

__all__ = [
    "DiffusionPlannerFeatureBuilder",
    "DiffusionPlannerLaneBoundaryFeatureBuilder",
    "NavsimE2EDataset",
    "NavsimE2EJointDataset",
    "NavsimPerceptionCacheWrapper",
    "NavsimPerceptionFeatureCacheDataset",
    "NavsimPerceptionMultitaskDataset",
    "NavsimPlannerCacheDataset",
    "TrajectoryTargetBuilder",
]


def __getattr__(name: str):
    # Hydra-backed NavsimDataset — only when explicitly requested.
    if name == "NavsimDataset":
        from heal.data.datasets.navsim.navsim_dataset import NavsimDataset

        return NavsimDataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
