# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import Dict

import torch
from hat_engine.registry import OBJECT_REGISTRY

try:
    from nuplan.planning.simulation.trajectory.trajectory_sampling import (
        TrajectorySampling,
    )
except ImportError:
    TrajectorySampling = None

try:
    from navsim.planning.training.abstract_feature_target_builder import (
        AbstractTargetBuilder,
    )

    _HAS_NAVSIM = True
except ImportError:
    AbstractTargetBuilder = object
    _HAS_NAVSIM = False


@OBJECT_REGISTRY.register
class TrajectoryTargetBuilder(AbstractTargetBuilder):
    def __init__(self, trajectory_sampling: "TrajectorySampling"):
        self._trajectory_sampling = trajectory_sampling

    def get_unique_name(self) -> str:
        return "trajectory_target"

    def compute_targets(self, scene) -> Dict[str, torch.Tensor]:
        future_trajectory = scene.get_future_trajectory(
            num_trajectory_frames=self._trajectory_sampling.num_poses
        )
        return {"trajectory": torch.tensor(future_trajectory.poses)}
