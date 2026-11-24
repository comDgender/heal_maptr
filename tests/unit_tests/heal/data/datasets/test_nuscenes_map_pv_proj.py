# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import numpy as np

from heal.data.datasets.nuscenes.nuscenes_map_dataset import (
    PV_PROJ_EGO_GROUND_Z,
    pv_proj_ground_z,
)


def test_pv_proj_ground_z_ego_frame():
    assert pv_proj_ground_z(use_lidar_gt=False) == PV_PROJ_EGO_GROUND_Z


def test_pv_proj_ground_z_lidar_frame_uses_lidar2ego():
    lidar2ego = np.eye(4, dtype=np.float64)
    lidar2ego[:3, 3] = [0.985, 0.0, 1.840]
    z_lidar = pv_proj_ground_z(use_lidar_gt=True, lidar2ego=lidar2ego)
    assert z_lidar != PV_PROJ_EGO_GROUND_Z
    assert z_lidar < 0
