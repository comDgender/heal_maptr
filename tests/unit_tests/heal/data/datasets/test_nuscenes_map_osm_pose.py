# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import numpy as np

from heal.data.datasets.nuscenes.nuscenes_map_dataset import NuscenesMapDataset


def _make_dataset(use_lidar_gt: bool) -> NuscenesMapDataset:
    ds = object.__new__(NuscenesMapDataset)
    ds.use_lidar_gt = use_lidar_gt
    ds.patch_size = (30, 60)
    return ds


def test_map_patch_from_example_uses_ego_pose():
    ds = _make_dataset(use_lidar_gt=False)
    example = {
        "meta": {
            "location": "boston-seaport",
            "ego2global_translation": [100.0, 200.0, 0.5],
            "ego2global_rotation": [1.0, 0.0, 0.0, 0.0],
        },
    }
    location, patch_box, _ = ds._map_patch_from_example(example)
    assert location == "boston-seaport"
    assert patch_box[:2] == (100.0, 200.0)
    assert patch_box[2:] == (30, 60)


def test_map_patch_from_example_uses_lidar_pose():
    ds = _make_dataset(use_lidar_gt=True)
    example = {
        "meta": {"location": "boston-seaport"},
        "lidar2global": np.array(
            [
                [1.0, 0.0, 0.0, 101.0],
                [0.0, 1.0, 0.0, 201.0],
                [0.0, 0.0, 1.0, 1.5],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
    }
    location, patch_box, _ = ds._map_patch_from_example(example)
    assert location == "boston-seaport"
    assert patch_box[:2] == (101.0, 201.0)
