# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501


def test_nuscenes_dataset_constants():
    from heal.data.datasets.nuscenes.nuscenes_dataset import (
        CAM_NAMES,
        CLASSES,
        NameMapping,
    )

    assert len(CAM_NAMES) == 6
    assert "CAM_FRONT" in CAM_NAMES
    assert len(CLASSES) == 10
    assert NameMapping["vehicle.car"] == "car"
