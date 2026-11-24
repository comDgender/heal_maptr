"""Tests for heal.data.collates.nusc_collates."""

import pytest


def test_collate_nuscenes_stacks_camera_tensors():
    pytest.importorskip("torch")
    import numpy as np

    from heal.data.collates.nusc_collates import collate_nuscenes

    batch = [
        {
            "img": [np.ones((3, 4, 4), dtype=np.float32)],
            "lidar2img": [np.eye(4, dtype=np.float32)],
            "sample_token": "a",
        },
        {
            "img": [np.zeros((3, 4, 4), dtype=np.float32)],
            "lidar2img": [np.eye(4, dtype=np.float32) * 2],
            "sample_token": "b",
        },
    ]
    out = collate_nuscenes(batch)
    assert out["img"].shape == (2, 3, 4, 4)
    assert out["lidar2img"].shape == (2, 4, 4)
    assert out["sample_token"] == ["a", "b"]


def test_collate_nuscenes_sequence_builds_seq_meta():
    pytest.importorskip("torch")
    import numpy as np

    from heal.data.collates.nusc_collates import collate_nuscenes_sequence

    sample = {
        "img": [np.ones((3, 2, 2), dtype=np.float32)],
        "lidar2img": [np.eye(4, dtype=np.float32)],
        "sample_token": "s0",
    }
    out = collate_nuscenes_sequence([[sample], [sample]])
    assert out["img"].shape[0] == 2
    assert "seq_meta" in out
    assert out["seq_meta"][0]["sample_token"] == ["s0", "s0"]


def test_collate_bevfusion_promotes_lidar_points_and_builds_seq_meta():
    pytest.importorskip("torch")
    import numpy as np

    from heal.data.collates.nusc_collates import collate_bevfusion

    sample = {
        "img": [np.ones((3, 2, 2), dtype=np.float32)],
        "lidar2img": [np.eye(4, dtype=np.float32)],
        "lidar": {"points": np.ones((4, 5), dtype=np.float32)},
        "sample_token": "s0",
    }
    out = collate_bevfusion([sample, sample])
    assert out["img"].shape[0] == 2
    assert "seq_meta" in out
    assert out["seq_meta"][0]["sample_token"] == ["s0", "s0"]
