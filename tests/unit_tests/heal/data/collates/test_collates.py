"""Tests for heal.data.collates.collates."""

import pytest


def test_default_collate_v2_stacks_tensors():
    pytest.importorskip("torch")
    import torch

    from heal.data.collates.collates import default_collate_v2

    batch = [torch.ones(2, 3), torch.zeros(2, 3)]
    out = default_collate_v2(batch)
    assert out.shape == (2, 2, 3)


def test_collate_2d_keeps_list_keys():
    pytest.importorskip("torch")
    import torch

    from heal.data.collates.collates import collate_2d

    b1 = {"img": torch.zeros(1, 1, 4, 4), "img_shape": (10, 20)}
    b2 = {"img": torch.ones(1, 1, 4, 4), "img_shape": (11, 21)}
    out = collate_2d([b1, b2])
    assert out["img"].shape[0] == 2
    assert isinstance(out["img_shape"], list)
    assert len(out["img_shape"]) == 2


def test_collate_lidar3d_batches_points():
    pytest.importorskip("torch")
    import numpy as np

    from heal.data.collates.collates import collate_lidar3d

    batch = [
        {
            "points": np.random.rand(10, 4).astype(np.float32),
            "num_gt": np.array([1]),
        },
        {
            "points": np.random.rand(12, 4).astype(np.float32),
            "num_gt": np.array([2]),
        },
    ]
    out = collate_lidar3d(batch)
    assert isinstance(out, dict)
    assert "points" in out
    assert len(out["points"]) == 2
    assert out["num_gt"].tolist() == [1, 2]


def test_collate_mot_seq_returns_dict():
    pytest.importorskip("torch")
    import numpy as np

    from heal.data.collates.collates import collate_mot_seq

    frame0 = {
        "img": np.zeros((3, 8, 8), dtype=np.float32),
        "img_shape": (8, 8),
        "layout": "hwc",
        "color_space": "rgb",
        "gt_bboxes": np.zeros((0, 4), dtype=np.float32),
        "gt_labels": np.zeros((0,), dtype=np.int64),
        "gt_classes": np.zeros((0,), dtype=np.int64),
        "gt_ids": np.zeros((0,), dtype=np.int64),
        "seq_name": "seq0",
    }
    frame1 = dict(frame0)
    frame1["img"] = np.ones((3, 8, 8), dtype=np.float32)
    batch = [{"frame_data_list": [frame0, frame1]}]
    out = collate_mot_seq(batch)
    assert isinstance(out, dict)
