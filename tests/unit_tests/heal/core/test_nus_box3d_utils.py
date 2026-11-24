"""Tests for heal.core.nus_box3d_utils."""

import pytest


def test_get_min_max_coords():
    from heal.core.nus_box3d_utils import get_min_max_coords

    min_x, max_x, min_y, max_y = get_min_max_coords((51.2, 51.2, 0.8))
    assert min_x < max_x
    assert min_y < max_y


def test_bbox_bev_ego_roundtrip():
    pytest.importorskip("numpy")
    import numpy as np

    from heal.core.nus_box3d_utils import bbox_bev2ego, bbox_ego2bev

    bev_size = (51.2, 51.2, 0.8)
    bbox = np.array([10.0, 20.0, 1.0, 4.0, 2.0, 1.5, 0.1, 0.2, 0.3, 0.9])
    ego = bbox_bev2ego([bbox], bev_size)[0]
    back = bbox_ego2bev([ego], bev_size)[0]
    assert np.allclose(bbox[:6], back[:6], atol=1e-4)


def test_adjust_coords_subtracts_grid():
    pytest.importorskip("torch")
    import torch

    from heal.core.nus_box3d_utils import adjust_coords

    coords = torch.ones(4, 4, 2)
    out = adjust_coords(coords, grid_size=(4, 4))
    assert out.shape == coords.shape
    assert not torch.allclose(out, coords)


def test_bbox_to_corner_filters_by_score():
    pytest.importorskip("nuscenes")
    pytest.importorskip("numpy")
    import numpy as np

    from heal.core.nus_box3d_utils import bbox_to_corner

    boxes = np.array(
        [
            [0, 0, 0, 4, 2, 1.5, 0.0, 0, 0, 0.05, 1],
            [1, 1, 0, 4, 2, 1.5, 0.0, 0, 0, 0.9, 2],
        ]
    )
    corners, scores, cat_ids = bbox_to_corner(boxes, score_thresh=0.1)
    assert len(corners) == 1
    assert len(scores) == 1
    assert cat_ids[0] == 2
