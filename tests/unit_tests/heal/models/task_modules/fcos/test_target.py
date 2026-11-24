"""Tests for heal.models.task_modules.fcos.target helpers."""

import pytest


def test_get_points_shapes():
    pytest.importorskip("torch")
    import torch

    from heal.models.task_modules.fcos.target import get_points

    pts = get_points(
        [(2, 3), (1, 2)], [4, 8], torch.float32, torch.device("cpu")
    )
    assert len(pts) == 2
    assert pts[0].shape == (6, 2)


def test_distance2bbox_clamp():
    pytest.importorskip("torch")
    import torch

    from heal.models.task_modules.fcos.target import distance2bbox

    p = torch.tensor([[4.0, 4.0]])
    d = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    box = distance2bbox(p, d, max_shape=(10, 10))
    assert box.shape == (1, 4)
