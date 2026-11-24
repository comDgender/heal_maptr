"""Tests for heal.models.task_modules.bevformer.utils."""

import pytest


def test_gen_coords_and_min_max():
    pytest.importorskip("torch")

    from heal.models.task_modules.bevformer.utils import (
        gen_coords,
        get_min_max_coords,
    )

    pc_range = [-10.0, -10.0, -1.0, 10.0, 10.0, 1.0]
    coords = gen_coords((4, 4), pc_range)
    assert coords.shape == (1, 4, 4, 2)

    real_range = (pc_range[0], pc_range[1], pc_range[3], pc_range[4])
    grid_resolution = (5.0, 5.0)
    min_x, max_x, min_y, max_y = get_min_max_coords(
        real_range, grid_resolution
    )
    assert min_x < max_x
    assert min_y < max_y


def test_get_clone_module():
    pytest.importorskip("torch")
    import torch.nn as nn

    from heal.models.task_modules.bevformer.utils import get_clone_module

    template = nn.Linear(4, 8)
    clones = get_clone_module(template, 3)
    assert len(clones) == 3
    assert clones[0] is not clones[1]
