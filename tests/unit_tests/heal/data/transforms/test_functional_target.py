"""Tests for heal.data.transforms.functional_target."""

import pytest


def test_one_hot_maps_negative_to_zeros():
    pytest.importorskip("torch")
    import torch

    from heal.data.transforms.functional_target import one_hot

    data = torch.tensor(
        [
            [[[0.0]]],
            [[[2.0]]],
        ],
        dtype=torch.float32,
    )
    out = one_hot(data, num_classes=3)
    assert out.shape[0] == 2
    assert out.shape[1] == 3
