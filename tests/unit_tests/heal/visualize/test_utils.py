"""Tests for heal.visualize.utils."""

import pytest


def test_colormap_tensor_branch():
    pytest.importorskip("torch")
    import torch

    from heal.visualize.utils import colormap

    cmap = torch.zeros(5, 3)
    gray = torch.tensor([0, 1, 2, 0, 1], dtype=torch.float32)
    rgb = colormap(gray, cmap, scale=1)
    assert rgb.shape[-1] == 3
