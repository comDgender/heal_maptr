"""Tests for heal.models.base_modules.extend_container."""

import pytest


def test_ext_sequential_forward():
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn

    from heal.models.base_modules.extend_container import ExtSequential

    seq = ExtSequential([nn.Conv2d(3, 8, 3, 1, 1), nn.BatchNorm2d(8)])
    x = torch.randn(1, 3, 16, 16)
    y = seq(x)
    assert y.shape == (1, 8, 16, 16)


def test_ext_sequential_fuse_model():
    pytest.importorskip("torch")
    import torch.nn as nn

    from heal.models.base_modules.extend_container import ExtSequential

    seq = ExtSequential([nn.Conv2d(3, 8, 3, 1, 1)])
    # fuse_model should not raise even without quantized layers
    seq.fuse_model()
