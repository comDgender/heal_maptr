"""Tests for heal.models.base_modules.second_neck."""

import pytest


def test_second_neck_forward():
    pytest.importorskip("torch")
    import torch

    from heal.models.base_modules.second_neck import SECONDNeck

    neck = SECONDNeck(
        in_feature_channel=64,
        down_layer_nums=[1, 1],
        down_layer_strides=[1, 2],
        down_layer_channels=[64, 128],
        up_layer_strides=[1, 2],
        up_layer_channels=[64, 64],
        quantize=False,
    ).eval()
    x = torch.randn(1, 64, 16, 16)
    out = neck(x)
    assert out.shape == (1, 128, 16, 16)
