"""Tests for heal.models.base_modules.dw_unet."""

import pytest


def test_dw_unet_forward_single_scale_output():
    pytest.importorskip("torch")
    pytest.importorskip("horizon_plugin_pytorch")
    import torch

    from heal.models.base_modules.dw_unet import DwUnet

    neck = DwUnet(base_channels=1, output_scales=(64,)).eval()
    x64 = torch.randn(1, 32, 4, 4)
    out = neck([x64])
    assert len(out) == 1
    assert out[0].dim() == 4
