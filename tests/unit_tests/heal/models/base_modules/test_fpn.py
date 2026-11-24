"""Tests for heal.models.base_modules.fpn."""

import pytest


def test_fpn_forward_bev_like_config():
    pytest.importorskip("torch")
    import torch

    from heal.models.base_modules.fpn import FPN

    neck = FPN(
        in_strides=[2, 4, 8, 16, 32],
        in_channels=[64, 256, 512, 1024, 2048],
        out_strides=[16, 32],
        out_channels=[1024, 2048],
        fix_out_channel=256,
        bn_kwargs={"eps": 1e-5, "momentum": 0.1},
    ).eval()
    img = 512
    feats = [
        torch.randn(1, ch, img // s, img // s)
        for s, ch in zip([2, 4, 8, 16, 32], [64, 256, 512, 1024, 2048])
    ]
    out = neck(feats)
    assert len(out) == 2
    assert out[0].shape[1] == 256
    assert out[1].shape[1] == 256
