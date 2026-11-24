"""Tests for heal.models.base_modules.bifpn."""

import pytest


def test_get_fpn_config_returns_namespace():
    from heal.models.base_modules.bifpn import get_fpn_config

    cfg = get_fpn_config("bifpn_sum", out_channels=32, upsample_type="module")
    assert cfg.level == 5
    assert len(cfg.nodes) > 0


def test_bifpn_forward_runs():
    pytest.importorskip("torch")
    import torch

    from heal.models.base_modules.bifpn import BiFPN

    # BiFPN topology uses 5 levels (fpn_config.level == 5).
    stride2channels = {8: 16, 16: 24, 32: 32, 64: 40, 128: 48}
    strides = [8, 16, 32, 64, 128]
    neck = BiFPN(
        in_strides=strides,
        out_strides=strides,
        stride2channels=stride2channels,
        out_channels=16,
        num_outs=5,
        stack=1,
    ).eval()
    img = 128
    feats = [
        torch.randn(1, stride2channels[s], img // s, img // s) for s in strides
    ]
    out = neck(feats)
    assert isinstance(out, list)
    assert len(out) == 5
