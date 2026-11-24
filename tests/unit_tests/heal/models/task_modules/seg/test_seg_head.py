"""Tests for heal.models.task_modules.seg.head."""

import pytest


def test_seg_head_forward():
    pytest.importorskip("torch")
    pytest.importorskip("horizon_plugin_pytorch")
    import torch

    from heal.models.task_modules.seg.head import SegHead

    head = SegHead(
        num_classes=3,
        in_strides=[4, 8],
        out_strides=[4],
        stride2channels={4: 16, 8: 32},
        stacked_convs=1,
        feat_channels=16,
        dequant_output=False,
    ).eval()
    feats = [torch.randn(1, 16, 32, 32), torch.randn(1, 32, 16, 16)]
    out = head(feats)
    assert out is not None
    assert hasattr(out, "shape") or (
        isinstance(out, (list, tuple)) and len(out) >= 1
    )
