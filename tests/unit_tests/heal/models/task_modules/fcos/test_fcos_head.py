"""Tests for heal.models.task_modules.fcos.head."""

import pytest


def test_fcos_head_forward():
    pytest.importorskip("torch")
    import torch

    from heal.models.task_modules.fcos.head import FCOSHead

    # FCOSHead uses stride2channels[in_strides[0]] for all level input widths.
    head = FCOSHead(
        num_classes=3,
        in_strides=[8, 16],
        out_strides=[8, 16],
        stride2channels={8: 32, 16: 32},
        upscale_bbox_pred=False,
        stacked_convs=1,
        feat_channels=32,
    ).eval()
    feats = [torch.randn(1, 32, 16, 16), torch.randn(1, 32, 8, 8)]
    cls_scores, bbox_preds, centernesses = head(feats)
    assert len(cls_scores) == 2
    assert len(bbox_preds) == 2
    assert len(centernesses) == 2
