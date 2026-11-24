"""Tests for heal.models.task_modules.pointpillars.head."""

import pytest


def test_pointpillars_head_forward():
    pytest.importorskip("torch")
    import torch

    from heal.models.task_modules.pointpillars.head import PointPillarsHead

    head = PointPillarsHead(
        in_channels=128,
        num_classes=1,
        use_direction_classifier=True,
    ).eval()
    feat = torch.randn(1, 128, 8, 8)
    box_preds, cls_preds, dir_preds = head(feat)
    assert box_preds.shape == (1, 14, 8, 8)
    assert cls_preds.shape == (1, 2, 8, 8)
    assert dir_preds.shape == (1, 4, 8, 8)
