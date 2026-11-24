"""Tests for heal.models.losses.softmax_focal_loss."""

import pytest


def test_softmax_focal_loss_forward():
    pytest.importorskip("torch")
    import torch

    from heal.models.losses.softmax_focal_loss import SoftmaxFocalLoss

    loss = SoftmaxFocalLoss("loss_seg", num_classes=3, reduction="mean")
    logits = torch.randn(2, 3, 8, 8, requires_grad=True)
    label = torch.zeros(2, 3, 8, 8)
    label[:, 0, :, :] = 1.0
    out = loss(logits, label)
    assert "loss_seg" in out
    assert out["loss_seg"].numel() == 1
