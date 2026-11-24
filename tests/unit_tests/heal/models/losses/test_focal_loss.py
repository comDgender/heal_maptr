"""Tests for heal.models.losses.focal_loss."""

import pytest


def test_focal_loss_forward_dict():
    pytest.importorskip("torch")
    import torch

    from heal.models.losses.focal_loss import FocalLoss

    loss = FocalLoss("loss_cls", num_classes=3, reduction="mean")
    pred = torch.randn(8, 2, requires_grad=True)
    target = torch.tensor([0, 1, 0, 2, -1, 0, 1, 0])
    out = loss(pred, target)
    assert "loss_cls" in out
    assert out["loss_cls"].numel() == 1


def test_focal_loss_one_hot_returns_tensor():
    pytest.importorskip("torch")
    import torch

    from heal.models.losses.focal_loss import FocalLoss

    loss = FocalLoss(
        alpha=0.25,
        gamma=2.0,
        from_logits=False,
        reduction="none",
        loss_weight=1.0,
    )
    pred = torch.rand(2, 4, 3, requires_grad=True)
    target = torch.zeros_like(pred)
    target[:, :, 0] = 1.0
    out = loss(pred, target)
    assert isinstance(out, torch.Tensor)
    assert out.shape == pred.shape


def test_focal_loss_extra_classes_importable():
    from heal.models.losses.focal_loss import GaussianFocalLoss

    assert GaussianFocalLoss is not None
