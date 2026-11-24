"""Tests for heal.models.losses.smooth_l1_loss."""

import pytest


def test_smooth_l1_loss_forward():
    pytest.importorskip("torch")
    import torch

    from heal.models.losses.smooth_l1_loss import SmoothL1Loss

    loss = SmoothL1Loss(reduction="mean")
    pred = torch.randn(4, 7, requires_grad=True)
    target = torch.randn(4, 7)
    out = loss(pred, target)
    assert out.numel() == 1
    assert out.item() >= 0
