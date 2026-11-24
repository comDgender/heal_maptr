"""Tests for heal.models.losses.giou_loss."""

import pytest


def test_giou_loss_forward():
    pytest.importorskip("torch")
    import torch

    from heal.models.losses.giou_loss import GIoULoss

    loss = GIoULoss("loss_reg", reduction="mean")
    pred = torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0]])
    target = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0]])
    out = loss(pred, target)
    assert "loss_reg" in out
    assert out["loss_reg"] is not None
