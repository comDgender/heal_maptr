"""Tests for heal.models.losses.laplace_loss."""

import pytest


def test_laplace_nll_loss_importable():
    from heal.models.losses.laplace_loss import LaplaceNLLLoss

    assert LaplaceNLLLoss is not None


def test_laplace_nll_loss_forward():
    pytest.importorskip("torch")
    import torch

    from heal.models.losses.laplace_loss import LaplaceNLLLoss

    loss_fn = LaplaceNLLLoss(reduction="mean")
    # pred holds loc and scale concatenated on the last dim.
    pred = torch.cat([torch.zeros(4, 2), torch.ones(4, 2)], dim=-1)
    target = torch.zeros(4, 2)
    loss = loss_fn(pred, target)
    assert loss.dim() == 0
    # nll = log(2 * 1) + |0 - 0| / 1 = log(2)
    assert torch.allclose(loss, torch.log(torch.tensor(2.0)), atol=1e-5)

    none_loss = LaplaceNLLLoss(reduction="none")(pred, target)
    assert none_loss.shape == (4, 2)
