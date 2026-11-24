"""Tests for heal.models.losses __init__."""


def test_losses_public_api():
    from heal.models.losses import FocalLoss, GIoULoss, SoftmaxFocalLoss

    assert FocalLoss is not None
    assert GIoULoss is not None
    assert SoftmaxFocalLoss is not None
