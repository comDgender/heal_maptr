"""Tests for heal.models.task_modules.ganet.head."""

import pytest


def test_ganet_head_forward():
    pytest.importorskip("torch")
    import torch

    from heal.models.task_modules.ganet.head import GaNetHead

    head = GaNetHead(in_channel=16).eval()
    feat = torch.randn(1, 16, 16, 32)
    kpts_hm, pts_offset, int_offset = head(feat)
    assert kpts_hm.shape == (1, 1, 16, 32)
    assert pts_offset.shape == (1, 2, 16, 32)
    assert int_offset.shape == (1, 2, 16, 32)
