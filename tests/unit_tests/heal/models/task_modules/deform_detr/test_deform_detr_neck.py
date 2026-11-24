"""Tests for heal.models.task_modules.deform_detr.neck."""

import pytest


def test_channel_mapper_neck_forward():
    pytest.importorskip("torch")
    import torch

    from heal.models.task_modules.deform_detr.neck import ChannelMapperNeck

    neck = ChannelMapperNeck(
        in_channels=[16, 32, 64],
        out_indices=[0, 1, 2],
        out_channel=32,
        kernel_size=1,
        stride=1,
    )
    feats = [
        torch.randn(1, 16, 16, 16),
        torch.randn(1, 32, 8, 8),
        torch.randn(1, 64, 4, 4),
    ]
    outs = neck(feats)
    assert len(outs) == 3
    for out in outs:
        assert out.shape[1] == 32
