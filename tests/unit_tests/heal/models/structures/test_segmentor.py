"""Tests for heal.models.structures.segmentor."""

import pytest


def test_segmentor_forward_with_dummy_modules():
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn

    from heal.models.structures.segmentor import Segmentor

    class BB(nn.Module):
        def forward(self, x):
            return [torch.randn(1, 8, 8, 8)]

    class Neck(nn.Module):
        def forward(self, feats):
            return feats[0]

    class Head(nn.Module):
        def forward(self, x):
            return torch.randn(1, 2, x.shape[-2], x.shape[-1])

    model = Segmentor(BB(), Neck(), Head()).eval()
    data = {"img": torch.randn(1, 3, 32, 32)}
    pred = model(data)
    assert pred.shape[0] == 1
