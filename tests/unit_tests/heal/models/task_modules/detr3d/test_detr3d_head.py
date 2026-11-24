"""Tests for heal.models.task_modules.detr3d.head."""

import pytest


def test_detr3d_head_importable():
    from heal.models.task_modules.detr3d.head import (
        FFN,
        Detr3dCrossAtten,
        Detr3dDecoder,
        Detr3dDecoderLayer,
        Detr3dHead,
        Detr3dTransformer,
        FeatureSampler,
    )

    assert Detr3dHead is not None
    assert Detr3dTransformer is not None
    assert Detr3dDecoder is not None
    assert Detr3dDecoderLayer is not None
    assert Detr3dCrossAtten is not None
    assert FeatureSampler is not None
    assert FFN is not None


def test_detr3d_ffn_forward():
    pytest.importorskip("torch")
    import torch

    from heal.models.task_modules.detr3d.head import FFN

    ffn = FFN(embed_dims=16, feedforward_channels=32, ffn_drop=0.0)
    x = torch.randn(2, 16, 4, 8)
    out = ffn(x)
    assert out.shape == (2, 16, 4, 8)
