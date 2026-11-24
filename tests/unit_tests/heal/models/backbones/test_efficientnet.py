"""Tests for heal.models.backbones.efficientnet."""

import pytest


def test_round_filters_and_repeats():
    from heal.models.backbones.efficientnet import round_filters, round_repeats

    assert round_filters(32, 1.0, 8) % 8 == 0
    assert round_repeats(3, 1.0) >= 3


def test_efficientnet_forward_feature_pyramid():
    pytest.importorskip("torch")
    pytest.importorskip("horizon_plugin_pytorch")
    import torch

    from heal.models.backbones.efficientnet import EfficientNet

    model = EfficientNet(
        model_type="b0tiny",
        coefficient_params=(0.5, 0.5, 224, 0.2),
        num_classes=10,
        include_top=False,
        quant_input=False,
    ).eval()
    x = torch.randn(1, 3, 128, 128)
    outs = model(x)
    assert isinstance(outs, list)
    assert len(outs) > 0
