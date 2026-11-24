"""Tests for heal.models.backbones.mixvargenet."""

import pytest


def test_mixvargenet_config_dataclass():
    from heal.models.backbones.mixvargenet import (
        IdentityConfig,
        MixVarGENetConfig,
    )

    cfg = MixVarGENetConfig(
        in_channels=32,
        out_channels=64,
        head_op="mixvarge_f4",
        stack_ops=["mixvarge_f4", "mixvarge_f4"],
        stride=2,
    )
    assert cfg.in_channels == 32
    assert cfg.out_channels == 64
    assert cfg.head_op == "mixvarge_f4"

    id_cfg = IdentityConfig(in_channels=32, out_channels=32, stride=1)
    assert id_cfg.in_channels == 32


def test_get_mixvargenet_stride2channels():
    from heal.models.backbones.mixvargenet import (
        MixVarGENetConfig,
        get_mixvargenet_stride2channels,
    )

    net_config = [
        [MixVarGENetConfig(32, 32, "mixvarge_f2", [], 1, 1)],
        [MixVarGENetConfig(32, 64, "mixvarge_f4", ["mixvarge_f4"], 1, 2)],
    ]
    s2c = get_mixvargenet_stride2channels(net_config)
    assert isinstance(s2c, dict)
    assert 2 in s2c


def test_mixvargenet_forward_feature_pyramid():
    pytest.importorskip("torch")
    import torch

    from heal.models.backbones.mixvargenet import (
        MixVarGENet,
        MixVarGENetConfig,
    )

    net_config = [
        [MixVarGENetConfig(32, 32, "mixvarge_f2", [], 1, 1)],
        [MixVarGENetConfig(32, 32, "mixvarge_f4", ["mixvarge_f4"], 1, 2)],
    ]
    model = MixVarGENet(
        net_config=net_config, num_classes=10, bn_kwargs={}, include_top=False
    ).eval()
    x = torch.randn(1, 3, 64, 64)
    outs = model(x)
    assert isinstance(outs, list)
    assert len(outs) > 0
