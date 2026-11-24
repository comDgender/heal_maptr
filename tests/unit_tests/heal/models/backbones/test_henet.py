import pytest


def _build_tiny_henet(include_top):
    from heal.models.backbones.henet import HENet

    return HENet(
        in_channels=3,
        block_nums=(1, 1, 1, 1),
        embed_dims=(16, 32, 64, 128),
        attention_block_num=(0, 0, 0, 0),
        mlp_ratios=(2, 2, 2, 2),
        act_layer=("nn.ReLU", "nn.ReLU", "nn.ReLU", "nn.ReLU"),
        use_layer_scale=(False, False, False, False),
        include_top=include_top,
        num_classes=10,
        stage_out_norm=False,
    )


def test_henet_forward_classification_path_runs():
    pytest.importorskip("torch")
    pytest.importorskip("horizon_plugin_pytorch")
    import torch

    model = _build_tiny_henet(include_top=True).eval()
    x = torch.randn(2, 3, 64, 64)
    y = model(x)

    assert y.shape == (2, 10)


def test_henet_forward_feature_path_runs():
    pytest.importorskip("torch")
    pytest.importorskip("horizon_plugin_pytorch")
    import torch

    model = _build_tiny_henet(include_top=False).eval()
    x = torch.randn(1, 3, 64, 64)
    outs = model(x)

    assert isinstance(outs, list)
    assert len(outs) == 5
