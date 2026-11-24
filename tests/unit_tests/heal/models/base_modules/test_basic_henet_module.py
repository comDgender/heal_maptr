import pytest


def test_s2ddown_forward_shape():
    pytest.importorskip("torch")
    pytest.importorskip("horizon_plugin_pytorch")
    import torch

    from heal.models.base_modules.basic_henet_module import S2DDown

    module = S2DDown(in_dim=16, out_dim=32, patch_size=2)
    x = torch.randn(2, 16, 32, 32)
    y = module(x)

    assert y.shape == (2, 32, 16, 16)


def test_basic_henet_stage_block_forward_shape():
    pytest.importorskip("torch")
    pytest.importorskip("horizon_plugin_pytorch")
    import torch

    from heal.models.base_modules.basic_henet_module import (
        BasicHENetStageBlock,
    )

    module = BasicHENetStageBlock(
        in_dim=32,
        block_num=2,
        attention_block_num=0,
        mlp_ratio=2.0,
        act_layer="nn.ReLU",
        use_layer_scale=False,
        extra_act=False,
        block_cls="DWCB",
    )
    x = torch.randn(2, 32, 16, 16)
    y = module(x)

    assert y.shape == x.shape
