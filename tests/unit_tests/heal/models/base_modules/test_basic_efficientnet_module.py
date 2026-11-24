"""Tests for heal.models.base_modules.basic_efficientnet_module."""

import pytest


def test_mbconv_block_forward_shape():
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn

    from heal.models.backbones.efficientnet import BlockArgs
    from heal.models.base_modules.basic_efficientnet_module import MBConvBlock

    args = BlockArgs(
        kernel_size=3,
        num_repeat=1,
        in_filters=8,
        out_filters=8,
        expand_ratio=1,
        id_skip=True,
        strides=1,
        se_ratio=None,
    )
    block = MBConvBlock(
        args,
        bn_kwargs={"momentum": 0.01, "eps": 1e-3},
        act_layer=nn.ReLU(inplace=True),
        use_se_block=False,
    ).eval()
    x = torch.randn(1, 8, 16, 16)
    y = block(x)
    assert y.shape == x.shape
