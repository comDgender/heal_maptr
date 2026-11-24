"""Tests for heal.models.base_modules.basic_mixvargenet_module."""

import pytest


def test_basic_mixvarge_block_forward_shape():
    pytest.importorskip("torch")
    import torch

    from heal.models.base_modules.basic_mixvargenet_module import (
        BasicMixVarGEBlock,
    )

    block = BasicMixVarGEBlock(
        in_channels=32,
        out_channels=32,
        stride=1,
        bias=True,
        bn_kwargs={},
        conv1_kernel_size=3,
        conv2_kernel_size=1,
        padding=0,
        factor=1,
        conv1_group_base=1,
        conv2_group_base=1,
    ).eval()
    x = torch.randn(1, 32, 16, 16)
    y = block(x)
    assert y.shape[0] == 1
    assert y.shape[1] == 32


def test_mixvarge_block_forward_shape():
    pytest.importorskip("torch")
    import torch

    from heal.models.base_modules.basic_mixvargenet_module import MixVarGEBlock

    block = MixVarGEBlock(
        in_ch=32,
        block_ch=32,
        head_op="mixvarge_f4",
        stack_ops=["mixvarge_f4"],
        stride=1,
        bias=True,
        bn_kwargs={},
        output_downsample=False,
    ).eval()
    x = torch.randn(1, 32, 16, 16)
    y = block(x)
    # MixVarGEBlock returns (output, downsample_list)
    # when output_downsample is True,
    # or just output when output_downsample is False
    out = y if not isinstance(y, tuple) else y[0]
    assert out.shape[0] == 1
    assert out.shape[1] == 32
