# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
from typing import Sequence, Tuple

import horizon_plugin_pytorch.nn as hnn
import torch
import torch.nn as nn
from hat_engine.models.base_modules.conv_module import ConvModule2d
from hat_engine.registry import OBJECT_REGISTRY
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from heal.models.base_modules.basic_henet_module import (
    BasicHENetStageBlock,
    S2DDown,
)


@OBJECT_REGISTRY.register
class HENet(nn.Module):
    """The HENet backbone for classification or feature extraction.

    Args:
        in_channels: Number of input channels.
        block_nums: Number of blocks in each stage.
        embed_dims: Output channels in each stage.
        attention_block_num: Number of attention blocks in each stage.
        mlp_ratios: MLP expansion ratios in each stage.
        mlp_ratio_attn: MLP expansion ratio in attention blocks.
        act_layer: Activation layer type for each stage.
        use_layer_scale: Whether to use a learnable residual scale.
        layer_scale_init_value: Initial value of the learnable scale factor.
        num_classes: Number of classifier output classes.
        include_top: Whether to include output layer.
        flat_output: Whether to flatten the classifier output.
        extra_act: Whether to use extra activation layers in each stage.
        final_expand_channel: Channel expansion before pooling.
        feature_mix_channel: Channel expansion performed before the head.
        block_cls: Basic block type used in each stage.
        down_cls: Downsample block type used in each stage.
        patch_embed: Stem convolution style used at the input.
        quant_input: Whether to add a quantization stub at the input.
        dequant_output: Whether to add a dequantization stub at the output.
        stage_out_norm: Whether to add normalization on stage outputs.
            Ignored if include_top is True.
    """

    def __init__(
        self,
        in_channels: int,
        block_nums: Tuple[int],
        embed_dims: Tuple[int],
        attention_block_num: Tuple[int],
        mlp_ratios: Tuple[int] = (2, 2, 2, 2),
        mlp_ratio_attn: int = 2,
        act_layer: Tuple[str] = ("nn.GELU", "nn.GELU", "nn.GELU", "nn.GELU"),
        use_layer_scale: Tuple[bool] = (True, True, True, True),
        layer_scale_init_value: float = 1e-5,
        num_classes: int = 1000,
        include_top: bool = True,
        flat_output: bool = True,
        extra_act: Tuple[bool] = (False, False, False, False),
        final_expand_channel: int = 0,
        feature_mix_channel: int = 0,
        block_cls: Tuple[str] = ("DWCB", "DWCB", "DWCB", "DWCB"),
        down_cls: Tuple[str] = ("S2DDown", "S2DDown", "S2DDown", "None"),
        patch_embed: str = "origin",
        quant_input: bool = True,
        dequant_output: bool = True,
        stage_out_norm: bool = True,
    ):
        super().__init__()

        self.final_expand_channel = final_expand_channel
        self.feature_mix_channel = feature_mix_channel
        self.stage_out_norm = stage_out_norm

        self.block_cls = block_cls

        self.include_top = include_top
        self.flat_output = flat_output

        if self.include_top:
            self.num_classes = num_classes

        if patch_embed in ["origin"]:
            self.patch_embed = nn.Sequential(
                ConvModule2d(
                    in_channels,
                    embed_dims[0] // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    norm_layer=nn.BatchNorm2d(embed_dims[0] // 2),
                    act_layer=nn.ReLU(),
                ),
                ConvModule2d(
                    embed_dims[0] // 2,
                    embed_dims[0],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    norm_layer=nn.BatchNorm2d(embed_dims[0]),
                    act_layer=nn.ReLU(),
                ),
            )
        elif patch_embed in ["conv_k5_s4"]:
            self.patch_embed = nn.Sequential(
                ConvModule2d(
                    in_channels,
                    embed_dims[0],
                    kernel_size=5,
                    stride=4,
                    padding=2,
                    norm_layer=nn.BatchNorm2d(embed_dims[0]),
                    act_layer=nn.GELU(),
                ),
            )
        elif patch_embed in ["conv_4_s4"]:
            self.patch_embed = nn.Sequential(
                ConvModule2d(
                    in_channels,
                    embed_dims[0],
                    kernel_size=4,
                    stride=4,
                    padding=0,
                    norm_layer=nn.BatchNorm2d(embed_dims[0]),
                    act_layer=nn.GELU(),
                ),
            )

        stages = []
        downsample_block = []
        for block_idx, block_num in enumerate(block_nums):
            stages.append(
                BasicHENetStageBlock(
                    in_dim=embed_dims[block_idx],
                    block_num=block_num,
                    attention_block_num=attention_block_num[block_idx],
                    mlp_ratio=mlp_ratios[block_idx],
                    mlp_ratio_attn=mlp_ratio_attn,
                    act_layer=act_layer[block_idx],
                    use_layer_scale=use_layer_scale[block_idx],
                    layer_scale_init_value=layer_scale_init_value,
                    extra_act=extra_act[block_idx],
                    block_cls=block_cls[block_idx],
                )
            )
            if block_idx < len(block_nums) - 1:
                assert eval(down_cls[block_idx]) in [S2DDown], down_cls[
                    block_idx
                ]
                downsample_block.append(
                    eval(down_cls[block_idx])(
                        patch_size=2,
                        in_dim=embed_dims[block_idx],
                        out_dim=embed_dims[block_idx + 1],
                    )
                )
        self.stages = nn.ModuleList(stages)
        self.downsample_block = nn.ModuleList(downsample_block)

        if final_expand_channel in [0, None]:
            self.final_expand_layer = nn.Identity()
            self.norm = nn.BatchNorm2d(embed_dims[-1])
            last_channels = embed_dims[-1]
        else:
            self.final_expand_layer = ConvModule2d(
                embed_dims[-1],
                final_expand_channel,
                kernel_size=1,
                bias=False,
                norm_layer=nn.BatchNorm2d(final_expand_channel),
                act_layer=eval(act_layer[-1])(),
            )
            last_channels = final_expand_channel

        if feature_mix_channel in [0, None]:
            self.feature_mix_layer = nn.Identity()
        else:
            self.feature_mix_layer = ConvModule2d(
                last_channels,
                feature_mix_channel,
                kernel_size=1,
                bias=False,
                norm_layer=nn.BatchNorm2d(feature_mix_channel),
                act_layer=eval(act_layer[-1])(),
            )
            last_channels = feature_mix_channel

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.head = (
                nn.Linear(last_channels, num_classes)
                if num_classes > 0
                else nn.Identity()
            )
        else:
            stage_norm = []
            for embed_dim in embed_dims:
                if self.stage_out_norm is True:
                    stage_norm.append(nn.BatchNorm2d(embed_dim))
                else:
                    stage_norm.append(nn.Identity())
            self.stage_norm = nn.ModuleList(stage_norm)

        self.up = hnn.Interpolate(
            scale_factor=2, mode="bilinear", recompute_scale_factor=True
        )
        self.quant = QuantStub() if quant_input else None
        self.dequant = DeQuantStub() if dequant_output else None

    def forward(self, x):
        """Run the forward pass.

        Args:
            x: Input tensor or a single-element sequence containing the tensor.

        Returns:
            torch.Tensor | list[torch.Tensor]: Classification logits when
            ``include_top`` is ``True``; otherwise stage feature maps.
        """
        if self.quant is not None:
            x = self.quant(x)
        if isinstance(x, Sequence) and len(x) == 1:
            x = x[0]

        x = self.patch_embed(x)
        outs = []
        for idx in range(len(self.stages)):
            x = self.stages[idx](x)
            if not self.include_top:
                x_normed = self.stage_norm[idx](x)
                if idx == 0:
                    outs.append(self.up(x_normed))
                outs.append(x_normed)
            if idx < len(self.stages) - 1:
                x = self.downsample_block[idx](x)

        if not self.include_top:
            return outs

        if self.final_expand_channel in [0, None]:
            x = self.norm(x)
        else:
            x = self.final_expand_layer(x)
        x = self.avgpool(x)
        x = self.feature_mix_layer(x)
        x = self.head(torch.flatten(x, 1))

        if self.dequant is not None:
            x = self.dequant(x)
        if self.flat_output:
            x = x.view(-1, self.num_classes)
        return x
