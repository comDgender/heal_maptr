# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
from typing import Optional

import torch

try:
    from horizon_plugin_pytorch.nn import ChannelScale2d
except ImportError:
    ChannelScale2d = None
from hat_engine.models.base_modules.conv_module import ConvModule2d
from horizon_plugin_pytorch.nn.quantized import FloatFunctional as FF
from horizon_plugin_pytorch.quantization import QuantStub
from torch import nn

__all__ = [
    "S2DDown",
    "DWCB",
    "GroupDWCB",
    "AltDWCB",
    "AttentionBlock",
    "LinearAttention",
    "BasicHENetStageBlock",
]


class S2DDown(nn.Module):
    """Perform strided projection for spatial downsampling.

    Args:
        in_dim: Number of input channels.
        out_dim: Number of output channels.
        patch_size: Stride and kernel size for downsampling.
        padding: Optional convolution padding.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        patch_size: int,
        padding: Optional[int] = 0,
    ):
        super().__init__()

        # Equivalent operation that merges downsampling and pointwise conv.
        self.proj = ConvModule2d(
            in_dim,
            out_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=padding,
            norm_layer=nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        """Project input feature maps to a lower spatial resolution.

        Args:
            x: Input feature map.

        Returns:
            torch.Tensor: Downsampled feature map.
        """
        x = self.proj(x)
        return x

    def fuse_model(self):
        """Fuse internal modules for quantization-aware deployment."""
        self.proj.fuse_model()


class DWCB(nn.Module):
    """Apply a depthwise convolution block with pointwise projection.

    Args:
        dim: Number of input and output channels.
        hidden_dim: Hidden channels used by pointwise projection.
        kernel_size: Kernel size of the depthwise convolution.
        act_layer: Activation layer type.
        use_layer_scale: Whether to use learnable layer scaling.
        extra_act: Whether to apply an extra activation after residual add.
        block_idx: Block index, reserved for interface compatibility.
    """

    def __init__(  # noqa: D205
        self,
        dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        act_layer: str = "nn.GELU",
        use_layer_scale: bool = True,
        extra_act: Optional[bool] = False,
        block_idx: Optional[int] = 0,
    ):
        super().__init__()

        self.extra_act = (
            eval(act_layer)() if extra_act is True else nn.Identity()
        )

        self.dwconv = ConvModule2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            norm_layer=nn.BatchNorm2d(dim),
        )
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = eval(act_layer)()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            assert (
                ChannelScale2d is not None
            ), "require horizon_plugin_pytorch>=2.4.4"
            self.layer_scale = ChannelScale2d(dim)

        self.add = FF()

    def forward(self, x):
        """Run the residual depthwise block.

        Args:
            x: Input feature map.

        Returns:
            torch.Tensor: Output feature map.
        """
        input = x
        x = self.dwconv(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = self.add.add(input, self.layer_scale(x))
        else:
            x = self.add.add(input, x)
        x = self.extra_act(x)
        return x

    def fuse_model(self):
        """Fuse internal modules for quantization-aware deployment."""
        self.dwconv.fuse_model()


class GroupDWCB(nn.Module):
    """Apply a depthwise block with grouped pointwise convolution.

    Args:
        dim: Number of input and output channels.
        hidden_dim: Hidden channels used by pointwise projection.
        kernel_size: Kernel size of the depthwise convolution.
        act_layer: Activation layer type.
        use_layer_scale: Whether to use learnable layer scaling.
        extra_act: Whether to apply an extra activation after residual add.
        block_idx: Block index, reserved for interface compatibility.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        act_layer: str = "nn.GELU",
        use_layer_scale: bool = True,
        extra_act: Optional[bool] = False,
        block_idx: Optional[int] = 0,
    ):
        super().__init__()

        self.extra_act = (
            eval(act_layer)() if extra_act is True else nn.Identity()
        )

        # Other settings are recommended to be validated first.
        group_width = {
            64: 64,
            128: 64,
            192: 64,
            384: 64,
            256: 128,
            48: 48,
            96: 48,
        }[dim]

        self.dwconv = ConvModule2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=dim,
            norm_layer=nn.BatchNorm2d(dim),
        )
        self.pwconv1 = nn.Conv2d(
            dim, hidden_dim, kernel_size=1, groups=dim // group_width
        )
        self.act = eval(act_layer)()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            assert (
                ChannelScale2d is not None
            ), "require horizon_plugin_pytorch>=2.4.4"
            self.layer_scale = ChannelScale2d(dim)

        self.add = FF()
        self.add_dw = FF()

    def forward(self, x):
        """Run the grouped residual depthwise block.

        Args:
            x: Input feature map.

        Returns:
            torch.Tensor: Output feature map.
        """
        input = x
        x = self.dwconv(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = self.add.add(input, self.layer_scale(x))
        else:
            x = self.add.add(input, x)
        x = self.extra_act(x)
        return x

    def fuse_model(self):
        """Fuse internal modules for quantization-aware deployment."""
        self.dwconv.fuse_model()


class AltDWCB(nn.Module):
    """Apply a depthwise block with alternating asymmetric kernels.

    Args:
        dim: Number of input and output channels.
        hidden_dim: Hidden channels used by pointwise projection.
        kernel_size: Unused interface-compatible kernel size argument.
        act_layer: Activation layer type.
        use_layer_scale: Whether to use learnable layer scaling.
        extra_act: Whether to apply an extra activation after residual add.
        block_idx: Block index used to alternate kernel shapes.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        act_layer: str = "nn.GELU",
        use_layer_scale: bool = True,
        extra_act: Optional[bool] = False,
        block_idx: Optional[int] = 0,
    ):
        super().__init__()

        self.extra_act = (
            eval(act_layer)() if extra_act is True else nn.Identity()
        )

        # Alternate between (1, n) and (n, 1) convolutions.
        if block_idx % 2 == 0:
            dw_k, dw_p = (1, 5), (0, 2)
        else:
            dw_k, dw_p = (5, 1), (2, 0)

        self.dwconv = nn.Sequential(
            ConvModule2d(
                dim,
                dim,
                kernel_size=dw_k,
                padding=dw_p,
                groups=dim,
                norm_layer=nn.BatchNorm2d(dim),
            ),
        )

        self.pwconv1 = ConvModule2d(
            dim, hidden_dim, kernel_size=1, act_layer=eval(act_layer)()
        )
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, kernel_size=1)
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            assert (
                ChannelScale2d is not None
            ), "require horizon_plugin_pytorch>=2.4.4"
            self.layer_scale = ChannelScale2d(dim)

        self.add = FF()

    def forward(self, x):
        """Run the alternating residual depthwise block.

        Args:
            x: Input feature map.

        Returns:
            torch.Tensor: Output feature map.
        """
        input = x
        x = self.dwconv(x)
        x = self.pwconv1(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = self.add.add(input, self.layer_scale(x))
        else:
            x = self.add.add(input, x)
        x = self.extra_act(x)
        return x

    def fuse_model(self):
        """Fuse internal modules for quantization-aware deployment."""
        if isinstance(self.dwconv, ConvModule2d):
            self.dwconv.fuse_model()
        else:
            for module in self.dwconv:
                module.fuse_model()
        self.pwconv1.fuse_model()


class ConvMlp(nn.Module):
    """Apply the convolutional MLP used in HENet blocks.

    Args:
        dim: Number of input and output channels.
        hidden_dim: Number of hidden channels.
        act_layer: Activation layer type.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        act_layer: str = "nn.GELU",
    ):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.act = eval(act_layer)()
        self.fc2 = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        """Transform features with normalization and pointwise convolutions.

        Args:
            x: Input feature map.

        Returns:
            torch.Tensor: Output feature map.
        """
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class LinearAttention(nn.Module):
    """Apply linear attention inside HENet attention blocks.

    Args:
        dim: Input feature dimension.
        dim_head: Dimension of each attention head.
        num_heads: Number of attention heads.
    """

    def __init__(
        self,
        dim: int,
        dim_head: int,
        num_heads: int,
    ):
        super().__init__()
        self.query = nn.Linear(dim, dim_head * num_heads)
        self.key = nn.Linear(dim, dim_head * num_heads)
        self.query_weight = nn.Parameter(torch.randn(dim_head * num_heads, 1))
        self.scale = dim_head**-0.5
        self.projection = nn.Linear(dim_head * num_heads, dim_head * num_heads)
        self.output = nn.Linear(dim_head * num_heads, dim_head)
        self.matmul = FF()
        self.mul_scalar = FF()
        self.sum = FF()
        self.add = FF()
        self.mul1 = FF()
        self.mul2 = FF()
        self.quant = QuantStub()

    def forward(self, x):
        """Compute linear attention output.

        Args:
            x: Input sequence tensor.

        Returns:
            torch.Tensor: Attention output.
        """
        query = self.query(x)
        key = self.key(x)
        query_weight = self.matmul.matmul(query, self.quant(self.query_weight))
        query_weight = self.mul_scalar.mul_scalar(query_weight, self.scale)
        attn_weight = self.sum.sum(
            self.mul1.mul(query_weight, query), dim=1, keepdim=True
        )
        attn_weight = attn_weight.repeat(1, key.shape[1], 1)

        out = self.add.add(
            self.projection(self.mul2.mul(attn_weight, key)), query
        )

        out = self.output(out)

        return out


class AttentionBlock(nn.Module):
    """Apply convolution, attention, and MLP branches in one block.

    Args:
        dim: Number of input and output channels.
        mlp_ratio: Hidden expansion ratio for the MLP branch.
        act_layer: Activation layer type.
        use_layer_scale: Whether to use learnable layer scaling.
        layer_scale_init_value: Initial value of layer-scale parameters.
        extra_act: Whether to apply an extra activation after convolution.
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 2.0,
        act_layer: str = "nn.GELU",
        use_layer_scale: bool = True,
        layer_scale_init_value: Optional[float] = 1e-5,
        extra_act: Optional[bool] = False,
    ):
        super().__init__()

        self.conv = DWCB(
            dim=dim,
            hidden_dim=dim,
            kernel_size=3,
            act_layer=act_layer,
            use_layer_scale=True,
            extra_act=extra_act,
        )
        self.attn = LinearAttention(dim=dim, dim_head=dim, num_heads=1)
        self.mlp = ConvMlp(
            dim=dim, hidden_dim=int(dim * mlp_ratio), act_layer=act_layer
        )
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_attn = nn.Parameter(
                layer_scale_init_value
                * torch.ones(dim).unsqueeze(-1).unsqueeze(-1),
                requires_grad=True,
            )
            self.layer_scale_mlp = nn.Parameter(
                layer_scale_init_value
                * torch.ones(dim).unsqueeze(-1).unsqueeze(-1),
                requires_grad=True,
            )
            self.quant_attn = QuantStub()
            self.quant_mlp = QuantStub()
            self.mul_attn = FF()
            self.mul_mlp = FF()
        self.add_attn = FF()
        self.add_mlp = FF()

    def forward(self, x):
        """Run convolution, attention, and MLP branches.

        Args:
            x: Input feature map.

        Returns:
            torch.Tensor: Output feature map.
        """
        x = self.conv(x)
        B, C, H, W = x.shape
        if self.use_layer_scale and isinstance(self.attn, LinearAttention):
            x = self.add_attn.add(
                x,
                self.mul_attn.mul(
                    self.quant_attn(self.layer_scale_attn),
                    self.attn(x.permute(0, 2, 3, 1).reshape(B, H * W, C))
                    .reshape(B, H, W, C)
                    .permute(0, 3, 1, 2),
                ),
            )
            x = self.add_mlp.add(
                x,
                self.mul_mlp.mul(
                    self.quant_mlp(self.layer_scale_mlp), self.mlp(x)
                ),
            )
        elif self.use_layer_scale and isinstance(self.attn, DWCB):
            x = self.add_attn.add(
                x,
                self.mul_attn.mul(
                    self.quant_attn(self.layer_scale_attn), self.attn(x)
                ),
            )
            x = self.add_mlp.add(
                x,
                self.mul_mlp.mul(
                    self.quant_mlp(self.layer_scale_mlp), self.mlp(x)
                ),
            )
            x = self.add_mlp.add(
                x,
                self.mul_mlp.mul(
                    self.quant_mlp(self.layer_scale_mlp), self.mlp(x)
                ),
            )
        else:
            x = self.add_attn.add(
                x,
                self.attn(x.permute(0, 2, 3, 1).reshape(B, H * W, C))
                .reshape(B, H, W, C)
                .permute(0, 3, 1, 2),
            )
            x = self.add_mlp.add(x, self.mlp(x))
        return x

    def fuse_model(self):
        """Fuse internal modules for quantization-aware deployment."""
        self.conv.fuse_model()


class BasicHENetStageBlock(nn.Module):
    """Stack HENet blocks to build one stage.

    Args:
        in_dim: Number of input channels.
        block_num: Total number of blocks in the stage.
        attention_block_num: Number of attention blocks in the stage.
        mlp_ratio: Hidden expansion ratio used by DWCB-like blocks.
        mlp_ratio_attn: Hidden expansion ratio used by attention blocks.
        act_layer: Activation layer type.
        use_layer_scale: Whether to use learnable layer scaling.
        layer_scale_init_value: Initial value of layer-scale parameters.
        extra_act: Whether to apply an extra activation after each block.
        block_cls: Block class name used for non-attention blocks.
    """

    def __init__(
        self,
        in_dim: int,
        block_num: int,
        attention_block_num: int,
        mlp_ratio: float = 2.0,
        mlp_ratio_attn: float = 2.0,
        act_layer: str = "nn.GELU",
        use_layer_scale: bool = True,
        layer_scale_init_value: Optional[float] = 1e-5,
        extra_act: Optional[bool] = False,
        block_cls: str = "DWCB",
    ):
        super().__init__()
        block = []

        for block_idx in range(block_num - attention_block_num):
            block.append(
                eval(block_cls)(
                    dim=in_dim,
                    hidden_dim=int(mlp_ratio * in_dim),
                    kernel_size=3,
                    act_layer=act_layer,
                    use_layer_scale=use_layer_scale,
                    extra_act=extra_act,
                    block_idx=block_idx,
                )
            )
        for _ in range(attention_block_num):
            block.append(
                AttentionBlock(
                    dim=in_dim,
                    mlp_ratio=mlp_ratio_attn,
                    act_layer=act_layer,
                    use_layer_scale=True,
                    layer_scale_init_value=layer_scale_init_value,
                    extra_act=extra_act,
                )
            )

        self.block = nn.Sequential(*block)

    def forward(self, x):
        """Run the whole stage.

        Args:
            x: Input feature map.

        Returns:
            torch.Tensor: Output feature map.
        """
        return self.block(x)

    def fuse_model(self):
        """Fuse child blocks for quantization-aware deployment."""
        for m in self.block:
            m.fuse_model()
