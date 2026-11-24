# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
"""Basic building blocks for EfficientNet backbones (MBConv and SE)."""

import copy
import math

import torch
import torch.nn as nn
from hat_engine.models.base_modules.conv_module import ConvModule2d
from hat_engine.utils.model_helpers import fx_wrap

if fx_wrap is not None:
    _fx_decorator = fx_wrap()
else:

    def _fx_decorator(f):
        return f


try:
    from horizon_plugin_pytorch.qtensor import QTensor
    from torch.nn.quantized import FloatFunctional
except (ImportError, RuntimeError):
    QTensor = None
    FloatFunctional = None

__all__ = ["SEBlock", "MBConvBlock"]


def _get_float_functional():
    """Return FloatFunctional for quantized add/mul.

    Falls back to ``nn.Identity`` when unavailable.
    """
    if FloatFunctional is not None:
        return FloatFunctional()
    return nn.Identity()


class SEBlock(nn.Module):
    """Apply squeeze-and-excitation channel recalibration.

    Args:
        in_channels: Number of input channels.
        num_squeezed_channels: Channels in the squeeze 1x1 conv.
        out_channels: Number of output channels for excitation.
        act_layer: Activation used in the squeeze branch.
        adapt_pooling: Whether to use adaptive average pooling.
    """

    def __init__(
        self,
        in_channels: int,
        num_squeezed_channels: int,
        out_channels: int,
        act_layer: torch.nn.Module,
        adapt_pooling: bool = False,
    ):
        super(SEBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.AvgPool2d(1) if not adapt_pooling else nn.AdaptiveAvgPool2d(1),
            ConvModule2d(
                in_channels=in_channels,
                out_channels=num_squeezed_channels,
                kernel_size=1,
                bias=True,
                norm_layer=None,
                act_layer=act_layer,
            ),
            ConvModule2d(
                in_channels=num_squeezed_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=True,
                norm_layer=None,
                act_layer=nn.Sigmoid(),
            ),
        )
        self.float_func = _get_float_functional()

    def forward(self, inputs):
        """Recalibrate channel responses and multiply back to the input.

        Args:
            inputs: Input feature map.

        Returns:
            torch.Tensor: Channel-recalibrated feature map.
        """
        x = self.conv(inputs)
        if hasattr(self.float_func, "mul"):
            x = self.float_func.mul(x, inputs)
        else:
            x = x * inputs
        return x


class MBConvBlock(nn.Module):
    """Apply an inverted residual MBConv block used in EfficientNet.

    Args:
        block_args: BlockArgs with kernel, stride, channels, and SE ratio.
        bn_kwargs: Keyword arguments passed to BatchNorm2d.
        act_layer: Activation layer used inside conv modules.
        use_se_block: Whether to insert a squeeze-and-excitation block.
        split_expand_conv: Whether to split expand 1x1 conv for large widths.
    """

    def __init__(
        self,
        block_args,
        bn_kwargs: dict,
        act_layer: torch.nn.Module,
        use_se_block: bool = False,
        split_expand_conv: bool = False,
    ):
        super(MBConvBlock, self).__init__()
        self._block_args = block_args
        self.expand_ratio = self._block_args.expand_ratio
        self.in_planes = self._block_args.in_filters
        self.out_planes = self.in_planes * self.expand_ratio
        self.final_out_planes = self._block_args.out_filters
        self.kernel_size = self._block_args.kernel_size
        self.stride = self._block_args.strides
        self.id_skip = self._block_args.id_skip
        self.split_expand_conv = split_expand_conv
        if split_expand_conv:
            self.out_planes1 = self.out_planes // 2
            self.out_planes2 = self.out_planes - self.out_planes1

        self.has_se = (
            use_se_block
            and (self._block_args.se_ratio is not None)
            and (0 < self._block_args.se_ratio <= 1)
        )

        if self.expand_ratio != 1:
            if not self.split_expand_conv or self.out_planes <= 2048:
                self._expand_conv = ConvModule2d(
                    in_channels=self.in_planes,
                    out_channels=self.out_planes,
                    kernel_size=1,
                    bias=False,
                    norm_layer=nn.BatchNorm2d(self.out_planes, **bn_kwargs),
                    act_layer=copy.deepcopy(act_layer),
                )
            else:
                self._expand_conv1 = ConvModule2d(
                    in_channels=self.in_planes,
                    out_channels=self.out_planes1,
                    kernel_size=1,
                    bias=False,
                    norm_layer=nn.BatchNorm2d(self.out_planes1, **bn_kwargs),
                    act_layer=copy.deepcopy(act_layer),
                )
                self._expand_conv2 = ConvModule2d(
                    in_channels=self.in_planes,
                    out_channels=self.out_planes2,
                    kernel_size=1,
                    bias=False,
                    norm_layer=nn.BatchNorm2d(self.out_planes2, **bn_kwargs),
                    act_layer=copy.deepcopy(act_layer),
                )

        if not self.split_expand_conv or self.out_planes <= 2048:
            self._depthwise_conv = ConvModule2d(
                in_channels=self.out_planes,
                out_channels=self.out_planes,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=(
                    math.ceil((self.kernel_size - 1) // 2)
                    if self.stride == 1
                    else math.ceil(self.kernel_size // self.stride)
                ),
                bias=False,
                groups=self.out_planes,
                norm_layer=nn.BatchNorm2d(self.out_planes, **bn_kwargs),
                act_layer=copy.deepcopy(act_layer),
            )
            self._project_conv = ConvModule2d(
                in_channels=self.out_planes,
                out_channels=self.final_out_planes,
                kernel_size=1,
                bias=False,
                norm_layer=nn.BatchNorm2d(self.final_out_planes, **bn_kwargs),
                act_layer=None,
            )
        else:
            self._depthwise_conv1 = ConvModule2d(
                in_channels=self.out_planes1,
                out_channels=self.out_planes1,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=(
                    math.ceil((self.kernel_size - 1) // 2)
                    if self.stride == 1
                    else math.ceil(self.kernel_size // self.stride)
                ),
                bias=False,
                groups=self.out_planes1,
                norm_layer=nn.BatchNorm2d(self.out_planes1, **bn_kwargs),
                act_layer=copy.deepcopy(act_layer),
            )
            self._depthwise_conv2 = ConvModule2d(
                in_channels=self.out_planes2,
                out_channels=self.out_planes2,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=(
                    math.ceil((self.kernel_size - 1) // 2)
                    if self.stride == 1
                    else math.ceil(self.kernel_size // self.stride)
                ),
                bias=False,
                groups=self.out_planes2,
                norm_layer=nn.BatchNorm2d(self.out_planes2, **bn_kwargs),
                act_layer=copy.deepcopy(act_layer),
            )
            self._project_conv1 = ConvModule2d(
                in_channels=self.out_planes1,
                out_channels=self.final_out_planes,
                kernel_size=1,
                bias=False,
                norm_layer=nn.BatchNorm2d(self.final_out_planes, **bn_kwargs),
                act_layer=None,
            )
            self._project_conv2 = ConvModule2d(
                in_channels=self.out_planes2,
                out_channels=self.final_out_planes,
                kernel_size=1,
                bias=False,
                norm_layer=nn.BatchNorm2d(self.final_out_planes, **bn_kwargs),
                act_layer=None,
            )
            self.float_func_proj_conv = _get_float_functional()

        if self.has_se:
            num_squeezed_num = max(
                1, int(self.in_planes * self._block_args.se_ratio)
            )
            self._se_block = SEBlock(
                in_channels=self.out_planes,
                num_squeezed_channels=num_squeezed_num,
                out_channels=self.out_planes,
                act_layer=copy.deepcopy(act_layer),
            )

        self.use_shortcut = (
            self.id_skip
            and self.stride == 1
            and self.in_planes == self.final_out_planes
        )
        if self.use_shortcut:
            self.float_func = _get_float_functional()

    def forward(self, inputs, drop_connect_rate=None):
        """Run expand, depthwise, optional SE, project, and residual add.

        Args:
            inputs: Input feature map.
            drop_connect_rate: Stochastic depth rate during training.

        Returns:
            torch.Tensor: Output feature map of the MBConv block.
        """
        x = inputs
        if self.expand_ratio != 1:
            if not self.split_expand_conv or self.out_planes <= 2048:
                x = self._expand_conv(inputs)
            else:
                x1 = self._expand_conv1(inputs)
                x2 = self._expand_conv2(inputs)

        if not self.split_expand_conv or self.out_planes <= 2048:
            x = self._depthwise_conv(x)
        else:
            x1 = self._depthwise_conv1(x1)
            x2 = self._depthwise_conv2(x2)

        if self.has_se:
            x = self._se_block(x)

        if not self.split_expand_conv or self.out_planes <= 2048:
            x = self._project_conv(x)
            if self.use_shortcut:
                if drop_connect_rate:
                    x = self.drop_connect(x, drop_connect_rate)
                if hasattr(self.float_func, "add"):
                    x = self.float_func.add(x, inputs)
                else:
                    x = x + inputs
        else:
            x1 = self._project_conv1(x1)
            x2 = self._project_conv2(x2)
            if self.use_shortcut:
                if drop_connect_rate:
                    x1 = self.drop_connect(x1, drop_connect_rate)
                if hasattr(self.float_func, "add"):
                    x1 = self.float_func.add(x1, inputs)
                else:
                    x1 = x1 + inputs
                if hasattr(self.float_func_proj_conv, "add"):
                    x = self.float_func_proj_conv.add(x2, x1)
                else:
                    x = x2 + x1
        return x

    @_fx_decorator
    def drop_connect(self, x, drop_connect_rate):
        """Apply stochastic depth (drop connect) during training.

        Args:
            x: Feature map to regularize.
            drop_connect_rate: Probability of dropping the residual branch.

        Returns:
            torch.Tensor: Feature map after drop connect.
        """
        if self.training:
            keep_prob = 1.0 - drop_connect_rate
            batch_size = x.shape[0]
            random_tensor = keep_prob
            if QTensor is not None and isinstance(x, QTensor):
                x_dtype = x.as_subclass(torch.Tensor).dtype
            else:
                x_dtype = x.dtype
            random_tensor += torch.rand(
                [batch_size, 1, 1, 1], dtype=x_dtype, device=x.device
            )
            binary_mask = torch.floor(random_tensor)
            if QTensor is not None and isinstance(x, QTensor):
                y = (x.as_subclass(torch.Tensor) / keep_prob) * binary_mask
                x = QTensor(y, x.q_scale(), x.dtype, x.per_channel_axis)
            else:
                x = (x / keep_prob) * binary_mask
        return x
