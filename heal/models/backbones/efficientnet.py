# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import collections
import copy
import functools
import math
from typing import Dict, Sequence

import torch.nn as nn
from hat_engine.models.base_modules.conv_module import ConvModule2d
from hat_engine.registry import OBJECT_REGISTRY, build_from_registry

from heal.models.base_modules.basic_efficientnet_module import MBConvBlock

try:
    from horizon_plugin_pytorch.quantization import QuantStub
    from torch.quantization import DeQuantStub
except (ImportError, RuntimeError):
    QuantStub = None
    DeQuantStub = None

__all__ = ["EfficientNet", "efficientnet", "efficientnet_lite"]

BlockArgs = collections.namedtuple(
    "BlockArgs",
    [
        "kernel_size",
        "num_repeat",
        "in_filters",
        "out_filters",
        "expand_ratio",
        "id_skip",
        "strides",
        "se_ratio",
    ],
)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

DEFAULT_BLOCKS_ARGS = (
    BlockArgs(3, 1, 32, 16, 1, True, 1, 0.25),
    BlockArgs(3, 2, 16, 24, 6, True, 2, 0.25),
    BlockArgs(5, 2, 24, 40, 6, True, 2, 0.25),
    BlockArgs(3, 3, 40, 80, 6, True, 2, 0.25),
    BlockArgs(5, 3, 80, 112, 6, True, 1, 0.25),
    BlockArgs(5, 4, 112, 192, 6, True, 2, 0.25),
    BlockArgs(3, 1, 192, 320, 6, True, 1, 0.25),
)


def round_filters(filters, width_coefficient, depth_division):
    """Scale channel count by width coefficient and divisor.

    Args:
        filters: Base channel count before scaling.
        width_coefficient: Width multiplier for the model variant.
        depth_division: Channel divisor used to keep hardware-friendly widths.

    Returns:
        int: Rounded channel count after scaling.
    """
    filters *= width_coefficient
    new_filters = max(
        int((filters + depth_division / 2) // depth_division * depth_division),
        depth_division,
    )
    if new_filters < 0.9 * filters:
        new_filters += depth_division
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    """Scale block repeat count by depth coefficient.

    Args:
        repeats: Base repeat count before scaling.
        depth_coefficient: Depth multiplier for the model variant.

    Returns:
        int: Rounded repeat count after scaling.
    """
    return int(math.ceil(depth_coefficient * repeats))


def _get_act_layer(activation: str):
    """Build activation layer by name.

    Args:
        activation: Activation name, one of ``relu``, ``relu6``, or
            ``swish``.

    Returns:
        nn.Module: Instantiated activation layer.
    """
    try:
        return build_from_registry({"type": activation, "inplace": True})
    except Exception:
        if activation == "relu":
            return nn.ReLU(inplace=True)
        if activation == "relu6":
            return nn.ReLU6(inplace=True)
        if activation == "swish":
            try:
                from torch.nn import SiLU

                return SiLU(inplace=True)
            except ImportError:
                pass
        raise ValueError(
            f"activation must be relu/relu6/swish, got {activation}"
        )


@OBJECT_REGISTRY.register
class EfficientNet(nn.Module):
    """EfficientNet backbone for classification or feature extraction.

    Args:
        model_type: EfficientNet variant name.
        coefficient_params: Tuple of width, depth, resolution, and dropout.
        num_classes: Number of classifier output classes.
        bn_kwargs: Keyword arguments passed to BatchNorm2d.
        bias: Whether convolution layers use bias.
        drop_connect_rate: Drop-connect rate used across MBConv blocks.
        depth_division: Channel divisor used when rounding filter counts.
        activation: Activation type used by convolution blocks.
        use_se_block: Whether to enable squeeze-and-excitation blocks.
        blocks_args: MBConv block definitions.
        include_top: Whether to include the classification head.
        flat_output: Whether to flatten classifier output.
        input_channels: Number of input channels.
        resolution: Input resolution override kept for config compatibility.
        split_expand_conv: Whether to split large expand convolutions.
        quant_input: Whether to apply the input quantization stub.
    """

    def __init__(
        self,
        model_type: str,
        coefficient_params: tuple,
        num_classes: int,
        bn_kwargs: dict = None,
        bias: bool = False,
        drop_connect_rate: float = None,
        depth_division: int = 8,
        activation: str = "relu",
        use_se_block: bool = False,
        blocks_args: Sequence[Dict] = DEFAULT_BLOCKS_ARGS,
        include_top: bool = True,
        flat_output: bool = True,
        input_channels: int = 3,
        resolution: int = 0,
        split_expand_conv: bool = False,
        quant_input: bool = True,
    ):
        super().__init__()
        assert activation in ["relu", "relu6", "swish"]
        self.model_type = model_type
        self.use_lite = "lite" in self.model_type
        if self.use_lite:
            assert use_se_block is False and activation != "swish"

        (
            self.width_coefficient,
            self.depth_coefficient,
            self.default_resolution,
            self.dropout_rate,
        ) = coefficient_params
        self.drop_connect_rate = drop_connect_rate
        self.depth_division = depth_division
        self.num_classes = num_classes
        self.include_top = include_top
        self.flat_output = flat_output
        self.split_expand_conv = split_expand_conv
        self.quant_input = quant_input
        self.blocks_args = [
            BlockArgs(**b) if isinstance(b, dict) else b for b in blocks_args
        ]
        self.activation = activation
        act_layer = _get_act_layer(activation)

        self.bn_kwargs = (
            bn_kwargs
            if bn_kwargs is not None
            else {
                "momentum": 0.01,
                "eps": 1e-3,
            }
        )
        self.use_se_block = use_se_block

        if self.use_lite:
            out_planes = 32
        else:
            out_planes = round_filters(
                32, self.width_coefficient, self.depth_division
            )
        self.mod1 = nn.Sequential(
            ConvModule2d(
                in_channels=input_channels,
                out_channels=out_planes,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=bias,
                norm_layer=nn.BatchNorm2d(out_planes, **self.bn_kwargs),
                act_layer=copy.deepcopy(act_layer),
            )
        )
        self.mod2 = self._make_stage(act_layer)

        in_planes = round_filters(
            self.blocks_args[-1].out_filters,
            self.width_coefficient,
            self.depth_division,
        )
        if self.use_lite:
            out_planes = 1280
        else:
            out_planes = round_filters(
                1280, self.width_coefficient, self.depth_division
            )

        if self.include_top:
            self.mod3 = nn.Sequential(
                ConvModule2d(
                    in_channels=in_planes,
                    out_channels=out_planes,
                    kernel_size=1,
                    bias=bias,
                    norm_layer=nn.BatchNorm2d(out_planes, **self.bn_kwargs),
                    act_layer=copy.deepcopy(act_layer),
                )
            )
            self.output = nn.Sequential(
                nn.AvgPool2d(self.default_resolution // (2**5)),
                nn.Dropout2d(self.dropout_rate),
                ConvModule2d(
                    in_channels=out_planes,
                    out_channels=num_classes,
                    kernel_size=1,
                    bias=bias,
                    norm_layer=nn.BatchNorm2d(num_classes, **self.bn_kwargs),
                ),
            )
        else:
            self.mod3 = None
            self.output = None

        if QuantStub is not None and DeQuantStub is not None:
            self.quant = QuantStub(scale=1.0 / 128.0)
            self.dequant = DeQuantStub()
        else:
            self.quant = nn.Identity()
            self.dequant = nn.Identity()

    def _make_stage(self, act_layer):
        """Build MBConv stages from block definitions.

        Args:
            act_layer: Activation layer used in MBConv blocks.

        Returns:
            nn.ModuleList: Constructed MBConv block list.
        """
        self._blocks_conv = nn.ModuleList([])
        self.stage_divider = []
        for idx, block_args in enumerate(self.blocks_args):
            if self.use_lite and idx == 0:
                in_filters = block_args.in_filters
            else:
                in_filters = round_filters(
                    block_args.in_filters,
                    self.width_coefficient,
                    self.depth_division,
                )
            out_filters = round_filters(
                block_args.out_filters,
                self.width_coefficient,
                self.depth_division,
            )
            if self.use_lite and (
                idx == 0 or idx == len(self.blocks_args) - 1
            ):
                num_repeat = block_args.num_repeat
            else:
                num_repeat = round_repeats(
                    block_args.num_repeat, self.depth_coefficient
                )
            block_args = block_args._replace(
                in_filters=in_filters,
                out_filters=out_filters,
                num_repeat=num_repeat,
            )
            self._blocks_conv.append(
                MBConvBlock(
                    block_args,
                    bn_kwargs=self.bn_kwargs,
                    act_layer=copy.deepcopy(act_layer),
                    use_se_block=self.use_se_block,
                )
            )
            if block_args.strides == 2:
                self.stage_divider.append(len(self._blocks_conv) - 1)
            if block_args.num_repeat > 1:
                block_args = block_args._replace(
                    in_filters=block_args.out_filters,
                    strides=1,
                )
            for _ in range(block_args.num_repeat - 1):
                self._blocks_conv.append(
                    MBConvBlock(
                        block_args,
                        bn_kwargs=self.bn_kwargs,
                        act_layer=copy.deepcopy(act_layer),
                        use_se_block=self.use_se_block,
                        split_expand_conv=self.split_expand_conv,
                    )
                )
        self.stage_divider.append(len(self._blocks_conv))
        return self._blocks_conv

    def forward(self, inputs):
        """Run the forward pass.

        Args:
            inputs: Input image tensor.

        Returns:
            torch.Tensor | list[torch.Tensor]: Classification logits when
            ``include_top`` is ``True``; otherwise stage feature maps.
        """
        output = []
        x = inputs
        if self.quant_input:
            x = self.quant(x)
        x = self.mod1(x)
        for idx, block in enumerate(self._blocks_conv):
            if self.drop_connect_rate:
                drop_connect_rate = self.drop_connect_rate * (
                    float(idx) / len(self._blocks_conv)
                )
                x = block(x, drop_connect_rate=drop_connect_rate)
            else:
                x = block(x)
            if idx + 1 in self.stage_divider:
                output.append(x)
        if not self.include_top:
            return output
        x = self.mod3(x)
        x = self.output(x)
        x = self.dequant(x)
        if self.flat_output:
            x = x.view(-1, self.num_classes)
        return x


def _efficient(mode_type, coefficient_params, **kwargs):
    """Instantiate an EfficientNet variant with fixed coefficients.

    Args:
        mode_type: EfficientNet variant name.
        coefficient_params: Tuple of width, depth, resolution, and dropout.
        **kwargs: Additional EfficientNet constructor arguments.

    Returns:
        EfficientNet: Built EfficientNet model.
    """
    return functools.partial(EfficientNet, mode_type, coefficient_params)(
        **kwargs
    )


@OBJECT_REGISTRY.register
def efficientnet(model_type, **kwargs):
    """Build an EfficientNet model by variant name.

    Args:
        model_type: Variant name, one of ``b0``-``b7``, ``b0tiny``, or
            ``b0small``.
        **kwargs: Additional EfficientNet constructor arguments.

    Returns:
        EfficientNet: Built EfficientNet model.
    """
    coefficient_params_dict = {
        "b0tiny": (0.5, 0.5, 224, 0.2),
        "b0small": (0.5, 1.0, 224, 0.2),
        "b0": (1.0, 1.0, 224, 0.2),
        "b1": (1.0, 1.1, 240, 0.2),
        "b2": (1.1, 1.2, 260, 0.3),
        "b3": (1.2, 1.4, 300, 0.3),
        "b4": (1.4, 1.8, 380, 0.4),
        "b5": (1.6, 2.2, 456, 0.4),
        "b6": (1.8, 2.6, 528, 0.5),
        "b7": (2.0, 3.1, 600, 0.5),
    }
    assert model_type in coefficient_params_dict
    coefficient_params = kwargs.pop(
        "coefficient_params",
        coefficient_params_dict[model_type],
    )
    return _efficient(model_type, coefficient_params, **kwargs)


@OBJECT_REGISTRY.register
def efficientnet_lite(model_type, **kwargs):
    """Build an EfficientNet-Lite model by variant name.

    Args:
        model_type: Variant name, one of ``lite0``-``lite4``.
        **kwargs: Additional EfficientNet constructor arguments.

    Returns:
        EfficientNet: Built EfficientNet-Lite model.
    """
    coefficient_params_dict = {
        "lite0": (1.0, 1.0, 224, 0.2),
        "lite1": (1.0, 1.1, 240, 0.2),
        "lite2": (1.1, 1.2, 260, 0.3),
        "lite3": (1.2, 1.4, 280, 0.3),
        "lite4": (1.4, 1.8, 300, 0.3),
    }
    assert model_type in coefficient_params_dict
    coefficient_params = kwargs.pop(
        "coefficient_params",
        coefficient_params_dict[model_type],
    )
    return _efficient(model_type, coefficient_params, **kwargs)
