# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
"""Reusable BiFPN building blocks."""

import copy
from types import SimpleNamespace
from typing import Dict, List, Union

import torch
import torch.nn as nn
from hat_engine.models.base_modules.conv_module import ConvModule2d
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.apply_func import _as_list
from hat_engine.utils.logger import rank_zero_info
from torch.nn import ModuleDict, ModuleList, Sequential

try:
    import horizon_plugin_pytorch.nn as nnf
except (ImportError, RuntimeError):
    nnf = None
try:
    from torch.nn.quantized import FloatFunctional
except ImportError:
    FloatFunctional = None

__all__ = [
    "BiFPN",
    "BifpnLayer",
    "Fusion",
    "MaybeApply1x1",
    "Resize",
    "get_fpn_config",
    "normal_init",
]


def _check_strides(strides, valid_strides):
    """Validate and normalize stride list.

    Args:
        strides: Input stride sequence.
        valid_strides: Allowed stride values.

    Returns:
        list: Normalized stride list.
    """
    s = (
        list(strides)
        if not isinstance(strides, (list, tuple))
        else list(strides)
    )
    for x in s:
        assert x in valid_strides
    return s


def normal_init(module, mean=0, std=0.01, bias=0):
    """Initialize conv weights with normal distribution.

    Args:
        module: Module containing ``weight`` and optional ``bias`` tensors.
        mean: Mean value of the normal distribution.
        std: Standard deviation of the normal distribution.
        bias: Constant bias value.
    """
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class MaybeApply1x1(nn.Module):
    """Conditionally apply 1x1 channel projection.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        use_bn: Whether to append BatchNorm2d after projection.
    """

    def __init__(self, in_channels, out_channels, use_bn=False):
        super(MaybeApply1x1, self).__init__()
        if in_channels == out_channels:
            return
        self.lateral_conv = ModuleList()
        self.lateral_conv.append(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0)
        )
        if use_bn:
            self.lateral_conv.append(nn.BatchNorm2d(out_channels))

    def forward(self, x):
        """Run optional projection.

        Args:
            x: Input feature map.

        Returns:
            torch.Tensor: Projected feature map when channel counts differ;
            otherwise the original feature map.
        """
        if hasattr(self, "lateral_conv"):
            for m in self.lateral_conv:
                x = m(x)
        return x


class Resize(nn.Module):
    """Feature resize block for up/down/keep sampling.

    Args:
        sampling: Resize mode, one of ``down``, ``up``, or ``keep``.
        in_channels: Input channel count.
        out_channels: Output channel count.
        pooling_type: Pooling method used for downsampling.
        use_bn: Whether 1x1 projections use BatchNorm2d.
        conv_after_downsample: Whether to project channels after downsampling.
        upsample_type: Upsampling backend.
    """

    def __init__(
        self,
        sampling: str,
        in_channels: int,
        out_channels: int,
        pooling_type: str = "max",
        use_bn: bool = True,
        conv_after_downsample: bool = False,
        upsample_type: str = "module",
    ):
        super(Resize, self).__init__()
        assert sampling in ["down", "up", "keep"]
        assert pooling_type in ["max", "avg"]
        assert upsample_type in ["module", "function", "torch_c"]
        self.sampling = sampling
        self.resize_layer = ModuleList()
        if sampling == "down":
            if not conv_after_downsample:
                lc = MaybeApply1x1(in_channels, out_channels, use_bn)
                if in_channels != out_channels:
                    self.resize_layer.append(lc)
            self.resize_layer.append(
                nn.MaxPool2d(2, 2, 0)
                if pooling_type == "max"
                else nn.AvgPool2d(2, 2, 0)
            )
            if conv_after_downsample:
                lc = MaybeApply1x1(in_channels, out_channels, use_bn)
                if in_channels != out_channels:
                    self.resize_layer.append(lc)
        else:
            lc = MaybeApply1x1(in_channels, out_channels, use_bn)
            if in_channels != out_channels:
                self.resize_layer.append(lc)
        self.upsample_type = upsample_type
        self.upsampling = None
        if sampling == "up" and upsample_type == "module" and nnf is not None:
            self.upsampling = nnf.Interpolate(
                scale_factor=2, mode="bilinear", recompute_scale_factor=True
            )

    def forward(self, x):
        """Resize feature map.

        Args:
            x: Input feature map.

        Returns:
            torch.Tensor: Resized feature map.
        """
        for m in self.resize_layer:
            x = m(x)
        if self.sampling == "up":
            if self.upsampling is not None:
                x = self.upsampling(x)
            elif self.upsample_type == "torch_c":
                x = torch._C._nn.upsample_bilinear2d(
                    x,
                    output_size=None,
                    align_corners=False,
                    scale_factors=(2, 2),
                )
            else:
                x = nn.functional.interpolate(
                    x,
                    scale_factor=2,
                    mode="bilinear",
                    recompute_scale_factor=True,
                )
        return x


class Fusion(nn.Module):
    """Fuse multi-branch features by sum or fast attention weights.

    Args:
        in_channels: Channel count of fused features.
        weight_len: Number of input branches.
        weight_method: Fusion method, ``sum`` or ``fastattn``.
        eps: Small value used to normalize attention weights.
        use_fx: Whether to use FX-friendly tensor additions.
    """

    def __init__(
        self,
        in_channels: int,
        weight_len: int,
        weight_method: str = "sum",
        eps: float = 0.0001,
        use_fx: bool = False,
    ):
        super(Fusion, self).__init__()
        self.weight_method = weight_method
        self.eps = eps
        self.use_fx = use_fx
        self.floatF = (
            None
            if use_fx
            else (FloatFunctional() if FloatFunctional else None)
        )
        if weight_method == "fastattn":
            self.edge_weights = nn.Parameter(
                torch.ones(weight_len, dtype=torch.float32), requires_grad=True
            )
            self.relu = nn.ReLU(inplace=True)

    def float_function_sum(self, x):
        """Accumulate inputs with FloatFunctional when available.

        Args:
            x: Input tensors to accumulate.

        Returns:
            torch.Tensor: Sum of input tensors.
        """
        x = _as_list(x)
        for i, val in enumerate(x):
            if i == 0:
                res = val
            else:
                if self.use_fx or self.floatF is None:
                    res = res + val
                else:
                    res = self.floatF.add(res, val)
        return res

    def forward(self, x):
        """Fuse multi-input tensors.

        Args:
            x: Sequence of feature tensors to fuse.

        Returns:
            torch.Tensor: Fused feature tensor.
        """
        x = list(x)
        if self.weight_method == "sum":
            return self.float_function_sum(x)
        relu_edge_weights = self.relu(self.edge_weights)
        weights_sum = self.float_function_sum(relu_edge_weights)
        for i in range(len(x)):
            x[i] = x[i] * relu_edge_weights[i]
            x[i] = x[i] / (weights_sum + self.eps)
        return self.float_function_sum(x)


class BifpnLayer(nn.Module):
    """Single BiFPN cell containing all top-down and bottom-up nodes.

    Args:
        fpn_config: BiFPN topology config.
        out_index: Optional output indexes selected from the final cell.
        use_fx: Whether to use FX-friendly operations in fusion.
    """

    def __init__(
        self, fpn_config, out_index: List[int] = None, use_fx: bool = False
    ):
        super(BifpnLayer, self).__init__()
        self.fpn_config = fpn_config
        self.out_index = out_index
        level = fpn_config.level
        in_channels = fpn_config.in_channels
        out_channels = fpn_config.out_channels
        if isinstance(out_channels, int):
            out_channels = [out_channels] * 5
        offset2inchannels = {
            0: out_channels[0],
            1: out_channels[1],
            2: out_channels[2],
            3: out_channels[3],
            4: out_channels[4],
            5: out_channels[3],
            6: out_channels[2],
            7: out_channels[1],
            8: out_channels[0],
            9: out_channels[1],
            10: out_channels[2],
            11: out_channels[3],
        }
        offset2out_channels = {
            0: {"keep": out_channels[0]},
            1: {"keep": out_channels[1]},
            2: {"keep": out_channels[2]},
            3: {"keep": out_channels[3]},
            4: {"keep": out_channels[4], "up": out_channels[3]},
            5: {"keep": out_channels[3], "up": out_channels[2]},
            6: {"keep": out_channels[2], "up": out_channels[1]},
            7: {"keep": out_channels[1], "up": out_channels[0]},
            8: {"down": out_channels[1]},
            9: {"down": out_channels[2]},
            10: {"down": out_channels[3]},
            11: {"down": out_channels[4]},
        }
        weight_method = fpn_config.weight_method
        self.all_nodes = ModuleDict()
        for i, fnode in enumerate(fpn_config.nodes):
            rank_zero_info("fnode %d : %s" % (i, fnode))
            node = ModuleList()
            out_ch = 0
            for offset, sampling, upsample_type in zip(
                fnode["inputs_offsets"],
                fnode["sampling"],
                fnode["upsample_type"],
            ):
                in_ch = (
                    in_channels[offset]
                    if offset < level
                    else offset2inchannels[offset]
                )
                out_ch = offset2out_channels[offset][sampling]
                node.append(
                    Resize(
                        sampling,
                        in_ch,
                        out_ch,
                        upsample_type=upsample_type,
                    )
                )
            node.append(
                Fusion(
                    out_ch,
                    len(fnode["inputs_offsets"]),
                    weight_method,
                    use_fx=use_fx,
                )
            )
            node.append(
                Sequential(
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        out_ch,
                        out_ch,
                        3,
                        stride=1,
                        bias=False,
                        groups=out_ch,
                        padding=1,
                    ),
                    ConvModule2d(
                        out_ch,
                        out_ch,
                        1,
                        stride=1,
                        groups=1,
                        padding=0,
                        norm_layer=nn.BatchNorm2d(out_ch),
                    ),
                )
            )
            self.all_nodes[str(i)] = node

    def forward(self, x):
        """Run one BiFPN layer.

        Args:
            x: Input feature maps for this BiFPN cell.

        Returns:
            list[torch.Tensor]: Output feature maps from this cell.
        """
        x = list(x)
        assert len(x) == self.fpn_config.level
        for i, fnode in enumerate(self.fpn_config.nodes):
            nodes = []
            for idx, input_offset in enumerate(fnode["inputs_offsets"]):
                input_node = self.all_nodes[str(i)][idx](x[input_offset])
                nodes.append(input_node)
            new_node = self.all_nodes[str(i)][idx + 1](nodes)
            new_node = self.all_nodes[str(i)][idx + 2](new_node)
            x.append(new_node)
        all_outs = [x[j] for j in range(-self.fpn_config.level, 0, 1)]
        if self.out_index is not None:
            return [all_outs[j] for j in self.out_index]
        return all_outs


def get_fpn_config(
    fpn_name="bifpn_sum", out_channels=64, upsample_type="module"
):
    """Build predefined BiFPN topology config.

    Args:
        fpn_name: BiFPN topology name.
        out_channels: Output channels per level or shared int.
        upsample_type: Upsampling backend used in resize edges.

    Returns:
        SimpleNamespace: BiFPN topology config.
    """
    assert fpn_name in ["bifpn_sum", "bifpn_fa"]
    if isinstance(out_channels, int):
        out_channels = [out_channels] * 5
    c = SimpleNamespace()
    c.out_channels = out_channels
    c.level = 5
    c.nodes = [
        {
            "inputs_offsets": [3, 4],
            "sampling": ["keep", "up"],
            "upsample_type": [upsample_type, upsample_type],
        },
        {
            "inputs_offsets": [2, 5],
            "sampling": ["keep", "up"],
            "upsample_type": [upsample_type, upsample_type],
        },
        {
            "inputs_offsets": [1, 6],
            "sampling": ["keep", "up"],
            "upsample_type": [upsample_type, upsample_type],
        },
        {
            "inputs_offsets": [0, 7],
            "sampling": ["keep", "up"],
            "upsample_type": [upsample_type, upsample_type],
        },
        {
            "inputs_offsets": [1, 7, 8],
            "sampling": ["keep", "keep", "down"],
            "upsample_type": [upsample_type] * 3,
        },
        {
            "inputs_offsets": [2, 6, 9],
            "sampling": ["keep", "keep", "down"],
            "upsample_type": [upsample_type] * 3,
        },
        {
            "inputs_offsets": [3, 5, 10],
            "sampling": ["keep", "keep", "down"],
            "upsample_type": [upsample_type] * 3,
        },
        {
            "inputs_offsets": [4, 11],
            "sampling": ["keep", "down"],
            "upsample_type": [upsample_type, upsample_type],
        },
    ]
    c.weight_method = "fastattn" if fpn_name == "bifpn_fa" else "sum"
    return c


@OBJECT_REGISTRY.register
class BiFPN(nn.Module):
    """Stacked BiFPN neck for multi-scale feature fusion.

    Args:
        in_strides: Input feature strides.
        out_strides: Required output feature strides.
        stride2channels: Mapping from stride to input channels.
        out_channels: Output channels per stride or shared int.
        num_outs: Number of output feature levels.
        stack: Number of stacked BiFPN layers.
        start_level: Start level index from input features.
        end_level: End level index (exclusive), `-1` means last level.
        fpn_name: BiFPN config name.
        upsample_type: Upsample backend in resize edges.
        use_fx: Whether to use FX-friendly operations in fusion.
    """

    def __init__(
        self,
        in_strides: List[int],
        out_strides: List[int],
        stride2channels: Dict,
        out_channels: Union[int, Dict],
        num_outs: int,
        stack: int = 3,
        start_level: int = 0,
        end_level: int = -1,
        fpn_name: str = "bifpn_sum",
        upsample_type: str = "module",
        use_fx: bool = False,
    ):
        super(BiFPN, self).__init__()
        self.in_strides = _check_strides(in_strides, stride2channels.keys())
        self.out_strides = out_strides
        self.stride2channels = stride2channels
        self.in_channels = [stride2channels[s] for s in self.in_strides]
        self.out_channels = (
            [out_channels[s] for s in out_strides]
            if isinstance(out_channels, dict)
            else [out_channels for _ in out_strides]
        )
        self.num_ins = len(in_strides)
        self.num_outs = num_outs
        assert len(out_strides) <= num_outs and stack >= 1
        self.stack = stack
        self.fpn_config = get_fpn_config(
            fpn_name, self.out_channels, upsample_type
        )

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert self.num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(self.in_channels)
            assert self.num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        extra_levels = (
            self.num_outs - self.backbone_end_level + self.start_level
        )
        self.fpn_config.in_channels = self.in_channels[
            self.start_level : self.backbone_end_level
        ] + [
            self.out_channels[self.backbone_end_level - self.start_level + i]
            for i in range(extra_levels)
        ]
        self.total_out_strides = self.in_strides[
            self.start_level : self.backbone_end_level
        ] + [
            self.in_strides[self.backbone_end_level - 1] * (2 ** (i + 1))
            for i in range(extra_levels)
        ]
        self.out_index = [
            self.total_out_strides.index(s) for s in self.out_strides
        ]
        self.extra_downsamples = ModuleList()
        for i in range(extra_levels):
            oc = self.out_channels[
                self.backbone_end_level - self.start_level + i
            ]
            downsample = Resize(
                sampling="down",
                in_channels=self.in_channels[-1],
                out_channels=oc,
                pooling_type="max",
                use_bn=True,
                conv_after_downsample=False,
                upsample_type=upsample_type,
            )
            self.in_channels[-1] = oc
            self.extra_downsamples.append(downsample)
        self.bifpn_layers = ModuleList()
        for i in range(self.stack):
            rank_zero_info("building bifpn cell %d" % i)
            out_index = self.out_index if i == self.stack - 1 else None
            self.bifpn_layers.append(
                BifpnLayer(
                    copy.copy(self.fpn_config),
                    out_index,
                    use_fx=use_fx,
                )
            )
            if i == 0:
                self.fpn_config.in_channels = [
                    self.out_channels[j] for j in range(self.num_outs)
                ]
        self._init_weights()

    def _init_weights(self):
        """Initialize convolution layers in BiFPN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, mean=0, std=0.01, bias=0)

    def forward(self, inputs):
        """Run BiFPN forward.

        Args:
            inputs: Input multi-scale feature maps.

        Returns:
            list[torch.Tensor]: Fused multi-scale output feature maps.
        """
        assert len(inputs) == len(self.in_channels)
        x = list(inputs[self.start_level : self.backbone_end_level])
        for d in self.extra_downsamples:
            x.append(d(x[-1]))
        for i in range(self.stack):
            x = self.bifpn_layers[i](x)
        return x
