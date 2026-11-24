# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
import copy
from typing import Dict, List, Sequence

import torch.nn as nn
from hat_engine.models.base_modules.conv_module import ConvModule2d
from hat_engine.registry import OBJECT_REGISTRY
from torch.quantization import DeQuantStub

from heal.models.weight_init import normal_init

__all__ = ["CenterPointHead"]


class TaskHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        heads: Dict,
        head_conv_channels: int,
        final_kernel: int = 1,
        init_bias: float = -2.19,
        int8_output: bool = False,
        bn_kwargs=None,
    ):
        super().__init__()
        self.heads = heads
        self.init_bias = init_bias
        self.int8_output = int8_output
        if bn_kwargs is None:
            bn_kwargs = {}
        self.dequant = DeQuantStub()
        for head in self.heads:
            classes, num_conv = self.heads[head]
            conv_layers = []
            c_in = in_channels
            for _ in range(num_conv - 1):
                conv_layers.append(
                    self._make_conv(
                        in_channels=c_in,
                        out_channels=head_conv_channels,
                        kernel_size=final_kernel,
                        stride=1,
                    )
                )
                c_in = head_conv_channels
            conv_layers.append(
                ConvModule2d(
                    in_channels=head_conv_channels,
                    out_channels=classes,
                    kernel_size=final_kernel,
                    padding=final_kernel // 2,
                    stride=1,
                    bias=True if head == "heatmap" else False,
                )
            )
            self.__setattr__(head, nn.Sequential(*conv_layers))
        self._init_weights()

    def _init_weights(self):
        for head in self.heads:
            head_convs = self.__getattr__(head)
            for m in head_convs.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
            if head == "heatmap":
                head_convs[-1][0].bias.data.fill_(self.init_bias)

    def forward(self, x):
        ret_dict = {}
        for head in self.heads:
            ret_dict[head] = self.dequant(self.__getattr__(head)(x))
        return ret_dict

    def fuse_model(self) -> None:
        for head in self.heads:
            head_convs = self.__getattr__(head)
            for m in head_convs:
                if hasattr(m, "fuse_model"):
                    m.fuse_model()

    def set_qconfig(self) -> None:
        from hat_engine.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        if not self.int8_output:
            for head in self.heads:
                head_convs = self.__getattr__(head)
                head_convs[-1].qconfig = (
                    qconfig_manager.get_default_qat_out_qconfig()
                )

    def _make_conv(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        bias=True,
    ):
        return ConvModule2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=bias,
            norm_layer=nn.BatchNorm2d(out_channels),
            act_layer=nn.ReLU(inplace=True),
        )


@OBJECT_REGISTRY.register
class CenterPointHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        tasks: List[dict],
        share_conv_channels: int,
        share_conv_num: int,
        common_heads: Dict,
        num_heatmap_convs: int = 2,
        bn_kwargs=None,
        **kwargs,
    ):
        super().__init__()
        self.bn_kwargs = bn_kwargs if bn_kwargs else {}
        self.shared_conv = nn.Sequential(
            *(
                self._make_conv(
                    in_channels=in_channels if i == 0 else share_conv_channels,
                    out_channels=share_conv_channels,
                    kernel_size=3,
                    stride=1,
                )
                for i in range(share_conv_num)
            )
        )
        self.task_heads = nn.ModuleList()
        num_classes = [len(t["class_names"]) for t in tasks]
        for num_cls in num_classes:
            heads = copy.deepcopy(common_heads)
            heads.update({"heatmap": (num_cls, num_heatmap_convs)})
            self.task_heads.append(
                TaskHead(
                    in_channels=share_conv_channels,
                    heads=heads,
                    bn_kwargs=self.bn_kwargs,
                    **kwargs,
                )
            )

    def forward(self, feats):
        rets = []
        if isinstance(feats, Sequence):
            feats = feats[0]
        feats = self.shared_conv(feats)
        for task in self.task_heads:
            rets.append(task(feats))
        return rets

    def _make_conv(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        bias=True,
    ):
        return ConvModule2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=bias,
            norm_layer=nn.BatchNorm2d(out_channels),
            act_layer=nn.ReLU(inplace=True),
        )

    def fuse_model(self) -> None:
        for m in self.shared_conv:
            if hasattr(m, "fuse_model"):
                m.fuse_model()
        for head in self.task_heads:
            if hasattr(head, "fuse_model"):
                head.fuse_model()

    def set_qconfig(self) -> None:
        from hat_engine.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        for head in self.task_heads:
            if hasattr(head, "set_qconfig"):
                head.set_qconfig()
