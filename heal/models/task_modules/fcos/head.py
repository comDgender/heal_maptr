# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import Sequence

import torch
import torch.nn as nn
from hat_engine.models.base_modules.separable_conv_module import (
    SeparableConvModule2d,
)
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.apply_func import _as_list
from hat_engine.utils.saved_tensor import support_saved_tensor
from torch.quantization import DeQuantStub

from heal.models.weight_init import bias_init_with_prob, normal_init

__all__ = ["FCOSHead"]


@OBJECT_REGISTRY.register
class FCOSHead(nn.Module):
    """FCOS detection head for multi-level feature maps.

    The head uses anchor-free prediction with three branches per level:
    classification logits, bbox distance regression, and centerness.
    Depthwise/pointwise conv weights are shared across FPN levels while
    BatchNorm is kept independent per level.
    """

    def __init__(
        self,
        num_classes: int,
        in_strides: Sequence[int],
        out_strides: Sequence[int],
        stride2channels: dict,
        upscale_bbox_pred: bool,
        feat_channels: int = 256,
        stacked_convs: int = 4,
        use_sigmoid: bool = True,
        dequant_output: bool = True,
        bbox_relu: bool = True,
        use_save_tensor: bool = True,
        pred_reg_channel: int = 4,
    ):
        """Initialize FCOS head.

        Args:
            num_classes: Number of detection classes.
            in_strides: Strides of input backbone/FPN features.
            out_strides: Strides selected for FCOS prediction outputs.
            stride2channels: Mapping from stride to input channel count.
            upscale_bbox_pred: Multiply bbox output by stride during eval.
            feat_channels: Hidden channel width of head conv towers.
            stacked_convs: Number of stacked conv blocks per branch.
            use_sigmoid: Classification uses sigmoid (no explicit bg class).
            dequant_output: Whether to dequantize output tensors.
            bbox_relu: Whether to apply ReLU on bbox regression output.
            use_save_tensor: Enable saved-tensor friendly feature routing.
            pred_reg_channel: Output channels of bbox regression branch.
        """
        super(FCOSHead, self).__init__()
        self.in_strides = sorted(_as_list(in_strides))
        self.out_strides = sorted(_as_list(out_strides))
        assert set(self.out_strides).issubset(self.in_strides)
        self.feat_indices = [
            self.in_strides.index(stride)
            for stride in self.out_strides
            if stride in self.in_strides
        ]
        self.in_channels = stride2channels[self.in_strides[0]]
        self.pred_reg_channel = pred_reg_channel
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.use_sigmoid = use_sigmoid
        self.upscale_bbox_pred = upscale_bbox_pred
        self.dequant_output = dequant_output
        self.bbox_relu = bbox_relu
        self.use_save_tensor = use_save_tensor
        self.cls_out_channels = num_classes if use_sigmoid else num_classes + 1
        self._init_layers()
        self._init_weights()

    def _init_layers(self):
        self.dequant = DeQuantStub()
        self._init_cls_reg_convs_with_independent_bn()
        self._init_predictor()

    def _init_cls_reg_convs_with_independent_bn(self):
        num_strides = len(self.out_strides)
        self.cls_convs = nn.ModuleList(
            [nn.ModuleList() for _ in range(num_strides)]
        )
        self.reg_convs = nn.ModuleList(
            [nn.ModuleList() for _ in range(num_strides)]
        )
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            for j in range(num_strides):
                cls_shared_dw_conv = None
                cls_shared_pw_conv = None
                reg_shared_dw_conv = None
                reg_shared_pw_conv = None
                if j > 0:
                    cls_shared_dw_conv = self.cls_convs[0][i][0][0]
                    cls_shared_pw_conv = self.cls_convs[0][i][1][0]
                    reg_shared_dw_conv = self.reg_convs[0][i][0][0]
                    reg_shared_pw_conv = self.reg_convs[0][i][1][0]
                if cls_shared_dw_conv is None:
                    self.cls_convs[j].append(
                        SeparableConvModule2d(
                            chn,
                            self.feat_channels,
                            kernel_size=3,
                            padding=1,
                            pw_norm_layer=nn.BatchNorm2d(self.feat_channels),
                            pw_act_layer=nn.ReLU(inplace=True),
                        )
                    )
                else:
                    self.cls_convs[j].append(
                        nn.Sequential(
                            cls_shared_dw_conv,
                            cls_shared_pw_conv,
                            nn.BatchNorm2d(self.feat_channels),
                            nn.ReLU(inplace=True),
                        )
                    )
                if reg_shared_dw_conv is None:
                    self.reg_convs[j].append(
                        SeparableConvModule2d(
                            chn,
                            self.feat_channels,
                            kernel_size=3,
                            padding=1,
                            pw_norm_layer=nn.BatchNorm2d(self.feat_channels),
                            pw_act_layer=nn.ReLU(inplace=True),
                        )
                    )
                else:
                    self.reg_convs[j].append(
                        nn.Sequential(
                            reg_shared_dw_conv,
                            reg_shared_pw_conv,
                            nn.BatchNorm2d(self.feat_channels),
                            nn.ReLU(inplace=True),
                        )
                    )

    def _init_predictor(self):
        self.conv_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1
        )
        self.conv_reg = nn.Conv2d(
            self.feat_channels, self.pred_reg_channel, 3, padding=1
        )
        self.single_relu = nn.ReLU(inplace=True)
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

    def _init_weights(self):
        for m in self.cls_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.reg_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.conv_centerness.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.conv_cls, std=0.01, bias=bias_cls)
        normal_init(self.conv_reg, std=0.01)
        normal_init(self.conv_centerness, std=0.01)

    def forward_single(self, x, i, stride):
        """Run FCOS head on one feature level.

        Args:
            x: Feature map tensor of one FPN level.
            i: Level index used to select per-level BN branches.
            stride: Spatial stride of current feature level.

        Returns:
            Tuple of `(cls_score, bbox_pred, centerness)` tensors.
        """
        cls_feat = x
        reg_feat = x
        for cls_layer in self.cls_convs[i]:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.conv_cls(cls_feat)
        for reg_layer in self.reg_convs[i]:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)
        if self.bbox_relu:
            bbox_pred = self.single_relu(bbox_pred)
        centerness = self.conv_centerness(reg_feat)
        if self.dequant_output:
            cls_score = self.dequant(cls_score)
            bbox_pred = self.dequant(bbox_pred)
            centerness = self.dequant(centerness)
        if self.upscale_bbox_pred and not self.training:
            bbox_pred = bbox_pred * stride
        return cls_score, bbox_pred, centerness

    def forward(self, feats):
        """Run FCOS head on selected multi-level features.

        Args:
            feats: Input feature maps from backbone/FPN.

        Returns:
            Three lists: class scores, bbox predictions, and centerness logits.
        """
        if self.use_save_tensor and support_saved_tensor():
            feats = _as_list(feats)
            ignore_feats = 0
            use_feats = []
            for i in range(len(feats)):
                if i in self.feat_indices:
                    use_feats.append(feats[i])
                else:
                    ignore_feats = ignore_feats + torch.sum(feats[i])
            feats = use_feats
            ignore_feats = ignore_feats * 0
            feats[0] = feats[0] + ignore_feats
        else:
            feats = [_as_list(feats)[index] for index in self.feat_indices]
        cls_scores, bbox_preds, centernesses = [], [], []
        for i, feat in enumerate(feats):
            cls_score, bbox_pred, centerness = self.forward_single(
                feat, i, self.out_strides[i]
            )
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
            centernesses.append(centerness)
        return cls_scores, bbox_preds, centernesses
