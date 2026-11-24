# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import copy
from typing import Tuple

import horizon_plugin_pytorch as horizon
import torch
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.apply_func import _as_list, multi_apply
from hat_engine.utils.model_helpers import fx_wrap
from hat_engine.utils.qconfig_manager import (
    get_default_out_qconfig,
    get_default_qat_out_qconfig,
)
from horizon_plugin_pytorch.nn.quantized import FloatFunctional
from torch import nn
from torch.quantization import DeQuantStub

from heal.models.weight_init import bias_init_with_prob, normal_init


@OBJECT_REGISTRY.register
class FCOS3DHead(nn.Module):
    """Anchor-free head used in FCOS3D.

    Args:
        num_classes: Number of categories excluding the background category.
        in_channels: Number of channels in the input feature map.
        feat_channels: Number of hidden channels. Defaults to 256.
        stacked_convs: Number of stacking convs of the head.
        strides: Downsample factor of each feature map.
        group_reg_dims: The dimension of each regression target group.
        use_direction_classifier: Whether to add a direction classifier.
        pred_attrs: Whether to predict attributes.
        num_attrs: The number of attributes to be predicted.
        cls_branch: Channels for classification branch.
        reg_branch: Channels for regression branch.
        dir_branch: Channels for direction classification branch.
        attr_branch: Channels for attribute classification branch.
        centerness_branch: Channels for centerness branch.
        centerness_on_reg: If true, position centerness on the regress branch.
        return_for_compiler: whether return the output using for compile.
        output_int32: whether the output is int32.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        feat_channels: int,
        stacked_convs: int,
        strides: Tuple[int],
        group_reg_dims: Tuple[int],
        use_direction_classifier: bool,
        pred_attrs: bool,
        num_attrs: int,
        cls_branch: Tuple[int],
        reg_branch: Tuple[int],
        dir_branch: Tuple[int],
        attr_branch: Tuple[int],
        centerness_branch: Tuple[int],
        centerness_on_reg: Tuple[int],
        return_for_compiler=False,
        output_int32=True,
    ):
        super(FCOS3DHead, self).__init__()
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.group_reg_dims = group_reg_dims
        self.use_direction_classifier = use_direction_classifier
        self.pred_attrs = pred_attrs
        self.num_attrs = num_attrs
        self.cls_branch = cls_branch
        self.reg_branch = reg_branch
        self.dir_branch = dir_branch
        self.attr_branch = attr_branch
        self.centerness_branch = centerness_branch
        self.centerness_on_reg = centerness_on_reg
        self.return_for_compiler = return_for_compiler
        self.output_int32 = output_int32

        self.out_channels = []
        for reg_branch_channels in reg_branch:
            if len(reg_branch_channels) > 0:
                self.out_channels.append(reg_branch_channels[-1])
            else:
                self.out_channels.append(-1)
        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        self._init_cls_convs()
        self._init_reg_convs()
        self._init_predictor()
        self.conv_centerness_prev = self._init_branch(
            conv_channels=self.centerness_branch,
            conv_strides=(1,) * len(self.centerness_branch),
        )
        self.conv_centerness = nn.Conv2d(self.centerness_branch[-1], 1, 1)
        self.dequant = DeQuantStub()
        self.cat = FloatFunctional()

    def _init_cls_convs(self):
        self.cls_convs = nn.ModuleList()
        for i in range(len(self.strides)):
            cls_convs_per_stride = nn.ModuleList()
            for j in range(self.stacked_convs):
                if i == 0:
                    chn = self.in_channels if j == 0 else self.feat_channels
                    cls_dw_conv = nn.Conv2d(
                        in_channels=chn,
                        out_channels=chn,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=chn,
                    )
                    cls_pw_conv = nn.Conv2d(
                        in_channels=chn,
                        out_channels=self.feat_channels,
                        kernel_size=1,
                        stride=1,
                    )
                    cls_convs_per_stride.append(
                        nn.Sequential(
                            cls_dw_conv,
                            cls_pw_conv,
                            nn.BatchNorm2d(self.feat_channels),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    cls_convs_per_stride.append(
                        nn.Sequential(
                            copy.deepcopy(self.cls_convs[0][j][0]),
                            copy.deepcopy(self.cls_convs[0][j][1]),
                            nn.BatchNorm2d(self.feat_channels),
                            nn.ReLU(inplace=True),
                        )
                    )
            self.cls_convs.append(cls_convs_per_stride)

    def _init_reg_convs(self):
        self.reg_convs = nn.ModuleList()
        for i in range(len(self.strides)):
            reg_convs_per_stride = nn.ModuleList()
            for j in range(self.stacked_convs):
                if i == 0:
                    chn = self.in_channels if j == 0 else self.feat_channels
                    reg_dw_conv = nn.Conv2d(
                        in_channels=chn,
                        out_channels=chn,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=chn,
                    )
                    reg_pw_conv = nn.Conv2d(
                        in_channels=chn,
                        out_channels=self.feat_channels,
                        kernel_size=1,
                        stride=1,
                    )
                    reg_convs_per_stride.append(
                        nn.Sequential(
                            reg_dw_conv,
                            reg_pw_conv,
                            nn.BatchNorm2d(self.feat_channels),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    reg_convs_per_stride.append(
                        nn.Sequential(
                            copy.deepcopy(self.reg_convs[0][j][0]),
                            copy.deepcopy(self.reg_convs[0][j][1]),
                            nn.BatchNorm2d(self.feat_channels),
                            nn.ReLU(inplace=True),
                        )
                    )

            self.reg_convs.append(reg_convs_per_stride)

    def _init_branch(self, conv_channels=(64), conv_strides=(1)):
        conv_before_pred = nn.ModuleList()
        if isinstance(conv_channels, int):
            conv_channels = [self.feat_channels] + [conv_channels]
            conv_strides = [conv_strides]
        else:
            conv_channels = [self.feat_channels] + list(conv_channels)
            conv_strides = list(conv_strides)
        for i in range(len(self.strides)):
            conv_before_pred_stride = nn.ModuleList()
            for j in range(len(conv_strides)):
                if i == 0:
                    dw_conv = nn.Conv2d(
                        in_channels=conv_channels[j],
                        out_channels=conv_channels[j],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=conv_channels[j],
                    )
                    pw_conv = nn.Conv2d(
                        in_channels=conv_channels[j],
                        out_channels=conv_channels[j + 1],
                        kernel_size=1,
                        stride=1,
                    )

                    conv_before_pred_stride.append(
                        nn.Sequential(
                            dw_conv,
                            pw_conv,
                            nn.BatchNorm2d(conv_channels[j + 1]),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    conv_before_pred_stride.append(
                        nn.Sequential(
                            copy.deepcopy(conv_before_pred[0][j][0]),
                            copy.deepcopy(conv_before_pred[0][j][1]),
                            nn.BatchNorm2d(conv_channels[j + 1]),
                            nn.ReLU(inplace=True),
                        )
                    )

            conv_before_pred.append(conv_before_pred_stride)
        return conv_before_pred

    def _init_predictor(self):
        """Build 1x1 predictors and optional pre-towers for each head."""
        self.conv_cls_prev = self._init_branch(
            conv_channels=self.cls_branch,
            conv_strides=tuple(1 for _ in self.cls_branch),
        )
        self.conv_cls = nn.Conv2d(
            self.cls_branch[-1], self.cls_out_channels, kernel_size=1
        )

        reg_prevs, reg_heads = [], []
        for reg_dim, channels, out_ch in zip(
            self.group_reg_dims, self.reg_branch, self.out_channels
        ):
            if channels:
                reg_prevs.append(
                    self._init_branch(
                        conv_channels=channels,
                        conv_strides=tuple(1 for _ in channels),
                    )
                )
                reg_heads.append(nn.Conv2d(out_ch, reg_dim, kernel_size=1))
            else:
                reg_prevs.append(None)
                reg_heads.append(
                    nn.Conv2d(self.feat_channels, reg_dim, kernel_size=1)
                )
        self.conv_reg_prevs = nn.ModuleList(reg_prevs)
        self.conv_regs = nn.ModuleList(reg_heads)

        if self.use_direction_classifier:
            self.conv_dir_cls_prev = self._init_branch(
                conv_channels=self.dir_branch,
                conv_strides=tuple(1 for _ in self.dir_branch),
            )
            self.conv_dir_cls = nn.Conv2d(
                self.dir_branch[-1], 2, kernel_size=1
            )

        if self.pred_attrs:
            self.conv_attr_prev = self._init_branch(
                conv_channels=self.attr_branch,
                conv_strides=tuple(1 for _ in self.attr_branch),
            )
            self.conv_attr = nn.Conv2d(
                self.attr_branch[-1], self.num_attrs, kernel_size=1
            )

    @staticmethod
    def _init_stride0_convs(tower, std: float = 0.01):
        """Normal-init every Conv2d in the stride-0 stack of a tower."""
        for block in tower[0]:
            for layer in block:
                if isinstance(layer, nn.Conv2d):
                    normal_init(layer, std=std)

    def init_weights(self):
        for tower in (self.cls_convs, self.reg_convs, self.conv_cls_prev):
            self._init_stride0_convs(tower)
        for reg_prev in self.conv_reg_prevs:
            if reg_prev is not None:
                self._init_stride0_convs(reg_prev)

        if self.use_direction_classifier:
            self._init_stride0_convs(self.conv_dir_cls_prev)
        if self.pred_attrs:
            self._init_stride0_convs(self.conv_attr_prev)

        prior_bias = bias_init_with_prob(0.01)
        normal_init(self.conv_cls, std=0.01, bias=prior_bias)
        for reg_head in self.conv_regs:
            normal_init(reg_head, std=0.01)
        if self.use_direction_classifier:
            normal_init(self.conv_dir_cls, std=0.01, bias=prior_bias)
        if self.pred_attrs:
            normal_init(self.conv_attr, std=0.01, bias=prior_bias)

        self._init_stride0_convs(self.conv_centerness_prev)
        normal_init(self.conv_centerness, std=0.01)

    def forward(self, feats):
        feats = _as_list(feats)
        return multi_apply(self.forward_single, feats, self.strides)

    @staticmethod
    def _run_prev_tower(feat, prev_tower, level: int):
        """Apply the level-``level`` pre-tower blocks to ``feat``."""
        out = feat
        for block in prev_tower[level]:
            out = block(out)
        return out

    def forward_single(self, x, stride):
        level = self.strides.index(stride)

        cls_feat = x
        for cls_layer in self.cls_convs[level]:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.conv_cls(
            self._run_prev_tower(cls_feat, self.conv_cls_prev, level)
        )

        reg_feat = x
        for reg_layer in self.reg_convs[level]:
            reg_feat = reg_layer(reg_feat)

        bbox_pred = []
        for gi, channels in enumerate(self.reg_branch):
            feat_i = reg_feat
            if channels:
                feat_i = self._run_prev_tower(
                    feat_i, self.conv_reg_prevs[gi], level
                )
            bbox_pred.append(self.conv_regs[gi](feat_i))

        dir_cls_pred = None
        if self.use_direction_classifier:
            dir_cls_pred = self.conv_dir_cls(
                self._run_prev_tower(reg_feat, self.conv_dir_cls_prev, level)
            )

        attr_pred = None
        if self.pred_attrs:
            attr_pred = self.conv_attr(
                self._run_prev_tower(cls_feat, self.conv_attr_prev, level)
            )

        ctr_src = reg_feat if self.centerness_on_reg else cls_feat
        centerness = self.conv_centerness(
            self._run_prev_tower(ctr_src, self.conv_centerness_prev, level)
        )

        cls_score = self.dequant(cls_score)
        bbox_pred = [self.dequant(x) for x in bbox_pred]
        dir_cls_pred = self.dequant(dir_cls_pred)
        attr_pred = self.dequant(attr_pred)
        centerness = self.dequant(centerness)
        if self.return_for_compiler:
            return (
                cls_score,
                *bbox_pred,
                dir_cls_pred,
                attr_pred,
                centerness,
            )

        bbox_pred = self._decode(bbox_pred, stride)

        return (
            cls_score,
            bbox_pred,
            dir_cls_pred,
            attr_pred,
            centerness,
        )

    @fx_wrap()
    def _decode(self, bbox_pred, stride):
        bbox_pred = torch.cat(bbox_pred, dim=1)
        clone_bbox = bbox_pred.clone()
        bbox_pred[:, :2] = clone_bbox[:, :2].float()
        bbox_pred[:, 2] = clone_bbox[:, 2].float()
        bbox_pred[:, 3:6] = clone_bbox[:, 3:6].float()

        bbox_pred[:, 2] = bbox_pred[:, 2].exp()
        bbox_pred[:, 3:6] = bbox_pred[:, 3:6].exp()
        if not self.training:
            bbox_pred[:, :2] *= stride
        return bbox_pred

    def fuse_model(self):
        def fuse_model_convs(convs: nn.ModuleList):
            for modules in convs:
                for m in modules:
                    if isinstance(m, nn.Sequential):
                        modules_to_fuse = [["1", "2", "3"]]
                    else:
                        raise NotImplementedError(
                            f"Not support type{type(m)} to fuse"
                        )
                    horizon.quantization.fuse_conv_shared_modules(
                        m, modules_to_fuse, inplace=True
                    )

        fuse_model_convs(self.cls_convs)
        fuse_model_convs(self.reg_convs)
        fuse_model_convs(self.conv_cls_prev)
        for module in self.conv_reg_prevs:
            if module is not None:
                fuse_model_convs(module)
        if self.use_direction_classifier:
            fuse_model_convs(self.conv_dir_cls_prev)
        if self.pred_attrs:
            fuse_model_convs(self.conv_attr_prev)
        fuse_model_convs(self.conv_centerness_prev)

    def set_qconfig(self):
        if self.output_int32:
            self.conv_cls.qconfig = get_default_qat_out_qconfig()
            for module in self.conv_regs:
                module.qconfig = get_default_qat_out_qconfig()
            if self.use_direction_classifier:
                self.conv_dir_cls.qconfig = get_default_qat_out_qconfig()
            if self.pred_attrs:
                self.conv_attr.qconfig = get_default_qat_out_qconfig()
            self.conv_centerness.qconfig = get_default_qat_out_qconfig()
        else:
            self.conv_cls.qconfig = get_default_out_qconfig()
            for module in self.conv_regs:
                module.qconfig = get_default_out_qconfig()
            if self.use_direction_classifier:
                self.conv_dir_cls.qconfig = get_default_out_qconfig()
            if self.pred_attrs:
                self.conv_attr.qconfig = get_default_out_qconfig()
            self.conv_centerness.qconfig = get_default_out_qconfig()
