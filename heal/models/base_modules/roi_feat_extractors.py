# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from collections import OrderedDict
from typing import List, Union

import horizon_plugin_pytorch.nn as hnn
import torch.nn as nn
from hat_engine.registry import OBJECT_REGISTRY

__all__ = ["RoiResize"]


@OBJECT_REGISTRY.register
class RoiResize(nn.Module):
    """Crop and optionally resize features from a feature map list."""

    def __init__(
        self,
        in_strides: List[int],
        roi_resize_cfgs: Union[dict, List[dict]],
        resize_mode: str = "bilinear",
    ):
        super().__init__()
        self.in_strides = in_strides
        if not isinstance(roi_resize_cfgs, list):
            roi_resize_cfgs = [roi_resize_cfgs]
        self.roi_resize_cfg = roi_resize_cfgs
        if "output_size" in roi_resize_cfgs[0]:
            self.resize_modules = nn.ModuleList(
                [
                    hnn.Interpolate(
                        size=roi_resize_cfg["output_size"], mode=resize_mode
                    )
                    for roi_resize_cfg in roi_resize_cfgs
                ]
            )
        else:
            self.resize_modules = None

    def _crop_feature(self, feature, crop_box):
        crop_box = [int(x) for x in crop_box]
        x1, y1, x2, y2 = crop_box
        return feature[:, :, y1:y2, x1:x2]

    def forward(self, data):
        out = OrderedDict()
        for i, roi_resize_cfg in enumerate(self.roi_resize_cfg):
            cur_feat = data[self.in_strides.index(roi_resize_cfg["in_stride"])]
            roi_feat = self._crop_feature(cur_feat, roi_resize_cfg["roi_box"])
            if self.resize_modules is not None:
                roi_resize_feat = self.resize_modules[i](roi_feat)
            else:
                roi_resize_feat = roi_feat
            if "roi_feat_name" in roi_resize_cfg:
                out[roi_resize_cfg["roi_feat_name"]] = roi_resize_feat
            else:
                out[f"roi_feat{i}"] = roi_resize_feat
        return out

    def set_qconfig(self):
        from hat_engine.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
