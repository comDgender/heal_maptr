# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import List, Optional

import torch
import torch.nn as nn
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.model_helpers import fx_wrap

torch.fx.wrap("len")

__all__ = ["CenterPointDetector", "CenterPointDetectorIrInfer"]


@OBJECT_REGISTRY.register
class CenterPointDetector(nn.Module):
    """CenterPoint lidar detector."""

    def __init__(
        self,
        feature_map_shape: List[int],
        pre_process: Optional[nn.Module] = None,
        reader: Optional[nn.Module] = None,
        scatter: Optional[nn.Module] = None,
        backbone: Optional[nn.Module] = None,
        neck: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
        targets: Optional[nn.Module] = None,
        loss: Optional[nn.Module] = None,
        postprocess: Optional[nn.Module] = None,
        quant_begin_neck: bool = False,
        is_deploy: bool = False,
    ):
        super().__init__()
        self.pre_process = pre_process
        self.reader = reader
        self.scatter = scatter
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.targets = targets
        self.loss = loss
        self.postprocess = postprocess
        self.quant_begin_neck = quant_begin_neck
        self.feature_map_shape = feature_map_shape
        self.is_deploy = is_deploy

    def forward(self, example):
        # pre_process is None: features/coors; else points (use_max if deploy).
        if self.pre_process is None:
            data = dict(  # noqa: C408
                features=example["features"],
                coors=example["coors"],
                num_points_in_voxel=None,
                batch_size=1,
                input_shape=self.feature_map_shape,
            )
        else:
            features, coords = self.pre_process(
                example["points"], self.is_deploy or (not self.training)
            )
            data = dict(  # noqa: C408
                features=features,
                coors=coords,
                num_points_in_voxel=None,
                batch_size=len(example["points"]),
                input_shape=self.feature_map_shape,
            )

        input_features = self.reader(
            data["features"],
            horizon_preprocess=True,
        )
        if self.scatter is not None:
            x = self.scatter(
                input_features,
                data["coors"],
                data["batch_size"],
                torch.tensor(self.feature_map_shape),
            )
            x = self.backbone(x)
        else:
            x = self.backbone(
                input_features,
                data["coors"],
                data["batch_size"],
                torch.tensor(self.feature_map_shape),
            )
        x = self.neck(x)
        if self.head:
            outs = self.head(x)
        else:
            return x

        if self.is_deploy:
            outs_list = []
            for out in outs:
                for _, v in out.items():
                    outs_list.append(v)
            return outs_list

        return self._post_process(example, outs)

    @fx_wrap()
    def _post_process(self, example, outs):
        if self.training:
            heatmaps, anno_boxes, inds, masks = self.targets(
                example["gt_boxes"], example["gt_classess"]
            )
            return self.loss(
                heatmaps,
                anno_boxes,
                inds,
                masks,
                outs,
            )

        if self.postprocess is not None:
            return self.postprocess(outs)
        return outs

    def fuse_model(self):
        if not self.quant_begin_neck and self.reader:
            self.reader.fuse_model()

        for module in [self.neck, self.head]:
            if module is not None and hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat_engine.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        if self.quant_begin_neck:
            for module in [self.reader, self.backbone]:
                if module is not None:
                    module.qconfig = None
        else:
            for module in [self.reader, self.backbone]:
                if module is not None and hasattr(module, "set_qconfig"):
                    module.set_qconfig()

        for module in [self.neck, self.head]:
            if module is not None and hasattr(module, "set_qconfig"):
                module.set_qconfig()

        for module in [self.targets, self.loss, self.postprocess]:
            if module is not None:
                module.qconfig = None

    def set_calibration_qconfig(self):
        from hat_engine.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_calibration_qconfig()

        if self.quant_begin_neck:
            for module in [self.reader, self.backbone]:
                if module is not None:
                    module.qconfig = None
        else:
            for module in [self.reader, self.backbone]:
                if module is not None and hasattr(
                    module, "set_calibration_qconfig"
                ):
                    module.set_calibration_qconfig()

        for module in [self.neck, self.head]:
            if module is not None and hasattr(
                module, "set_calibration_qconfig"
            ):
                module.set_calibration_qconfig()

        for module in [self.targets, self.loss, self.postprocess]:
            if module is not None:
                module.qconfig = None


@OBJECT_REGISTRY.register
class CenterPointDetectorIrInfer(nn.Module):
    """CenterPoint HBIR inference wrapper."""

    def __init__(
        self,
        ir_model: nn.Module,
        pre_process: nn.Module,
        feature_map_shape: List[int],
        postprocess: nn.Module,
        tasks: Optional[List[dict]],
        headkeys: List[str] = (
            "reg",
            "height",
            "dim",
            "rot",
            "vel",
            "heatmap",
        ),
        transpose_output: bool = False,
    ):
        super().__init__()
        self.ir_model = ir_model
        self.pre_process = pre_process
        self.feature_map_shape = feature_map_shape
        self.postprocess = postprocess
        self.headkeys = headkeys
        self.tasks = tasks
        self.transpose_output = transpose_output

    def forward(self, example):
        features, coords = self.pre_process(example["points"], True)
        data = dict(  # noqa: C408
            features=features,
            coors=coords.to(torch.int32),
            num_points_in_voxel=None,
            batch_size=len(example["points"]),
            input_shape=self.feature_map_shape,
        )
        hbir_outputs = self.ir_model(data)
        if isinstance(hbir_outputs, dict):
            hbir_outputs = list(hbir_outputs.values())
        if self.transpose_output:
            hbir_outputs = [out.permute(0, 3, 1, 2) for out in hbir_outputs]
        results_convert = []
        start = 0
        for _ in range(len(self.tasks)):
            task_dict = {}
            for j, key in enumerate(self.headkeys):
                task_dict.update({key: hbir_outputs[start + j]})
            results_convert.append(task_dict)
            start += len(self.headkeys)
        return self.postprocess(results_convert)
