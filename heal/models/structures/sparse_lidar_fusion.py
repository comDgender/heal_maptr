# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
from typing import Any, Dict, List, Optional, Union

import torch
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.global_var import get_value
from horizon_plugin_pytorch.fx.jit_scheme import Tracer
from torch import nn

from heal.models.task_modules.sparsebev.head import SparseBEVHead
from heal.models.task_modules.sparsebev.instance_bank import InstanceBank
from .sparsebev import SparseBEVIrInfer

__all__ = ["SparseBevFusion", "SparseBEVFusionIrInfer"]


@OBJECT_REGISTRY.register
class SparseBevFusion(nn.Module):
    """Sparse end-to-end BEV model with camera and lidar fusion."""

    def __init__(
        self,
        lidar_net: nn.Module,
        backbone: nn.Module,
        head: nn.Module,
        neck: Optional[nn.Module] = None,
        depth_branch: Optional[nn.Module] = None,
        lidar_level_idx: Optional[List[int]] = None,
        compiler_model: bool = False,
        only_lidar: bool = False,
    ):
        super(SparseBevFusion, self).__init__()
        self.lidar_net = lidar_net
        self.lidar_level_idx = lidar_level_idx
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.depth_branch = depth_branch
        self.compiler_model = compiler_model
        self.only_lidar = only_lidar

    def extract_feat(self, img: torch.Tensor) -> List:
        feature_maps = self.backbone(img)
        return list(self.neck(feature_maps))

    def _post_process(
        self,
        feature_maps: List,
        model_outs: Dict[str, Any],
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.compiler_model is False:
            if (
                self.training is True
                and Tracer.is_tracing() is False
                and not get_value("preparing_qat")
            ):
                if self.depth_branch is not None and not self.only_lidar:
                    depths = self.depth_branch(feature_maps, data)
                    model_outs["loss_dense_depth"] = self.depth_branch.loss(
                        depths, data
                    )
        return model_outs

    def forward_lidar_feature(self, example: dict):
        if self.lidar_net.pre_process is None:
            data = dict(  # noqa: C408
                features=example["features"],
                coors=example["coors"],
                num_points_in_voxel=None,
                batch_size=1,
                input_shape=self.lidar_net.feature_map_shape,
            )
        else:
            features, coords = self.lidar_net.pre_process(
                example["points"], not self.training
            )
            data = dict(  # noqa: C408
                features=features,
                coors=coords,
                num_points_in_voxel=None,
                batch_size=len(example["points"]),
                input_shape=self.lidar_net.feature_map_shape,
            )

        input_features = self.lidar_net.reader(
            data["features"],
            horizon_preprocess=True,
        )
        x = self.lidar_net.scatter(
            input_features,
            data["coors"],
            data["batch_size"],
            torch.tensor(self.lidar_net.feature_map_shape),
        )
        if self.lidar_net.backbone:
            x = self.lidar_net.backbone(x)
        if self.lidar_net.neck:
            x = self.lidar_net.neck(x)
        if self.lidar_net.head:
            x = self.lidar_net.head(x)
        if self.lidar_level_idx is not None:
            x = [x[i] for i in self.lidar_level_idx]
        return x

    def forward(self, data: Dict[str, Any]) -> Union[Dict, List]:
        if not self.only_lidar:
            feature_maps = self.extract_feat(data["img"])
        else:
            feature_maps = None
        lidar_feature = self.forward_lidar_feature(data)
        model_outs = self.head(
            feature_maps,
            data,
            compiler_model=self.compiler_model,
            lidar_feature=lidar_feature,
        )
        return self._post_process(feature_maps, model_outs, data)

    def set_qconfig(self) -> None:
        for module in [self.backbone, self.neck, self.head, self.lidar_net]:
            if module is not None and hasattr(module, "set_qconfig"):
                module.set_qconfig()


@OBJECT_REGISTRY.register
class SparseBEVFusionIrInfer(SparseBEVIrInfer):
    """HBIR inference wrapper for SparseBevFusion."""

    def __init__(
        self,
        ir_model: nn.Module,
        projection_mat_key: str,
        global_mat_key: str,
        first_frame_input: Dict[str, Any],
        max_time_interval: int = 2,
        lidar_preprocess: nn.Module = None,
        decoder: nn.Module = None,
        use_memory_bank: bool = True,
        confidence_decay: float = 0.6,
        num_temp_instances: int = 128,
        num_memory_instances: int = 384,
        use_nv12: bool = False,
    ):
        super(SparseBEVFusionIrInfer, self).__init__(
            ir_model=ir_model,
            projection_mat_key=projection_mat_key,
            global_mat_key=global_mat_key,
            first_frame_input=first_frame_input,
            max_time_interval=max_time_interval,
            decoder=decoder,
            use_memory_bank=use_memory_bank,
            confidence_decay=confidence_decay,
            num_temp_instances=num_temp_instances,
            num_memory_instances=num_memory_instances,
            use_nv12=use_nv12,
        )
        self.lidar_preprocess = lidar_preprocess

    def forward(self, metas: Dict[str, Any]) -> Dict[str, Any]:
        input_data = {"img": metas["img"]}
        if self.use_nv12:
            input_data = self.nv12_runtime(input_data["img"])

        projection_mat = SparseBEVHead.gen_projection_mat(
            self.projection_mat_key, metas
        ).float()
        input_data["projection_mat"] = projection_mat
        features, coords = self.lidar_preprocess(metas["points"], True)
        input_data["features"] = features
        input_data["coors"] = coords.to(torch.int32)

        if self.metas is None:
            for k, v in self.first_frame_input.items():
                setattr(self, k, v.to(metas["img"].device))
        else:
            history_time = self.metas["timestamp"]
            time_interval = metas["timestamp"] - history_time
            time_interval = time_interval.to(device=self.cached_anchor.device)
            self.mask = torch.abs(time_interval) <= self.max_time_interval
            T_temp2cur = InstanceBank.gen_temp2cur(
                self.global_mat_key, self.metas, metas
            ).to(self.cached_anchor.device)
            self.cached_anchor = InstanceBank.anchor_projection(
                self.cached_anchor,
                T_temp2cur,
                time_interval,
            )
        input_data["cached_anchor"] = self.cached_anchor
        input_data["cached_feature"] = self.cached_feature
        if self.use_memory_bank is False:
            input_data["cached_confidence"] = self.cached_confidence
            input_data["mask"] = self.mask

        outs = self.ir_model(input_data)
        for k, v in outs.items():
            outs[k] = v.to(self.cached_anchor.device)
        self.cache(metas, outs)
        return self.decoder(
            outs["classification"], outs["prediction"], outs.get("quality")
        )
