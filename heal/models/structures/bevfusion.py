# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.apply_func import _as_list

__all__ = ["BevFusion", "BevFusionHbirInfer"]


@OBJECT_REGISTRY.register
class BevFusion(nn.Module):
    """Fuse LiDAR and camera inputs for multi-task BEV prediction."""

    def __init__(
        self,
        lidar_network: Optional[nn.Module] = None,
        camera_network: Optional[nn.Module] = None,
        bev_decoders: Optional[List[nn.Module]] = None,
        fuse_module: Optional[nn.Module] = None,
        bev_h: int = 128,
        bev_w: int = 128,
        use_nv12: bool = False,
    ):
        super().__init__()
        self.lidar_net = lidar_network
        self.camera_net = camera_network
        self.fuse_module = fuse_module
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.use_nv12 = use_nv12
        assert lidar_network or camera_network, "at least has one input"
        self.bev_decoders = nn.ModuleList(bev_decoders)

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
        x = self.lidar_net.backbone(
            input_features,
            data["coors"],
            data["batch_size"],
            torch.tensor(self.lidar_net.feature_map_shape),
        )
        return self.lidar_net.neck(x)

    def forward_camera_feature(self, data: dict):
        imgs = data["img"]
        feats = self.camera_net.extract_feat(imgs)
        return self.camera_net.view_transformer(feats, data)

    def forward(self, data: dict):
        if self.lidar_net:
            lidar_feature = self.forward_lidar_feature(data)

        if self.camera_net:
            camera_feature = self.forward_camera_feature(data)
            B, _, C = camera_feature.shape
            cx = camera_feature.permute(0, 2, 1).reshape(
                B, C, self.bev_h, self.bev_w
            )
        if self.camera_net and self.lidar_net:
            if cx.shape[-1] != self.bev_w:
                cx = F.interpolate(cx, (self.bev_h, self.bev_w))
            cat_bev = torch.cat([cx, lidar_feature], dim=1)
            pts_feats = self.fuse_module(cat_bev)
        elif self.camera_net:
            pts_feats = cx
        else:
            pts_feats = lidar_feature

        return self.post_process(data, pts_feats)

    def post_process(self, data: dict, pts_feats: torch.Tensor):
        if self.training:
            loss = {}
            for bev_decoder in self.bev_decoders:
                result = bev_decoder(pts_feats, data)
                loss.update(result)
            return loss
        results = []
        for bev_decoder in self.bev_decoders:
            result = bev_decoder(pts_feats, data)
            results.append(result)
        return results


@OBJECT_REGISTRY.register
class BevFusionHbirInfer(nn.Module):
    """HBIR inference wrapper for BevFusion."""

    def __init__(
        self,
        ir_model: nn.Module,
        deploy_model: nn.Module,
        use_nv12=False,
    ):
        super().__init__()
        self.ir_model = ir_model
        self.deploy_model = deploy_model
        self.deploy_model.eval()
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "ego2global": None,
        }
        self.use_nv12 = use_nv12

    def export_reference_points(self, data: Dict) -> Dict:
        return self.deploy_model.camera_net.export_reference_points(data)

    def nv12_runtime(self, data):
        """Convert an NCHW RGB uint8 image to NV12 (y + uv) tensors.

        Mirrors SparseBEVIrInfer.nv12_runtime: for batch>1 (e.g. 6 cameras)
        returns per-camera ``img_{i}_y`` / ``img_{i}_uv`` entries; for batch==1
        returns ``img_y`` / ``img_uv``.
        """

        def img2nv12(input_image):
            image = input_image
            image = image.squeeze(0)
            image = np.transpose(image, (1, 2, 0))
            height, width = image.shape[0], image.shape[1]
            image = image.astype(np.uint8)
            yuv420p = cv2.cvtColor(image, cv2.COLOR_RGB2YUV_I420).reshape(
                (height * width * 3 // 2,)
            )
            y = yuv420p[: height * width]
            uv_planar = yuv420p[height * width :].reshape(
                (2, height * width // 4)
            )
            uv_packed = uv_planar.transpose((1, 0)).reshape(
                (height * width // 2,)
            )
            return torch.tensor(
                y.reshape(1, height, width, 1), dtype=torch.uint8
            ), torch.tensor(
                uv_packed.reshape(1, height // 2, width // 2, 2),
                dtype=torch.uint8,
            )

        dict_data = {}
        if data.shape[0] == 1:
            dict_data["img_y"], dict_data["img_uv"] = img2nv12(
                data.cpu().numpy()
            )
        else:
            for i in range(data.shape[0]):
                (
                    dict_data[f"img_{i}_y"],
                    dict_data[f"img_{i}_uv"],
                ) = img2nv12(data[i : i + 1, :, :, :].cpu().numpy())
        return dict_data

    def process_input(self, data):
        image = data["img"]
        features, coords = self.deploy_model.lidar_net.pre_process(
            data["points"], True
        )
        (
            queries_rebatch_grid,
            restore_bev_grid,
            reference_points_rebatch,
            bev_pillar_counts,
        ) = self.export_reference_points(data)
        inputs = {
            "features": features,
            "coors": coords.to(torch.int32),
            "queries_rebatch_grid": queries_rebatch_grid,
            "restore_bev_grid": restore_bev_grid,
            "reference_points_rebatch": reference_points_rebatch,
            "bev_pillar_counts": bev_pillar_counts,
        }
        if self.use_nv12:
            # NV12 path: replace the ``img`` tensor with per-camera
            # img_y/img_uv NV12 planes the HbmModule consumes
            # (input_source=pyramid).
            inputs.update(self.nv12_runtime(image))
        else:
            inputs["img"] = image
        return inputs

    def forward(self, data):
        inputs = self.process_input(data)
        outputs = self.ir_model(inputs)
        if isinstance(outputs, dict):
            outputs = list(outputs.values())
            outputs = [
                [outputs[0], outputs[1], outputs[2], outputs[3]],
                [outputs[4]],
            ]
        bev_embed, outputs_classes, reference_out, outputs_coords = outputs[0]
        bev_outputs = self.deploy_model.bev_decoders[0].get_outputs(
            [outputs_classes], [reference_out], [outputs_coords], bev_embed
        )
        result = self.deploy_model.bev_decoders[0].post_process(bev_outputs)
        bev_results = self._update_res(result, None)

        occ_outputs = outputs[1]
        occ_score = occ_outputs[0].softmax(-1)
        occ_res = occ_score.argmax(-1)
        return bev_results, [occ_res]

    def _update_res(self, result: List, results: List) -> List:
        if results is None:
            return _as_list(result)
        results.extend(_as_list(result))
        return results
