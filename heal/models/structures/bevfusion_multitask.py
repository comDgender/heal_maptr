# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hat_engine.models.ir_modules.hbm_module import HbmModule
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.apply_func import (  # noqa: F401  (kept for parity)
    _as_list,
)
from torch.quantization import DeQuantStub

from heal.models.task_modules.flashocc.decoder import BevformerOccDetDecoder
from heal.models.task_modules.maptr.decoder import MaptrDetDecoder
from heal.models.task_modules.sparsebev.dense_bev_head import (
    DenseBEVSparseHead,
)
from heal.models.task_modules.sparsebev.instance_bank import InstanceBank

__all__ = ["BEVFusionMultitask", "BevMultitaskIrInfer"]


@OBJECT_REGISTRY.register
class BEVFusionMultitask(nn.Module):
    """BEVFusionMultitask fuses LiDAR and camera data for BEV tasks.

    BEVFusionMultitask is a module that fuses features from LiDAR and
    camera inputs to produce a Bird's Eye View (BEV) representation.
    The fused features are then passed through a series of decoders to generate
    the final outputs, which can be used for tasks like object detection,
    semantic segmentation, etc.

    Args:
        lidar_network: The neural network module for processing LiDAR data.
        camera_network: The neural network module for processing camera data.
        task_decoders: A dict (or module) of decoder modules that process
                the fused BEV features.
        fuse_module: Fusion module that combines LiDAR and camera features.
        bev_h: The height of the BEV feature map.
        bev_w: The width of the BEV feature map.
        loss_weight_dict: Per-task loss weights.
        return_bev_embed: Whether to return the dequantized BEV embed.
        dequantize_bev_embed: Whether to dequantize BEV embed.
    """

    def __init__(
        self,
        lidar_network: Optional[nn.Module] = None,
        camera_network: Optional[nn.Module] = None,
        fuse_module: Optional[nn.Module] = None,
        bev_h: int = 128,
        bev_w: int = 128,
        loss_weight_dict=None,
        task_decoders: Union[nn.Module, Dict[str, nn.Module]] = None,
        return_bev_embed: bool = False,
        dequantize_bev_embed: bool = True,
    ):
        super(BEVFusionMultitask, self).__init__()
        self.loss_weight_dict = loss_weight_dict
        if self.loss_weight_dict is None:
            self.loss_weight_dict = {
                "dynamic_objects_det": 1,
                "occ": 1,
                "maptr": 1,
                "discrete_map": 1,
                "static_objects_det": 1,
            }
        self.lidar_net = lidar_network
        self.camera_net = camera_network
        self.fuse_module = fuse_module
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.task_decoders = task_decoders
        assert lidar_network or camera_network, "at least has one input"

        if isinstance(self.task_decoders, dict):
            self.task_decoders = nn.ModuleDict(self.task_decoders)

        self.return_bev_embed = return_bev_embed
        self.dequantize_bev_embed = dequantize_bev_embed
        if self.return_bev_embed and self.dequantize_bev_embed:
            self.dequant = DeQuantStub()

    def forward_lidar_feature(self, example: dict):
        if self.lidar_net.pre_process is None:
            data = dict(  # noqa C408
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
            data = dict(  # noqa C408
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
        x = self.lidar_net.neck(x)
        return x

    def forward_camera_feature(self, data: dict):
        imgs = data["img"]
        feats = self.camera_net.extract_feat(imgs)
        bev_feat = self.camera_net.view_transformer(feats, data)
        return bev_feat, feats

    def forward(self, data: dict):
        cm_feats = None
        if self.lidar_net:
            lidar_feature = self.forward_lidar_feature(data)

        if self.camera_net:
            # camera_feature = self.forward_camera_feature(data)
            camera_feature, cm_feats = self.forward_camera_feature(data)
            B, _, C = camera_feature.shape
            cx = camera_feature.permute(0, 2, 1).reshape(
                B, C, self.bev_h, self.bev_w
            )
        if self.camera_net and self.lidar_net:
            if cx.shape[-1] != self.bev_w:
                cx = F.interpolate(cx, (self.bev_h, self.bev_w))
            cat_bev = torch.cat([cx, lidar_feature], dim=1)
            pts_feats = self.fuse_module(cat_bev)
            B, C = pts_feats.shape[:2]
        elif self.camera_net:
            pts_feats = cx
        else:
            pts_feats = lidar_feature
            B, C = pts_feats.shape[:2]

        # results = self.post_process(data, pts_feats)
        results = self.post_process(data, pts_feats, cm_feats)
        return results

    def forward_single_task(self, decoder, pts_feats, data, cm_feats):
        # Lazy import to avoid a circular import at module load time.
        from heal.models.task_modules.bevformer_decoder.decoder import (
            BEVFormerDet2DDecoder,
        )

        if isinstance(decoder, MaptrDetDecoder):
            if self.training:
                return decoder(pts_feats, data, cm_feats)
            else:
                return decoder(pts_feats, data)
        elif isinstance(
            decoder,
            (
                DenseBEVSparseHead,
                BevformerOccDetDecoder,
                BEVFormerDet2DDecoder,
            ),
        ):
            return decoder(pts_feats, data)
        else:
            raise ValueError(
                f"Does not support head:{decoder.__class__.__name__}, "
                f"which should be one of MaptrDetDecoder, "  # noqa
                f"DenseBEVSparseHead, BevformerOccDetDecoder, "
                f"BEVFormerDet2DDecoder"
            )

    def post_process(self, data: dict, pts_feats: torch.Tensor, feats=None):
        model_outs = OrderedDict()
        if self.task_decoders is not None:
            if isinstance(self.task_decoders, nn.ModuleDict):
                for decoder_name, decoder in self.task_decoders.items():
                    task_outputs = self.forward_single_task(
                        decoder, pts_feats, data, feats
                    )
                    if isinstance(task_outputs, dict):
                        if self.training:
                            model_outs.update(
                                {
                                    f"{decoder_name}.{k}": (
                                        v * self.loss_weight_dict[decoder_name]
                                        if k
                                        not in (
                                            "dynamic_bev_embed",
                                            "maptr_bev_embed",
                                            "occ_bev_embed",
                                        )
                                        else v
                                    )
                                    for k, v in task_outputs.items()
                                }
                            )
                        else:
                            model_outs.update(
                                {
                                    f"{decoder_name}.{k}": v
                                    for k, v in task_outputs.items()
                                }
                            )
                    else:
                        model_outs[decoder_name] = task_outputs
            elif isinstance(self.task_decoders, nn.Module):
                model_outs = self.forward_single_task(
                    self.task_decoders, pts_feats, data, feats
                )
            else:
                raise ValueError(
                    "task_decoder should be either a list of modules or a "  # noqa
                    "single module."
                )
        if self.return_bev_embed:
            if self.dequantize_bev_embed:
                bev_embed = self.dequant(pts_feats)
            else:
                bev_embed = pts_feats
            model_outs["bev_embed"] = bev_embed
        return model_outs


@OBJECT_REGISTRY.register
class BevMultitaskIrInfer(nn.Module):
    """BevMultitaskIrInfer is a module for performing inference.

    Args:
        ir_model: The ir model.
        lidar_preprocess: The lidar preprocess module.
        global_mat_key: Key to retrieve global matrix.
        first_frame_input: Input data for the first frame.
        max_time_interval: Maximum time interval allowed. Default is 2.
        dynamic_objects_det_decoder: Decoder module of
            dynamic objects det head.
        static_objects_det_decoder: Decoder module of static objects det head.
        maptr_post_process: Decoder module of maptr head.
        discrete_map_post_process: Decoder module of discrete map head.
        use_memory_bank: If True, use memory bank for storing instances.
        dynamic_objects_confidence_decay: Confidence decay factor of dynamic
            objects detection head.
        dynamic_objects_num_temp_instances: Number of temporal instances of
            dynamic objects detection head.
        dynamic_objects_num_memory_instances: Number of memory instances of
            dynamic objects detection head.
        static_objects_confidence_decay: Confidence decay factor of static
            objects detection head.
        static_objects_num_temp_instances: Number of temporal instances of
            static objects detection head.
        static_objects_num_memory_instances: Number of memory instances of
            static objects detection head.
        deploy_model: The deploy model for post-processing.
        batch_size: batch size.
        model_converter_pipeline: Pipeline for converting the model.
        use_nv12: Whether to use nv12 input type for hbm_infer. Default True.
    """

    def __init__(
        self,
        ir_model: nn.Module,
        global_mat_key: str,
        first_frame_input: Dict[str, Any],
        max_time_interval: int = 2,
        lidar_preprocess: nn.Module = None,
        dynamic_objects_det_decoder: nn.Module = None,
        static_objects_det_decoder: nn.Module = None,
        maptr_post_process: nn.Module = None,
        discrete_map_post_process: nn.Module = None,
        use_memory_bank: bool = True,
        dynamic_objects_confidence_decay: float = 0.6,
        dynamic_objects_num_temp_instances: int = 128,
        dynamic_objects_num_memory_instances: int = 384,
        static_objects_confidence_decay: float = 0.6,
        static_objects_num_temp_instances: int = 80,
        static_objects_num_memory_instances: int = 200,
        deploy_model: nn.Module = None,
        batch_size: int = 1,
        model_converter_pipeline: List[callable] = None,
        use_nv12: bool = True,
        # data_converter_pipeline: List[callable] = None,
    ):
        """Initialize BevMultitaskIrInfer with IR model and decoders.

        Args:
            ir_model: IR model module.
            global_mat_key: Key to retrieve global matrix.
            first_frame_input: Input data for the first frame.
            max_time_interval: Maximum time interval allowed.
            lidar_preprocess: LiDAR preprocess module.
            dynamic_objects_det_decoder: Decoder of dynamic objects det head.
            static_objects_det_decoder: Decoder of static objects det head.
            maptr_post_process: Decoder of maptr head.
            discrete_map_post_process: Decoder of discrete map head.
            use_memory_bank: Whether to use memory bank for instances.
            dynamic_objects_confidence_decay: Confidence decay of dynamic head.
            dynamic_objects_num_temp_instances: Dynamic temporal instances.
            dynamic_objects_num_memory_instances: Dynamic memory instances.
            static_objects_confidence_decay: Confidence decay of static head.
            static_objects_num_temp_instances: Static temporal instances.
            static_objects_num_memory_instances: Static memory instances.
            deploy_model: Deploy model for post-processing.
            batch_size: Batch size.
            model_converter_pipeline: Pipeline converting the model.
            use_nv12: Whether to use nv12 input for hbm_infer.
        """
        super(BevMultitaskIrInfer, self).__init__()
        self.ir_model = ir_model
        self.first_frame_input = first_frame_input
        self.dynamic_objects_det_decoder = dynamic_objects_det_decoder
        self.static_objects_det_decoder = static_objects_det_decoder
        self.maptr_post_process = maptr_post_process
        self.discrete_map_post_process = discrete_map_post_process
        self.metas = None
        self.global_mat_key = global_mat_key
        self.max_time_interval = max_time_interval
        self.use_memory_bank = use_memory_bank
        self.dynamic_objects_confidence_decay = (
            dynamic_objects_confidence_decay
        )
        self.dynamic_objects_num_temp_instances = (
            dynamic_objects_num_temp_instances
        )
        self.dynamic_objects_num_memory_instances = (
            dynamic_objects_num_memory_instances
        )
        self.static_objects_confidence_decay = static_objects_confidence_decay
        self.static_objects_num_temp_instances = (
            static_objects_num_temp_instances
        )
        self.static_objects_num_memory_instances = (
            static_objects_num_memory_instances
        )
        self.use_nv12 = use_nv12
        self.batch_size = batch_size
        self.lidar_preprocess = lidar_preprocess
        if model_converter_pipeline is not None:
            self.ir_model = model_converter_pipeline(ir_model)
        self.deploy_model = deploy_model
        self.deploy_model.eval()
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "ego2global": None,
        }

    def process_input(self, data):
        """Export camera-derived reference points and grids for IR input.

        Args:
            data: Input data dict.

        Returns:
            Dict of reference points and BEV grid tensors.
        """
        (
            queries_rebatch_grid,
            restore_bev_grid,
            reference_points_rebatch,
            bev_pillar_counts,
        ) = self.deploy_model.camera_net.export_reference_points(data)

        return {
            "queries_rebatch_grid": queries_rebatch_grid,
            "restore_bev_grid": restore_bev_grid,
            "reference_points_rebatch": reference_points_rebatch,
            "bev_pillar_counts": bev_pillar_counts,
        }

    def img2nv12(self, input_image, device):
        """Convert RGB image to packed NV12 Y and UV tensors.

        Args:
            input_image: Input RGB image array (N, C, H, W).
            device: Torch device of output tensors.

        Returns:
            Tuple of Y tensor and packed UV tensor.
        """
        image = input_image.astype(np.uint8)
        image = image.squeeze(0)
        image = np.transpose(image, (1, 2, 0))
        height, width = image.shape[0], image.shape[1]
        yuv420p = cv2.cvtColor(image, cv2.COLOR_RGB2YUV_I420).reshape(
            (height * width * 3 // 2,)
        )
        y = yuv420p[: height * width]
        uv_planar = yuv420p[height * width :].reshape((2, height * width // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape((height * width // 2,))
        return torch.tensor(
            y.reshape(1, height, width, 1), dtype=torch.uint8, device=device
        ), torch.tensor(
            uv_packed.reshape(1, height // 2, width // 2, 2),
            dtype=torch.uint8,
            device=device,
        )

    def dynamic_objects_cache(self, data: Dict[str, Any]) -> None:
        """Cache input data for future reference.

        Args:
            metas: Metadata associated with current input.
            data: Input data dictionary.
        """
        confidence = data["classification"]
        anchor = data["prediction"]
        instance_feature = data["feature"]
        (
            cached_confidence,
            cached_feature,
            cached_anchor,
        ) = InstanceBank.cache_instance(
            self.dynamic_objects_num_temp_instances,
            instance_feature,
            anchor,
            confidence,
            self.dynamic_objects_cached_confidence,
            self.dynamic_objects_confidence_decay,
        )

        if self.use_memory_bank is True:
            self.dynamic_objects_cached_confidence = torch.cat(
                [cached_confidence, self.dynamic_objects_cached_confidence],
                dim=1,
            )[:, : self.dynamic_objects_num_memory_instances]
            self.dynamic_objects_cached_feature = torch.cat(
                [cached_feature, self.dynamic_objects_cached_feature], dim=1
            )[:, : self.dynamic_objects_num_memory_instances]
            self.dynamic_objects_cached_anchor = torch.cat(
                [cached_anchor, self.dynamic_objects_cached_anchor], dim=1
            )[:, : self.dynamic_objects_num_memory_instances]
        else:
            self.dynamic_objects_cached_anchor = cached_anchor
            self.dynamic_objects_cached_feature = cached_feature
            self.dynamic_objects_cached_confidence = cached_confidence

    def static_objects_cache(self, data: Dict[str, Any]) -> None:
        """Cache metadata and input data for future reference.

        Args:
            metas: Metadata associated with current input.
            data: Input data dictionary.
        """
        confidence = data["classification"]
        anchor = data["prediction"]
        instance_feature = data["feature"]
        (
            cached_confidence,
            cached_feature,
            cached_anchor,
        ) = InstanceBank.cache_instance(
            self.static_objects_num_temp_instances,
            instance_feature,
            anchor,
            confidence,
            self.static_objects_cached_confidence,
            self.static_objects_confidence_decay,
        )

        if self.use_memory_bank is True:
            self.static_objects_cached_confidence = torch.cat(
                [cached_confidence, self.static_objects_cached_confidence],
                dim=1,
            )[:, : self.static_objects_num_memory_instances]
            self.static_objects_cached_feature = torch.cat(
                [cached_feature, self.static_objects_cached_feature], dim=1
            )[:, : self.static_objects_num_memory_instances]
            self.static_objects_cached_anchor = torch.cat(
                [cached_anchor, self.static_objects_cached_anchor], dim=1
            )[:, : self.static_objects_num_memory_instances]
        else:
            self.static_objects_cached_anchor = cached_anchor
            self.static_objects_cached_feature = cached_feature
            self.static_objects_cached_confidence = cached_confidence

    def gen_projection_mat(self):
        """Generate per-view projection matrices from BEV size and view count.

        Returns:
            Projection matrix tensor of shape (batch * num_views, 4, 4).
        """
        bev_size = self.deploy_model.task_decoders.dynamic_objects_det.bev_size
        num_views = (
            self.deploy_model.task_decoders.dynamic_objects_det.num_views
        )
        projection_mat = torch.eye(4)
        projection_mat = projection_mat[None, :, :]
        projection_mat = projection_mat.tile(self.batch_size * num_views, 1, 1)
        scales = [(bev_size[0], bev_size[1])] * (self.batch_size * num_views)

        views = []
        for scale in scales:
            view = torch.eye(4)
            view[0, 0] = 1 / scale[0]
            view[1, 1] = 1 / scale[1]
            views.append(torch.tensor(view).to(device=projection_mat.device))
        views = torch.stack(views)
        projection_mat = torch.matmul(views.double(), projection_mat.double())
        return projection_mat

    def forward(self, metas: Dict[str, Any]) -> Dict[str, Any]:
        """Perform forward pass through the model.

        Args:
            metas: Metadata associated with current input.

        Returns:
            Output dictionary containing model predictions or a tuple of
            decoded results depending on whether a decoder is provided.
        """
        if isinstance(self.ir_model, HbmModule) and self.use_nv12:
            input_data = {}
            for i in range(metas["img"].shape[0]):
                (
                    input_data[f"img_{i}_y"],
                    input_data[f"img_{i}_uv"],
                ) = self.img2nv12(
                    metas["img"][i : i + 1, :, :, :].cpu().numpy(),
                    device=metas["img"].device,
                )
        else:
            input_data = {
                "img": metas["img"],
            }

        projection_mat = self.gen_projection_mat()
        input_data["projection_mat"] = projection_mat.float()
        input_data.update(self.process_input(metas))

        device = metas["img"].device
        if self.metas is None:
            for k, v in self.first_frame_input.items():
                setattr(self, k, v.to(device))
        else:
            history_time = self.metas["timestamp"]
            time_interval = metas["timestamp"] - history_time
            time_interval = time_interval.to(
                device=self.dynamic_objects_cached_anchor.device
            )
            self.mask = torch.abs(time_interval) <= self.max_time_interval
            T_temp2cur = InstanceBank.gen_temp2cur(
                self.global_mat_key, self.metas, metas
            ).to(device)
            # dynamic_objects_det_head
            self.dynamic_objects_cached_anchor = (
                InstanceBank.anchor_projection(
                    self.dynamic_objects_cached_anchor,
                    T_temp2cur,
                    time_interval,
                )
            )

            # static_objects_det_head
            self.static_objects_cached_anchor = InstanceBank.anchor_projection(
                self.static_objects_cached_anchor,
                T_temp2cur,
                time_interval,
            )

            self.dynamic_objects_cached_anchor = torch.where(
                self.mask[:, None, None],
                self.dynamic_objects_cached_anchor,
                torch.zeros_like(self.dynamic_objects_cached_anchor),
            )
            self.dynamic_objects_cached_feature = torch.where(
                self.mask[:, None, None],
                self.dynamic_objects_cached_feature,
                torch.zeros_like(self.dynamic_objects_cached_feature),
            )
            self.static_objects_cached_anchor = torch.where(
                self.mask[:, None, None],
                self.static_objects_cached_anchor,
                torch.zeros_like(self.static_objects_cached_anchor),
            )
            self.static_objects_cached_feature = torch.where(
                self.mask[:, None, None],
                self.static_objects_cached_feature,
                torch.zeros_like(self.static_objects_cached_feature),
            )
        input_data["dynamic_objects_cached_anchor"] = (
            self.dynamic_objects_cached_anchor
        )  # noqa
        input_data["dynamic_objects_cached_feature"] = (
            self.dynamic_objects_cached_feature
        )  # noqa
        input_data["static_objects_cached_anchor"] = (
            self.static_objects_cached_anchor
        )  # noqa
        input_data["static_objects_cached_feature"] = (
            self.static_objects_cached_feature
        )  # noqa
        if self.use_memory_bank is False:
            input_data["dynamic_objects_cached_confidence"] = (
                self.dynamic_objects_cached_confidence
            )  # noqa
            input_data["dynamic_objects_mask"] = self.mask
            input_data["static_objects_cached_confidence"] = (
                self.static_objects_cached_confidence
            )  # noqa
            input_data["static_objects_mask"] = self.mask

        outs = self.ir_model(input_data)
        if isinstance(self.ir_model, (HbmModule,)):
            output_names = [
                "dynamic_objects_det.classification",
                "dynamic_objects_det.prediction",
                "dynamic_objects_det.quality",
                "dynamic_objects_det.feature",
                "static_objects_det.classification",
                "static_objects_det.prediction",
                "static_objects_det.quality",
                "static_objects_det.feature",
                "occ",
                "bev_embed",
                "discrete_map.classification",
                "discrete_map.ref_out",
                "discrete_map.prediction",
                "om.classification",
                "om.prediction",
            ]

            new_outs = {}
            outs_flatten = [outs[k] for k in outs]
            for i, name in enumerate(output_names):
                new_outs[name] = outs_flatten[i]
            outs = new_outs
            occ_pred = [outs["occ"]]
        else:
            outs["dynamic_objects_det.classification"] = outs[
                "dynamic_objects_det.classification"
            ]
            outs["dynamic_objects_det.prediction"] = outs[
                "dynamic_objects_det.prediction"
            ]
            outs["dynamic_objects_det.quality"] = outs[
                "dynamic_objects_det.quality"
            ]
            outs["dynamic_objects_det.feature"] = outs[
                "dynamic_objects_det.feature"
            ]
            outs["static_objects_det.classification"] = outs[
                "static_objects_det.classification"
            ]
            outs["static_objects_det.prediction"] = outs[
                "static_objects_det.prediction"
            ]
            outs["static_objects_det.quality"] = outs[
                "static_objects_det.quality"
            ]
            outs["static_objects_det.feature"] = outs[
                "static_objects_det.feature"
            ]

            outs["bev_embed"] = outs["discrete_map"][0]
            outs["discrete_map.classification"] = outs["discrete_map"][1]
            outs["discrete_map.ref_out"] = outs["discrete_map"][2]
            outs["discrete_map.prediction"] = outs["discrete_map"][3]
            outs["om.classification"] = outs["maptr"][0]
            outs["om.prediction"] = outs["maptr"][1]
            occ_pred = outs["occ"]
        occ_score = occ_pred[0].softmax(-1)
        occ_res = occ_score.argmax(-1)

        dynamic_objects_cache_data = {
            "classification": outs["dynamic_objects_det.classification"],
            "prediction": outs["dynamic_objects_det.prediction"],
            "feature": outs["dynamic_objects_det.feature"],
        }
        static_objects_cache_data = {
            "classification": outs["static_objects_det.classification"],
            "prediction": outs["static_objects_det.prediction"],
            "feature": outs["static_objects_det.feature"],
        }

        self.metas = metas
        self.dynamic_objects_cache(dynamic_objects_cache_data)
        self.static_objects_cache(static_objects_cache_data)
        dynamic_objects_det_results = self.dynamic_objects_det_decoder(
            outs["dynamic_objects_det.classification"],
            outs["dynamic_objects_det.prediction"],
            outs.get("dynamic_objects_det.quality"),
        )
        static_objects_det_results = self.static_objects_det_decoder(
            outs["static_objects_det.classification"],
            outs["static_objects_det.prediction"],
            outs.get("static_objects_det.quality"),
        )
        om_results = self.deploy_model.task_decoders.maptr.bev_decoders[
            0
        ].get_outputs(
            outputs_classes=[outs["om.classification"]],
            reference_out=[outs["om.prediction"]],
        )
        om_results = self.maptr_post_process(om_results, metas)

        discrete_map_result = (
            self.deploy_model.task_decoders.discrete_map.get_outputs(
                [outs["discrete_map.classification"]],
                [outs["discrete_map.ref_out"]],
                [outs["discrete_map.prediction"]],
                outs["bev_embed"],
            )
        )
        discrete_map_result = self.discrete_map_post_process(
            discrete_map_result
        )
        results = {}
        results["dynamic_objects_det"] = dynamic_objects_det_results
        results["static_objects_det"] = static_objects_det_results
        # Match float OccDecoder ``[occ_pre]`` and SparseMultitaskIrInfer
        # so val callback ``model_outs["occ"][0]`` keeps the batch axis.
        results["occ"] = [occ_res]
        results["maptr"] = om_results
        results["discrete_map"] = discrete_map_result
        return results
