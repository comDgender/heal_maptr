# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.global_var import get_value
from horizon_plugin_pytorch.fx.jit_scheme import Tracer
from torch import nn

from heal.models.task_modules.sparsebev.head import SparseBEVHead
from heal.models.task_modules.sparsebev.instance_bank import InstanceBank

__all__ = ["SparseBEV", "SparseBEVIrInfer"]


@OBJECT_REGISTRY.register
class SparseBEV(nn.Module):
    """
    SparseBEV is a module for a sparse end-to-end bev model.

    Args:
        backbone : The backbone network for feature extraction.
        head : The module for processing extracted features.
        neck : Optional module for additional feature processing.
        depth_branch : Optional module for depth estimation.
        compiler_model : Indicates if the model is for compiler purposes.
    """

    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        neck: Optional[nn.Module] = None,
        depth_branch: Optional[nn.Module] = None,
        compiler_model: bool = False,
    ):
        super(SparseBEV, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.depth_branch = depth_branch
        self.compiler_model = compiler_model

    def extract_feat(self, img: torch.Tensor) -> List:
        """
        Extract features from an input image.

        Args:
            img : Input image tensor.

        Returns:
            List: List of feature maps extracted.
        """
        feature_maps = self.backbone(img)
        feature_maps = list(self.neck(feature_maps))

        return feature_maps

    def _post_process(
        self,
        feature_maps: List,
        model_outs: Dict[str, Any],
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Perform post-processing on model outputs.

        Args:
            feature_maps : List of feature maps extracted from input image.
            model_outs : Output dictionary from the model.
            data : Input data dictionary.

        Returns:
            Output dictionary containing processed outputs or losses.
        """
        if self.compiler_model is False:
            if (
                self.training is True
                and Tracer.is_tracing() is False
                and not get_value("preparing_qat")
            ):
                if self.depth_branch is not None:
                    depths = self.depth_branch(feature_maps, data)
                    model_outs["loss_dense_depth"] = self.depth_branch.loss(
                        depths, data
                    )
        return model_outs

    def forward(self, data: Dict[str, Any]) -> Union[Dict, List]:
        """
        Define the forward pass of the model.

        Args:
            data : Input data dictionary containing 'img'.

        Returns:
            Output dictionary or list containing processed outputs.
        """
        feature_maps = self.extract_feat(data["img"])

        model_outs = self.head(feature_maps, data, self.compiler_model)
        model_outs = self._post_process(feature_maps, model_outs, data)

        return model_outs

    def set_qconfig(self) -> None:
        """Set the qconfig."""
        for module in [self.backbone, self.neck, self.head]:
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()


@OBJECT_REGISTRY.register
class SparseBEVIrInfer(nn.Module):
    """
    SparseBEVIrInfer is a module for performing inference.

    Args:
        ir_model: The ir model.
        projection_mat_key: Key to retrieve projection matrix.
        global_mat_key: Key to retrieve global matrix.
        first_frame_input: Input data for the first frame.
        max_time_interval: Maximum time interval allowed. Default is 2.
        decoder: Decoder module for processing model outputs.
    """

    def __init__(
        self,
        ir_model: nn.Module,
        projection_mat_key: str,
        global_mat_key: str,
        first_frame_input: Dict[str, Any],
        max_time_interval: int = 2,
        decoder: nn.Module = None,
        use_memory_bank: bool = True,
        confidence_decay: float = 0.6,
        num_temp_instances: int = 128,
        num_memory_instances: int = 384,
        use_nv12: bool = False,
    ):
        super(SparseBEVIrInfer, self).__init__()

        self.ir_model = ir_model
        self.first_frame_input = first_frame_input
        self.decoder = decoder
        self.metas = None
        self.projection_mat_key = projection_mat_key
        self.global_mat_key = global_mat_key
        self.max_time_interval = max_time_interval
        self.use_memory_bank = use_memory_bank
        self.confidence_decay = confidence_decay
        self.num_temp_instances = num_temp_instances
        self.num_memory_instances = num_memory_instances
        self.use_nv12 = use_nv12

    def cache(self, metas: Dict[str, Any], data: Dict[str, Any]) -> None:
        """
        Cache metadata and input data for future reference.

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
            self.num_temp_instances,
            instance_feature,
            anchor,
            confidence,
            self.cached_confidence,
            self.confidence_decay,
        )
        self.metas = metas
        if self.use_memory_bank is True:
            self.cached_confidence = torch.cat(
                [cached_confidence, self.cached_confidence], dim=1
            )[:, : self.num_memory_instances]
            self.cached_feature = torch.cat(
                [cached_feature, self.cached_feature], dim=1
            )[:, : self.num_memory_instances]
            self.cached_anchor = torch.cat(
                [cached_anchor, self.cached_anchor], dim=1
            )[:, : self.num_memory_instances]
        else:
            self.cached_anchor = cached_anchor
            self.cached_feature = cached_feature
            self.cached_confidence = cached_confidence

    def nv12_runtime(self, data):
        import cv2

        def img2nv12(input_image):
            image = input_image  # .astype(np.uint8)
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

    def forward(self, metas: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform forward pass through the model.

        Args:
            metas: Metadata associated with current input.

        Returns:
            Output dictionary containing model predictions
            or tuple of decoded results depending on whether
            a decoder is provided.
        """
        input_data = {"img": metas["img"]}
        if self.use_nv12 is True:
            nv12_img = self.nv12_runtime(input_data["img"])
            input_data.update(nv12_img)

        projection_mat = SparseBEVHead.gen_projection_mat(
            self.projection_mat_key, metas
        ).float()
        input_data["projection_mat"] = projection_mat
        if self.metas is None:
            for k, v in self.first_frame_input.items():
                setattr(self, k, v)
        else:
            history_time = self.metas["timestamp"]
            time_interval = metas["timestamp"] - history_time
            time_interval = time_interval.to(device=self.cached_anchor.device)
            self.mask = torch.abs(time_interval) <= self.max_time_interval
            T_temp2cur = InstanceBank.gen_temp2cur(
                self.global_mat_key, self.metas, metas
            )
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
        results = self.decoder(
            outs["classification"], outs["prediction"], outs.get("quality")
        )
        return results
