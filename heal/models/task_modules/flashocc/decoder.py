# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import horizon_plugin_pytorch.nn as hnn
import torch
from hat_engine.models.base_modules.conv_module import ConvModule2d
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.model_helpers import fx_wrap
from horizon_plugin_pytorch.qtensor import QTensor
from torch import Tensor, nn
from torch.quantization import DeQuantStub

from heal.models.task_modules.bev.decoder import BevDecoderInfer

__all__ = [
    "FlashOccDetDecoder",
    "FlashOccDecoderInfer",
    "FlashOccHead",
    "BevformerOccDetDecoder",
]


@OBJECT_REGISTRY.register
class FlashOccDetDecoder(nn.Module):
    """The basic decoder structure of flashocc.

    Args:
        occ_head: Head module.
        loss_occ: Classify loss module.
        roi_resizer: RoI resizer module.
        use_mask: Whether to use masking in the decoder.
        lidar_input: Whether to use LiDAR input.
        camera_input: Whether to use camera input.
        num_classes: Number of classes for prediction.
        is_compile: Whether for compile.
    """

    def __init__(
        self,
        occ_head: Optional[nn.Module] = None,
        loss_occ: nn.Module = None,
        roi_resizer: nn.Module = None,
        use_mask: bool = True,
        lidar_input: bool = False,
        camera_input: bool = True,
        num_classes: int = 18,
        is_compile: bool = False,
        is_train: bool = False,
        **kwargs,
    ):
        super(FlashOccDetDecoder, self).__init__(**kwargs)
        self.occ_head = occ_head
        self.roi_resizer = roi_resizer
        self.is_compile = is_compile
        self.loss_occ = loss_occ
        self.use_mask = use_mask
        self.lidar_input = lidar_input
        self.camera_input = camera_input
        self.num_classes = num_classes
        self.is_train = is_train
        self.dequant = DeQuantStub()

    @fx_wrap()
    def _post_process(self, occ_preds, data):
        """Post process."""
        if self.training or self.is_train:
            voxel_semantics = data["gt_occ_info"]["voxel_semantics"]
            if self.lidar_input and self.camera_input:
                mask = (
                    data["gt_occ_info"]["mask_lidar"]
                    | data["gt_occ_info"]["mask_camera"]
                )
            elif self.lidar_input:
                mask = data["gt_occ_info"]["mask_lidar"]
            elif self.camera_input:
                mask = data["gt_occ_info"]["mask_camera"]
            loss_occ = self._loss(
                occ_preds,
                voxel_semantics,
                mask,
            )
            return loss_occ
        else:
            occ_pred = self._decode(occ_preds)
            return {"occ_pre": occ_pred}

    def _loss(self, occ_pred, voxel_semantics, mask_camera):
        """Compute losses for occ task."""
        loss = {}
        voxel_semantics = voxel_semantics.long()
        if self.use_mask:
            mask_camera = mask_camera.to(torch.int32)
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = occ_pred.reshape(-1, self.num_classes)
            mask_camera = mask_camera.reshape(-1)

            num_total_samples = mask_camera.sum()

            loss_occ = self.loss_occ(
                preds,
                voxel_semantics,
                mask_camera,
                avg_factor=num_total_samples,
            )
            loss["loss_occ"] = loss_occ
        else:
            voxel_semantics = voxel_semantics.reshape(-1)
            preds = occ_pred.reshape(-1, self.num_classes)

            num_total_samples = len(voxel_semantics)

            loss_occ = self.loss_occ(
                preds, voxel_semantics, avg_factor=num_total_samples
            )

            loss["loss_occ"] = loss_occ
        return loss

    def _decode(self, pred: Tensor):
        """Decode the predicted values."""
        occ_score = pred.softmax(-1)
        occ_res = occ_score.argmax(-1)
        return occ_res

    def forward(self, bev_feat: List[Tensor], data: Dict):
        """Perform the forward pass of the model."""
        if isinstance(bev_feat, Sequence):
            bev_feat = bev_feat[0]
        if self.roi_resizer is not None:
            bev_feat = self.roi_resizer([bev_feat])["roi_feat0"]

        occ_preds = self.occ_head(bev_feat)
        occ_preds = self.dequant(occ_preds)
        if self.is_compile:
            return [occ_preds], {}
        return [occ_preds], self._post_process(occ_preds, data)


@OBJECT_REGISTRY.register
class BevformerOccDetDecoder(FlashOccDetDecoder):
    """The basic decoder structure of flashocc.

    Args:
        occ_head: Head module.
        loss_occ: Classify loss module.
        roi_resizer: RoI resizer module.
        use_mask: Whether to use masking in the decoder.
        lidar_input: Whether to use LiDAR input.
        camera_input: Whether to use camera input.
        num_classes: Number of classes for prediction.
        is_compile: Whether for compile.
        use_adapter: Whther to use adapter
    """

    def __init__(
        self,
        occ_head: Optional[nn.Module] = None,
        loss_occ: nn.Module = None,
        roi_resizer: nn.Module = None,
        use_mask: bool = True,
        lidar_input: bool = False,
        camera_input: bool = True,
        num_classes: bool = 18,
        is_compile: bool = False,
        use_adapter: bool = False,
        in_dim: int = 256,
        is_train: bool = False,
        return_bev_embed: bool = False,
        **kwargs,
    ):
        super(BevformerOccDetDecoder, self).__init__(
            occ_head,
            loss_occ,
            roi_resizer,
            use_mask,
            lidar_input,
            camera_input,
            num_classes,
            is_compile,
            is_train,
            **kwargs,
        )
        self.use_adapter = use_adapter
        self.is_train = is_train
        self.return_bev_embed = return_bev_embed
        if self.use_adapter:
            self.skip_add = nn.quantized.FloatFunctional()
            self.adapter = nn.Sequential(
                ConvModule2d(
                    in_dim,
                    in_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    norm_layer=nn.BatchNorm2d(in_dim),
                    act_layer=nn.ReLU(inplace=False),
                ),
                ConvModule2d(
                    in_dim,
                    in_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_layer=nn.BatchNorm2d(in_dim),
                    act_layer=None,
                    # act_layer=nn.ReLU(inplace=False),
                ),
            )
        if self.return_bev_embed:
            self.bev_dequant = DeQuantStub()

    def forward(self, bev_feat: List[Tensor], data: Dict):
        if not self.is_compile:
            gt_occ_info = data["seq_meta"][0]
        else:
            gt_occ_info = {}
        if self.use_adapter:
            bev_feat = self.skip_add.add(bev_feat, self.adapter(bev_feat))
        # Save occ_bev_embed before roi_resizer modifies spatial dims
        occ_bev_embed = None
        if self.return_bev_embed:
            occ_bev_embed = self.bev_dequant(bev_feat)
        pred, result = super(BevformerOccDetDecoder, self).forward(
            bev_feat, gt_occ_info
        )
        if os.getenv("HAT_TRAINING_STEP", "float") == "quant_analysis":
            return pred
        if self.is_compile:
            return pred
        if self.training or self.is_train:
            # result is loss dict — attach embed after so it
            # survives the parent _post_process which drops it
            if occ_bev_embed is not None:
                result["occ_bev_embed"] = occ_bev_embed
            return result
        else:
            # result is {"occ_pre": ...} — attach embed
            if occ_bev_embed is not None:
                result["occ_bev_embed"] = occ_bev_embed
                return [result]
            else:
                return [result["occ_pre"]]


@OBJECT_REGISTRY.register
class FlashOccDecoderInfer(BevDecoderInfer):
    def forward(self, occ_pred, data):
        occ_score = occ_pred[0].softmax(-1)
        occ_res = occ_score.argmax(-1)
        return None, {"occ_pre": occ_res}


@OBJECT_REGISTRY.register
class FlashOccHead(nn.Module):
    """FlashOcc head: LSS view transform + BEV encoder + occ decoder."""

    def __init__(
        self,
        view_transformer: nn.Module,
        bev_encoder: nn.Module,
        bev_decoder: nn.Module,
        bev_feat_index: int,
        bev_upscale: float,
        bev_transforms: Optional[List] = None,
    ):
        super(FlashOccHead, self).__init__()
        self.bev_feat_index = bev_feat_index
        self.bev_upscale = bev_upscale
        self.view_transformer = view_transformer
        self.bev_encoder = bev_encoder
        self.bev_decoder = bev_decoder
        if self.bev_upscale > 1:
            self.resize = hnn.Interpolate(
                scale_factor=self.bev_upscale,
                align_corners=None,
                recompute_scale_factor=True,
            )
        self.bev_transforms = bev_transforms
        if self.bev_transforms is None:
            self.bev_transforms = []

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: Dict[str, Any],
        compiler_model: bool = False,
        lidar_feature=None,
    ):
        feat = feature_maps[self.bev_feat_index]
        bev_feat, aux_data = self.view_transformer(feat, metas, compiler_model)
        if self.bev_upscale > 1:
            bev_feat = self.resize(bev_feat)

        bev_feat, metas = self._transform(bev_feat, metas)
        if self.bev_encoder:
            bev_feat = self.bev_encoder(bev_feat, metas)
        if not isinstance(bev_feat, Sequence):
            bev_feat = [bev_feat]

        pred, result = self.bev_decoder(bev_feat, metas)

        if compiler_model:
            return pred
        if self.training:
            return result
        return [result["occ_pre"]]

    @fx_wrap()
    def _transform(self, bev_feat: Tensor, data: Dict) -> Tuple[Tensor, Dict]:
        if self.training and not isinstance(bev_feat, QTensor):
            for transform in self.bev_transforms:
                bev_feat, data = transform(bev_feat, data)
        return bev_feat, data
