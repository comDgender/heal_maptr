# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import copy
import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.model_helpers import fx_wrap
from horizon_plugin_pytorch.dtype import qint16
from horizon_plugin_pytorch.nn.quantized import FloatFunctional
from torch.quantization import DeQuantStub

from heal.core.box_utils import box_corner_to_center
from heal.models.base_modules.mlp_module import MLP

__all__ = ["DeformableDETR", "DeformDetrIrInfer"]


@OBJECT_REGISTRY.register
class DeformableDETR(nn.Module):
    """Implements the Deformable DETR model.

    Code is modified from the `official github repo
    <https://github.com/fundamentalvision/Deformable-DETR>`_.

    Args:
        backbone: the backbone module.
        position_embedding: the position embedding module.
        neck: the neck module.
        transformer: the transformer module.
        embed_dim: the dimension of the embedding.
        num_classes: Number of total categories.
        num_queries: Number of proposal dynamic anchor boxes in Transformer.
        criterion: Criterion for calculating the total losses.
        post_process: Post process module for inference.
        aux_loss: whether to use auxiliary loss.
        with_box_refine: whether to use box refinement.
        as_two_stage: whether to use two-stage.

    """

    def __init__(
        self,
        backbone: nn.Module,
        position_embedding: nn.Module,
        neck: nn.Module,
        transformer: nn.Module,
        embed_dim: int,
        num_classes: int,
        num_queries: int,
        criterion: Optional[nn.Module] = None,
        post_process: Optional[nn.Module] = None,
        aux_loss: bool = True,
        with_box_refine: bool = False,
        as_two_stage: bool = False,
        set_int16_qconfig: bool = True,
    ):
        super().__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding
        self.neck = neck
        self.set_int16_qconfig = set_int16_qconfig
        self.num_queries = num_queries
        self.transformer = transformer
        self.post_process = post_process
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        self.criterion = criterion
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage

        # Single-stage models keep a learned query bank (content + pos).
        if not as_two_stage:
            self.query_embedding = nn.Embedding(num_queries, embed_dim * 2)

        self.class_embed = nn.Linear(embed_dim, num_classes)
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)

        n_dec = self.transformer.decoder.num_layers
        self.coord_add = nn.ModuleList(
            [FloatFunctional() for _ in range(n_dec)]
        )
        self.tmp_cat = nn.ModuleList([FloatFunctional() for _ in range(n_dec)])
        self.dequant = DeQuantStub()
        self._build_head()

    def _build_head(self):
        """Init cls/bbox heads; clone them for refine / two-stage if needed."""
        # Focal-style prior on classification bias.
        prior = 0.01
        self.class_embed.bias.data.fill_(-math.log((1.0 - prior) / prior))
        # Start bbox residual head near zero.
        nn.init.constant_(self.bbox_embed.layers[-1].weight, 0.0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias, 0.0)

        n_layers = self.transformer.decoder.num_layers
        n_heads = n_layers + 1 if self.as_two_stage else n_layers

        if self.with_box_refine:
            self.class_embed = nn.ModuleList(
                [copy.deepcopy(self.class_embed) for _ in range(n_heads)]
            )
            self.bbox_embed = nn.ModuleList(
                [copy.deepcopy(self.bbox_embed) for _ in range(n_heads)]
            )
            # First refine stage prefers small boxes (log-space wh bias).
            nn.init.constant_(
                self.bbox_embed[0].layers[-1].bias.data[2:], -2.0
            )
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            # Share one head instance across decoder stages.
            self.class_embed = nn.ModuleList([self.class_embed] * n_heads)
            self.bbox_embed = nn.ModuleList([self.bbox_embed] * n_heads)
            self.transformer.decoder.bbox_embed = None

        if self.as_two_stage:
            # Extra last head is used as the region-proposal generator.
            self.transformer.decoder.class_embed = self.class_embed
            for box_head in self.bbox_embed:
                nn.init.constant_(box_head.layers[-1].bias.data[2:], 0.0)

    def gen_img_masks(self, batched_inputs: Dict):
        images = batched_inputs["img"]
        if self.training:
            batch_size, _, H, W = images.shape
            img_masks = images.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                if "before_pad_shape" in batched_inputs:
                    img_h, img_w = batched_inputs["before_pad_shape"][img_id]
                elif "img_shape" in batched_inputs:
                    _, img_h, img_w = batched_inputs["img_shape"][img_id]
                else:
                    img_h, img_w = self.input_shape
                img_masks[img_id, :img_h, :img_w] = 0
        else:
            batch_size, _, H, W = images.shape
            img_masks = torch.zeros(batch_size, H, W).to(images.device)
        return img_masks

    @fx_wrap()
    def compute_pos_emdding(
        self, multi_level_feats: List[torch.Tensor], batched_inputs: Dict
    ):
        """Compute multi scale position embedding and mask."""
        img_masks = self.gen_img_masks(batched_inputs)
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            multi_level_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:])
                .to(torch.bool)
                .squeeze(0)
            )
            multi_level_position_embeddings.append(
                self.position_embedding(multi_level_masks[-1])
            )
        return multi_level_masks, multi_level_position_embeddings

    @fx_wrap()
    def add_tmp2(self, lvl: int, tmp: torch.Tensor, reference: torch.Tensor):
        a, b = tmp[..., :2], tmp[..., 2:]
        a = self.coord_add[lvl].add(a, reference)
        tmp = self.tmp_cat[lvl].cat([a, b], dim=-1)
        return tmp

    def forward(self, batched_inputs: Dict):
        images = batched_inputs["img"]
        features = self.backbone(images)

        multi_level_feats = self.neck(features)

        multi_level_masks, multi_level_pos_embeds = self.compute_pos_emdding(
            multi_level_feats, batched_inputs
        )
        # initialize object query embeddings
        query_embeds = None
        if not self.as_two_stage:
            query_embeds = self.query_embedding.weight

        (
            inter_states,
            inter_bbox_out,
            init_reference_unact,
            inter_references_unact,
            enc_outputs_class,
            enc_outputs_coord_unact,
        ) = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_pos_embeds,
            query_embeds,
        )

        outputs_classes, outputs_coords = self.bbox_head(
            inter_states,
            inter_bbox_out,
            init_reference_unact,
            inter_references_unact,
        )

        outputs = self._post_process(
            batched_inputs,
            outputs_classes,
            outputs_coords,
            enc_outputs_class,
            enc_outputs_coord_unact,
        )
        return outputs

    def bbox_head(
        self,
        inter_states: List[torch.Tensor],
        inter_bbox_out: List[torch.Tensor],
        init_reference_unact: torch.Tensor,
        inter_references_unact: List[torch.Tensor],
    ):
        outputs_classes = []
        outputs_coords = []
        for lvl in range(self.transformer.decoder.num_layers):
            if lvl == 0:
                reference = init_reference_unact
            else:
                reference = inter_references_unact[lvl - 1]

            outputs_class = self.class_embed[lvl](inter_states[lvl])
            if self.with_box_refine:
                tmp = inter_bbox_out[lvl]
            else:
                tmp = self.bbox_embed[lvl](inter_states[lvl])
            if self.as_two_stage or (self.with_box_refine and lvl > 0):
                tmp = self.coord_add[lvl].add(tmp, reference)
            else:
                tmp = self.add_tmp2(lvl, tmp, reference)
            outputs_coord_unact = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord_unact)
        for i in range(self.transformer.decoder.num_layers):
            outputs_classes[i] = self.dequant(outputs_classes[i])
            outputs_coords[i] = self.dequant(outputs_coords[i])
        return outputs_classes, outputs_coords

    @fx_wrap()
    def _post_process(
        self,
        batched_inputs: Dict,
        outputs_classes: List[torch.Tensor],
        outputs_coords: List[torch.Tensor],
        enc_outputs_class: torch.Tensor,
        enc_outputs_coord_unact: torch.Tensor,
    ):
        if self.training:
            outputs_class = torch.stack(outputs_classes)
            outputs_coord = torch.stack(
                [coord.sigmoid() for coord in outputs_coords]
            )
            output = {
                "pred_logits": outputs_class[-1],
                "pred_boxes": outputs_coord[-1],
            }
            if self.aux_loss:
                output["aux_outputs"] = [
                    {"pred_logits": a, "pred_boxes": b}
                    for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
                ]

            if self.as_two_stage:
                enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
                output["enc_outputs"] = {
                    "pred_logits": enc_outputs_class,
                    "pred_boxes": enc_outputs_coord,
                }

            targets = self.prepare_targets(batched_inputs)
            loss_dict = self.criterion(output, targets)
            return loss_dict
        else:
            box_cls = outputs_classes[-1]
            box_pred = outputs_coords[-1]
            results = (box_cls, box_pred)
            if self.post_process:
                results = self.post_process(batched_inputs, box_cls, box_pred)
            return results

    def prepare_targets(self, targets: Dict):
        bs = len(targets["gt_classes"])
        if "before_pad_shape" in targets:
            shapes = targets["before_pad_shape"]
        else:
            shapes = targets["img_shape"]
        for i in range(bs):
            bbox = targets["gt_bboxes"][i].float()
            h, w = shapes[i][-2:]
            image_size_xyxy = torch.as_tensor(
                [w, h, w, h], dtype=torch.float, device=bbox.device
            )
            bbox = bbox / image_size_xyxy
            bbox = box_corner_to_center(bbox)
            targets["gt_classes"] = [c.long() for c in targets["gt_classes"]]
            targets["gt_bboxes"][i] = bbox
        return targets

    def set_qconfig(self):
        from hat_engine.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        if self.set_int16_qconfig:
            self.bbox_embed[self.transformer.decoder.num_layers - 1].layers[
                -1
            ].qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={"dtype": qint16},
            )
            self.coord_add.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={"dtype": qint16},
            )
            self.tmp_cat.qconfig = qconfig_manager.get_qconfig(
                activation_qat_qkwargs={"dtype": qint16},
                activation_calibration_qkwargs={"dtype": qint16},
            )
            for module in [self.backbone, self.transformer]:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()

        self.class_embed[self.transformer.decoder.num_layers - 1].qconfig = (
            qconfig_manager.get_default_qat_out_qconfig()
        )

        self.query_embedding.qconfig = None


@OBJECT_REGISTRY.register
class DeformDetrIrInfer(nn.Module):
    """
    The basic structure of DeformDetrIrInfer.

    Args:
        ir_model: The ir model.
        model_path: The path of hbir model.
        post_process: Post process module.
    """

    def __init__(
        self,
        ir_model: nn.Module,
        use_nv12=False,
        post_process: nn.Module = None,
    ):
        super().__init__()
        self.ir_model = ir_model
        self.post_process = post_process
        self.use_nv12 = use_nv12

    def nv12_runtime(self, data):
        import cv2

        def img2nv12(image):
            image = image.squeeze(0)
            if image.ndim == 3 and image.shape[0] in (1, 3):
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
        dict_data["img_y"], dict_data["img_uv"] = img2nv12(data.cpu().numpy())
        return dict_data

    def forward(self, data):
        if self.use_nv12 is True:
            nv12_img = self.nv12_runtime(data["img"])
            data.update(nv12_img)

        outputs = self.ir_model(data)
        if isinstance(outputs, dict):
            outputs = list(outputs.values())
        if self.post_process is not None:
            box_cls, box_pred = outputs
            results = self.post_process(data, box_cls, box_pred)
            return results
        return outputs
