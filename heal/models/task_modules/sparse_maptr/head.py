# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
"""Sparse MapTR detection head.

Each decoder layer repeats::

    interaction -> norm -> deformable -> ffn -> norm -> refine

- ``interaction``: self-attention among map query instances.
- ``deformable``: multi-view feature aggregation at 3D key points.
- ``refine``: update anchor polyline coords and class logits.

Quantization stubs (``quant_anchor``, ``mat_quant_stub``, etc.) wrap inputs
that cross float/int boundaries; ``dequant`` is applied on head outputs.
Per-op qconfig is configured via ``QconfigSetter`` in task config, not
``set_qconfig`` on the module.
"""

import copy
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from hat_engine.registry import OBJECT_REGISTRY
from horizon_plugin_pytorch.quantization import QuantStub
from torch.amp import autocast
from torch.quantization import DeQuantStub

__all__ = [
    "SparseMapHead",
]


@OBJECT_REGISTRY.register
class SparseMapHead(nn.Module):
    """Sparse query head for online map element detection.

    Maintains ``num_anchor`` learnable map queries. Each query represents one
    polyline (divider / crossing / boundary) with ``num_pts_per_vec`` 2D points.  # noqa: E501

    The head stacks ``num_decoder`` layers; each layer runs the ops listed in
    ``operation_order``. Temporal interaction is optional via ``instance_bank``.  # noqa: E501
    """

    def __init__(
        self,
        feat_indices: Sequence[int],
        num_anchor: int,
        num_pts_per_vec: int,
        anchor_encoder: nn.Module,
        instance_interaction: nn.Module,
        norm_layer: nn.Module,
        ffn: nn.Module,
        deformable_model: nn.Module,
        refine_layer: nn.Module,
        instance_bank: nn.Module,
        temp_instance_interaction: Optional[nn.Module] = None,
        num_decoder: int = 6,
        num_single_frame_decoder: int = -1,
        operation_order: Optional[List[str]] = None,
        num_views: int = 6,
        projection_mat_key: str = "lidar2img",
    ):
        super(SparseMapHead, self).__init__()

        self.feat_indices = feat_indices
        self.num_anchor = num_anchor
        self.num_pts = num_pts_per_vec

        self.num_decoder = num_decoder
        self.num_single_frame_decoder = num_single_frame_decoder
        if operation_order is None:
            operation_order = [
                "interaction",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_decoder
        self.operation_order = operation_order

        self.op_map = {
            "temp_interaction": [temp_instance_interaction, 0],
            "interaction": [instance_interaction, 0],
            "norm": [norm_layer, 0],
            "ffn": [ffn, 0],
            "deformable": [deformable_model, 0],
            "lidar_deformable": [deformable_model, 0],
            "refine": [refine_layer, 0],
        }
        self.layers = nn.ModuleList()
        for op in self.operation_order:
            if op not in self.op_map:
                self.layers.append(None)
            elif self.op_map[op][1] == 0:
                self.layers.append(self.op_map[op][0])
            else:
                self.layers.append(copy.deepcopy(self.op_map[op][0]))
            self.op_map[op][1] += 1
        self.anchor_encoder = anchor_encoder

        self.init_weights()

        # Quant boundaries for anchor coords, query features, and projection mat.  # noqa: E501
        self.quant_anchor = QuantStub()
        self.quant_instance_feature = QuantStub()

        self.mat_quant_stub = QuantStub()

        # Dequant before returning cls/reg outputs to loss / post-process.
        self.dequant = DeQuantStub()

        self.num_views = num_views
        self.projection_mat_key = projection_mat_key
        self.instance_bank = instance_bank

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    @staticmethod
    def gen_projection_mat(
        projection_mat_key: str, metas: Dict[str, Any]
    ) -> torch.Tensor:
        """Modify the given homography matrix based on the feature and image shapes.

        Args:
            projection_mat_key: key in the metas for projection_mat.
            meta: Metadata dictionary.

        Returns:
            homopgrahpy: Modified homography matrix.
        """
        scales = [np.array(img.shape[::-1][:2]) for img in metas["img"]]
        projection_mat = metas[projection_mat_key]

        views = []
        for scale in scales:
            view = np.eye(4)
            view[0, 0] = 2 / scale[0]
            view[1, 1] = 2 / scale[1]
            view[0, 2] = -1
            view[1, 2] = -1
            views.append(torch.tensor(view).to(device=projection_mat.device))
        views = torch.stack(views)
        projection_mat = torch.matmul(views.double(), projection_mat.double())
        return projection_mat

    @autocast("cuda", enabled=False)
    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
        compiler_model: str = False,
        lidar_feature: nn.Module = None,
    ):
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]

        feature_maps = [
            feature_maps[index].float() for index in self.feat_indices
        ]
        batch_size = feature_maps[0].shape[0] // self.num_views

        dn_metas = None
        (
            instance_feature,
            anchor,
            temp_instance_feature,
            temp_anchor,
        ) = self.instance_bank.get(batch_size, metas, dn_metas, compiler_model)

        attn_mask = None

        anchor = self.quant_anchor(anchor)
        instance_feature = self.quant_instance_feature(instance_feature)

        # Encode normalized anchor geometry into query positional embedding.
        anchor_embed = self.anchor_encoder(anchor)
        if self.instance_bank.num_temp_instances > 0:
            temp_anchor_embed = self.anchor_encoder(temp_anchor)
        else:
            temp_anchor_embed = None
            temp_instance_feature = None

        if compiler_model is True:
            # Deploy path: metas carries a precomputed ``projection_mat`` (the
            # deploy example_inputs / IR-infer input set). Calibration tracing
            # (Float2Calibration with example_data_loader, convert_mode
            # "jit-strip" -> horizon.quantization.prepare) drives this branch
            # via preparing_qat=True, but feeds a *real nuScenes batch* (no
            # ``projection_mat`` key — that key only exists in the precomputed
            # deploy input set). Fall back to recomputing it from lidar2img
            # (which the real batch has) instead of KeyError'ing. Use
            # try/except rather than ``"projection_mat" in metas``: the latter
            # iterates a dict Proxy under FX and raises TraceError.
            try:
                projection_mat = metas["projection_mat"]
            except KeyError:
                projection_mat = SparseMapHead.gen_projection_mat(
                    self.projection_mat_key, metas
                )
        else:
            projection_mat = SparseMapHead.gen_projection_mat(
                self.projection_mat_key, metas
            )
        projection_mat = self.mat_quant_stub(
            projection_mat.view(batch_size, -1, 4, 4).float()
        )

        # Stack decoder layers; collect per-layer cls/reg for deep supervision.
        prediction = []
        classification = []
        for i, op in enumerate(self.operation_order):
            if op == "temp_interaction":
                value = temp_instance_feature
                instance_feature = self.layers[i](
                    instance_feature,
                    temp_instance_feature,
                    value,
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                    attn_mask=(
                        attn_mask if temp_instance_feature is None else None
                    ),
                )
            elif op == "interaction":
                instance_feature = self.layers[i](
                    instance_feature,
                    value=instance_feature,
                    query_pos=anchor_embed,
                    attn_mask=attn_mask,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "identity":
                identity = instance_feature
            elif op == "add":
                instance_feature = instance_feature + identity
            elif op == "deformable":
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    projection_mat,
                )
            elif op == "lidar_deformable":
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    lidar_feature,
                    projection_mat,
                )

            elif op == "refine":
                # Residual update of polyline coords; emit dequantized outputs.
                anchor, cls = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    return_cls=True,
                )
                # prediction.append(anchor)
                prediction.append(self.dequant(anchor))
                classification.append(self.dequant(cls))
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
                if (
                    len(prediction) > self.num_single_frame_decoder
                    and temp_anchor_embed is not None
                ):
                    temp_anchor_embed = anchor_embed[
                        :, : self.instance_bank.num_temp_instances
                    ]
            else:
                raise NotImplementedError(f"{op} is not supported.")
        output = {
            "classification": (
                classification[-1] if compiler_model else classification
            ),  # classification
            "prediction": (
                prediction[-1] if compiler_model else prediction
            ),  # regression
        }
        return output
