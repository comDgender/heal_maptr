# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
"""Sparse MapTR building blocks for anchor geometry.

These modules sit inside ``SparseMapHead`` and ``DeformableFeatureAggregation``:  # noqa: E501

- ``SparsePoint3DEncoder``: polyline coord -> positional embedding.
- ``SparsePoint3DKeyPointsGenerator``: sample 3D key points on each anchor.
- ``SparsePoint3DRefinementModule``: residual coord/class update per layer.

Several ops here (``quant_fix_height``, ``Scale``) are qconfig-sensitive;
their dtypes/scales are configured in ``configs/maptr_sparse_henet_tinym/qconfig.py``.  # noqa: E501
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from hat_engine.registry import OBJECT_REGISTRY
from horizon_plugin_pytorch.quantization import QuantStub

from heal.models.task_modules.sparsebev.blocks import Scale, linear_relu_ln
from heal.models.weight_init import xavier_init

__all__ = [
    "SparsePoint3DEncoder",
    "SparsePoint3DKeyPointsGenerator",
    "SparsePoint3DRefinementModule",
]


@OBJECT_REGISTRY.register
class SparsePoint3DEncoder(nn.Module):
    """MLP encoder: flattened polyline coords -> ``embed_dims`` query embedding."""  # noqa: E501

    def __init__(
        self,
        embed_dims: int = 256,
        input_dim: int = 2,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.input_dim = input_dim

        def embedding_layer(input_dims):
            return nn.Sequential(*linear_relu_ln(embed_dims, 1, 2, input_dims))

        self.pos_fc = embedding_layer(input_dim)

    def forward(self, points: torch.Tensor):
        point_feat = self.pos_fc(points)
        return point_feat


@OBJECT_REGISTRY.register
class SparsePoint3DKeyPointsGenerator(nn.Module):
    """Generate 3D sampling key points for deformable feature aggregation.

    Each map query anchor is a polyline with ``num_sample`` points. Learnable
    offsets plus fixed height offsets expand them into 3D key points projected
    onto camera views by ``DeformableFeatureAggregationOEv2``.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_sample: int = 20,
        num_learnable_pts: int = 0,
        fix_height: Tuple = (0,),
        ground_height: int = 0,
    ):
        super(SparsePoint3DKeyPointsGenerator, self).__init__()
        self.embed_dims = embed_dims
        self.num_sample = num_sample
        self.num_learnable_pts = num_learnable_pts
        self.num_pts = num_sample * len(fix_height) * num_learnable_pts
        if self.num_learnable_pts > 0:
            self.learnable_fc = nn.Linear(self.embed_dims, self.num_pts * 2)

        self.fix_height = np.array(fix_height)
        self.ground_height = ground_height
        # Fixed height offsets are quantized constants at export time.
        self.quant_fix_height = QuantStub()

    def init_weight(self):
        if self.num_learnable_pts > 0:
            xavier_init(self.learnable_fc, distribution="uniform", bias=0.0)

    def forward(
        self,
        anchor,
        instance_feature=None,
        T_cur2temp_list=None,
        cur_timestamp=None,
        temp_timestamps=None,
    ):
        assert self.num_learnable_pts > 0, "No learnable pts"
        bs, num_anchor, _ = anchor.shape
        key_points = anchor.view(bs, num_anchor, self.num_sample, -1)
        offset = self.learnable_fc(instance_feature).reshape(
            bs,
            num_anchor,
            self.num_sample,
            len(self.fix_height),
            self.num_learnable_pts,
            2,
        )
        key_points = offset + key_points[..., None, None, :]
        key_points = torch.cat(
            [
                key_points,
                key_points.new_ones(key_points.shape[:-1] + (1,))
                * self.ground_height,
            ],
            dim=-1,
        )
        fix_height = key_points.new_tensor(self.fix_height)
        fix_height = self.quant_fix_height(fix_height)
        height_offset = key_points.new_zeros([len(fix_height), 2])
        height_offset = torch.cat([height_offset, fix_height[:, None]], dim=-1)
        key_points = key_points + height_offset[None, None, None, :, None]
        key_points = key_points.flatten(2, 4)
        skip_temp = (
            cur_timestamp is None
            or temp_timestamps is None
            or T_cur2temp_list is None
            or len(temp_timestamps) == 0
        )
        if skip_temp:
            return key_points

        temp_key_points_list = []
        ones = torch.ones_like(key_points[..., :1])
        for ti in range(len(temp_timestamps)):
            T = T_cur2temp_list[ti].to(dtype=key_points.dtype)
            homo = torch.cat([key_points, ones], dim=-1).unsqueeze(-1)
            # Apply SE(3) rotation+translation to homogeneous keypoints.
            warped = (T[:, None, None, :3] @ homo).squeeze(-1)
            temp_key_points_list.append(warped)
        return key_points, temp_key_points_list

    def anchor_projection(
        self,
        anchor,
        T_src2dst_list,
        src_timestamp=None,
        dst_timestamps=None,
        time_intervals=None,
    ):
        dst_anchors = []
        bs, num_anchor, _ = anchor.shape
        for T_mat in T_src2dst_list:
            pts = anchor.reshape(bs, num_anchor, self.num_sample, -1).flatten(
                1, 2
            )
            T = T_mat.to(dtype=anchor.dtype).unsqueeze(1)
            # 2D rigid transform on polyline samples.
            rot = T[..., :2, :2]
            trans = T[..., :2, 3]
            pts = (rot @ pts.unsqueeze(-1)).squeeze(-1) + trans
            pts = pts.reshape(bs, num_anchor, self.num_sample, -1).flatten(
                2, 3
            )
            dst_anchors.append(pts)
        return dst_anchors


@OBJECT_REGISTRY.register
class SparsePoint3DRefinementModule(nn.Module):
    """Per-layer residual refinement of polyline coords and class logits.

    ``Scale`` applies a learnable per-dim gain on coord deltas; its output is
    one of the qconfig-sensitive nodes kept at higher precision during cali.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_sample: int = 20,
        coords_dim: int = 2,
        num_cls: int = 3,
        with_cls_branch: bool = True,
    ):
        super(SparsePoint3DRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.num_sample = num_sample
        self.output_dim = num_sample * coords_dim
        self.num_cls = num_cls
        self.with_cls_branch = with_cls_branch

        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            nn.Linear(embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim),
        )
        if with_cls_branch:
            self.cls_layers = nn.Sequential(
                *linear_relu_ln(embed_dims, 1, 2),
                nn.Linear(embed_dims, num_cls),
            )

    def init_weight(self):
        if not self.with_cls_branch:
            return
        prior = 0.01
        bias = float(-np.log((1.0 - prior) / prior))
        nn.init.constant_(self.cls_layers[-1].bias, bias)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        time_interval: torch.Tensor = 1.0,
        return_cls=True,
    ):
        fused = instance_feature + anchor_embed
        output = self.layers(fused) + anchor
        if return_cls:
            assert self.with_cls_branch, "Without classification layers !!!"
            cls = self.cls_layers(instance_feature)
        else:
            cls = None
        return output, cls
