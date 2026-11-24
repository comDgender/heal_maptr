# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

"""BEVFormer BEV encoder (Transformer encoder stack).

Each ``BEVFormerEncoderLayer`` runs three sub-steps in order:

1. **TSA** (``HorizonTemporalSelfAttention``): fuse current BEV queries with
   ego-aligned previous-frame BEV.
2. **SCA** (``HorizonSpatialCrossAttention``): deformable cross-attention from
   multi-view image features into BEV grid.
3. **FFN**: per-query feed-forward refinement.

``SingleFrameBEVFormerEncoder*`` variants drop temporal fusion (TSA uses
self-attention on current queries only).
"""

import logging
from typing import List, Optional, Tuple, Union

import torch
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.model_helpers import fx_wrap
from horizon_plugin_pytorch.nn.quantized import FloatFunctional as FF
from horizon_plugin_pytorch.quantization import QuantStub
from torch import Tensor, nn
from torch.nn.init import normal_

from heal.models.base_modules.bevformer_attention import (
    HorizonMultiScaleDeformableAttention,
    HorizonMultiScaleDeformableAttention3D,
    HorizonSpatialCrossAttention,
    HorizonTemporalSelfAttention,
)
from heal.models.base_modules.ffn import FFN
from heal.models.task_modules.bevformer.utils import get_clone_module

logger = logging.getLogger(__name__)

__all__ = [
    "BEVFormerEncoder",
    "BEVFormerEncoderLayer",
    "SingleFrameBEVFormerEncoder",
    "SingleFrameBEVFormerEncoderLayer",
]


@OBJECT_REGISTRY.register
class BEVFormerEncoder(nn.Module):
    """Stack of BEVFormer encoder layers with cam/level embeddings.

    Flattens multi-camera multi-level FPN features, adds learnable camera and
    level embeddings, then iterates ``num_layers`` encoder layers. When
    ``prev_bev`` is provided, current and previous queries are concatenated
    for TSA inside each layer.

    Args:
        bev_h: BEV grid height (number of pillars along Y).
        bev_w: BEV grid width (number of pillars along X).
        embed_dims: Transformer hidden dimension.
        encoder_layer: Single layer template (cloned ``num_layers`` times).
        use_cams_embeds: Add per-camera embedding to image features.
        num_feature_levels: Number of FPN levels fed into SCA.
        num_cams: Number of surround cameras.
        num_layers: Number of encoder layers.
        return_intermediate: Return outputs from every layer.
    """

    def __init__(
        self,
        bev_h: int,
        bev_w: int,
        embed_dims: int,
        encoder_layer: nn.Module,
        use_cams_embeds: bool = True,
        num_feature_levels: int = 1,
        num_cams: int = 6,
        num_layers: int = 3,
        return_intermediate: bool = False,
    ):

        super(BEVFormerEncoder, self).__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.embed_dims = embed_dims
        self.use_cams_embeds = use_cams_embeds

        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims)
        )
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims)
        )

        # Concat [prev_bev, cur_bev_query] along query dim for TSA value input.
        self.cat_pre_cur_query = FF()

        self.addcams_embeds = nn.ModuleList()
        self.add_level_embeds = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            self.addcams_embeds.append(FF())
            self.add_level_embeds.append(FF())

        self.quant_cams_embeds = QuantStub()
        self.quant_level_embeds = QuantStub()
        self.cat_fl = FF()

        self.layers = get_clone_module(encoder_layer, num_layers)
        self.return_intermediate = return_intermediate
        self.init_weights()

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if (
                isinstance(m, HorizonSpatialCrossAttention)
                or isinstance(m, HorizonMultiScaleDeformableAttention3D)
                or isinstance(m, HorizonTemporalSelfAttention)
                or isinstance(m, HorizonMultiScaleDeformableAttention)
            ):
                m.init_weights()

        normal_(self.level_embeds)
        normal_(self.cams_embeds)

    @fx_wrap()
    def _flatten_mlvl_feats(
        self, mlvl_feats: List[Tensor]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Flatten (B, N_cam, C, H, W) feats to SCA key/value sequence.

        Adds camera embedding and FPN-level embedding per pixel, then
        concatenates all levels along the spatial-token dimension.
        """
        feat_flatten = []
        spatial_shapes = []

        level_embed_spilted = self.quant_level_embeds(self.level_embeds).split(
            1, dim=0
        )
        cams_embeds = (
            self.quant_cams_embeds(self.cams_embeds).unsqueeze(1).unsqueeze(1)
        )

        for lvl, feat in enumerate(mlvl_feats):
            _, _, _, h, w = feat.shape
            spatial_shape = (w, h)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if self.use_cams_embeds:
                feat = self.addcams_embeds[lvl].add(feat, cams_embeds)
            feat = self.add_level_embeds[lvl].add(
                feat, level_embed_spilted[lvl].unsqueeze(0).unsqueeze(0)
            )
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = self.cat_fl.cat(feat_flatten, 2)
        cross_spatial_shapes = torch.tensor(
            spatial_shapes, dtype=torch.float32, device=feat_flatten.device
        )
        # BEV grid shape for TSA deformable attention (single level).
        sa_spatial_shapes = torch.tensor(
            [[self.bev_w, self.bev_h]], device=feat_flatten.device
        ).to(torch.float32)
        feat_flatten = feat_flatten.permute(0, 2, 1, 3)
        return feat_flatten, cross_spatial_shapes, sa_spatial_shapes

    def _run_encoder_layers(
        self,
        bev_query: Tensor,
        feat_flatten: Tensor,
        bev_pos: Tensor,
        ref_2d: Tensor,
        sa_spatial_shapes: Tensor,
        cross_spatial_shapes: Tensor,
        queries_rebatch_grid: Tensor,
        restore_bev_grid: Tensor,
        reference_points_rebatch: Tensor,
        bev_pillar_counts: Tensor,
        prev_bev: Optional[Tensor] = None,
    ) -> Union[Tensor, List[Tensor]]:
        """Run stacked encoder layers; each layer = TSA -> SCA -> FFN."""
        output = bev_query
        intermediate = []
        for _, layer in enumerate(self.layers):
            layer_inputs = dict(  # noqa: C408
                query=bev_query,
                key=feat_flatten,
                value=feat_flatten,
                query_pos=bev_pos,
                ref_2d=ref_2d,
                sa_spatial_shapes=sa_spatial_shapes,
                cross_spatial_shapes=cross_spatial_shapes,
                queries_rebatch_grid=queries_rebatch_grid,
                restore_bev_grid=restore_bev_grid,
                reference_points_rebatch=reference_points_rebatch,
                bev_pillar_counts=bev_pillar_counts,
            )
            if prev_bev is not None:
                layer_inputs["prev_bev"] = prev_bev
            output = layer(**layer_inputs)
            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        if self.return_intermediate:
            return intermediate
        return output

    @fx_wrap()
    def get_encoder_inputs(
        self,
        hybird_ref_2d: Tensor,
        bev_query: Tensor,
        bev_pos: Tensor,
        prev_bev: Tensor,
        mlvl_feats: List[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Prepare encoder inputs: layout fix + temporal query concat."""
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        bs, len_bev, _, _ = hybird_ref_2d.shape
        # Stack prev and cur BEV queries for TSA (2 * len_bev tokens).
        prev_bev = self.cat_pre_cur_query.cat(
            [prev_bev, bev_query], dim=1
        ).reshape(bs, len_bev, -1)

        (
            feat_flatten,
            cross_spatial_shapes,
            sa_spatial_shapes,
        ) = self._flatten_mlvl_feats(mlvl_feats)

        return (
            bev_query,
            feat_flatten,
            bev_pos,
            hybird_ref_2d,
            cross_spatial_shapes,
            sa_spatial_shapes,
            prev_bev,
        )

    def forward(
        self,
        bev_query: Tensor,
        mlvl_feats: List[Tensor],
        bev_pos: Tensor,
        prev_bev: Tensor,
        hybird_ref_2d: Tensor,
        queries_rebatch_grid: Tensor,
        restore_bev_grid: Tensor,
        reference_points_rebatch: Tensor,
        bev_pillar_counts: Tensor,
    ) -> Tensor:
        """Forward BEVFormerEncoder.

        Args:
            bev_query: Input bev query.
            mlvl_feats: Input multi-cameta features.
            bev_pos: bev query pos embed.
            prev_bev: Previous frame bev feat.
            hybird_ref_2d: Hybird refpoints for temporal attention.
            queries_rebatch_grid: The grid for gridsample to sample bev \
                query for rebatch.
            restore_bev_grid: The grid for gridsample to restore bev query.
            reference_points_rebatch: The refpoints for spatial attention.
            bev_pillar_counts: Each bev pillar corresponds to the \
                number of cameras.
        """

        (
            bev_query,
            feat_flatten,
            bev_pos,
            hybird_ref_2d,
            cross_spatial_shapes,
            sa_spatial_shapes,
            prev_bev,
        ) = self.get_encoder_inputs(
            hybird_ref_2d, bev_query, bev_pos, prev_bev, mlvl_feats
        )

        return self._run_encoder_layers(
            bev_query,
            feat_flatten,
            bev_pos,
            hybird_ref_2d,
            sa_spatial_shapes,
            cross_spatial_shapes,
            queries_rebatch_grid,
            restore_bev_grid,
            reference_points_rebatch,
            bev_pillar_counts,
            prev_bev=prev_bev,
        )


@OBJECT_REGISTRY.register
class BEVFormerEncoderLayer(nn.Module):
    """Single BEVFormer encoder layer: TSA -> SCA -> FFN (each with LayerNorm).

    Args:
        selfattention: Temporal self-attention (``HorizonTemporalSelfAttention``).  # noqa: E501
        crossattention: Spatial cross-attention (``HorizonSpatialCrossAttention``).  # noqa: E501
        embed_dims: Transformer hidden dimension.
        dropout: Dropout rate inside FFN.
    """

    def __init__(
        self,
        selfattention: nn.Module,
        crossattention: nn.Module,
        embed_dims: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.sa = selfattention
        self.sa_norm = nn.LayerNorm(embed_dims)
        self.ca = crossattention
        self.ca_norm = nn.LayerNorm(embed_dims)
        self.ffn = FFN(embed_dims, dropout=dropout)
        self.ffn_norm = nn.LayerNorm(embed_dims)

    def _cross_attn_and_ffn(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_pos: Tensor,
        cross_spatial_shapes: Tensor,
        queries_rebatch_grid: Tensor,
        restore_bev_grid: Tensor,
        reference_points_rebatch: Tensor,
        bev_pillar_counts: Tensor,
    ) -> Tensor:
        """SCA samples image feats into BEV, then FFN refines each query."""
        query = self.ca(
            query=query,
            key=key,
            value=value,
            query_pos=query_pos,
            spatial_shapes=cross_spatial_shapes,
            queries_rebatch_grid=queries_rebatch_grid,
            restore_bev_grid=restore_bev_grid,
            reference_points_rebatch=reference_points_rebatch,
            bev_pillar_counts=bev_pillar_counts,
        )
        query = self.ca_norm(query)
        query = self.ffn(query, query)
        query = self.ffn_norm(query)
        return query

    def forward(
        self,
        query: Tensor,
        key: Tensor = None,
        value: Tensor = None,
        query_pos: Tensor = None,
        ref_2d: Tensor = None,
        sa_spatial_shapes: Tensor = None,
        cross_spatial_shapes: Tensor = None,
        prev_bev: Tensor = None,
        queries_rebatch_grid=None,
        restore_bev_grid=None,
        reference_points_rebatch=None,
        bev_pillar_counts=None,
    ):
        """Forward one encoder layer: TSA(prev+cur) -> SCA(img->bev) -> FFN."""
        # Step 1: temporal self-attention (value = stacked prev/cur BEV).
        query = self.sa(
            query=query,
            value=prev_bev,
            query_pos=query_pos,
            reference_points=ref_2d,
            spatial_shapes=sa_spatial_shapes,
        )
        query = self.sa_norm(query)
        # Step 2+3: spatial cross-attention + FFN.
        return self._cross_attn_and_ffn(
            query,
            key,
            value,
            query_pos,
            cross_spatial_shapes,
            queries_rebatch_grid,
            restore_bev_grid,
            reference_points_rebatch,
            bev_pillar_counts,
        )


@OBJECT_REGISTRY.register
class SingleFrameBEVFormerEncoder(BEVFormerEncoder):
    """BEVFormer encoder without temporal fusion (single frame)."""

    @fx_wrap()
    def get_encoder_inputs(
        self,
        mlvl_feats: List[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Get encoder layer inputs."""
        return self._flatten_mlvl_feats(mlvl_feats)

    def forward(
        self,
        bev_query: Tensor,
        mlvl_feats: List[Tensor],
        bev_pos: Tensor,
        ref_2d: Tensor,
        queries_rebatch_grid: Tensor,
        restore_bev_grid: Tensor,
        reference_points_rebatch: Tensor,
        bev_pillar_counts: Tensor,
    ) -> Tensor:
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        (
            feat_flatten,
            cross_spatial_shapes,
            sa_spatial_shapes,
        ) = self.get_encoder_inputs(mlvl_feats)

        return self._run_encoder_layers(
            bev_query,
            feat_flatten,
            bev_pos,
            ref_2d,
            sa_spatial_shapes,
            cross_spatial_shapes,
            queries_rebatch_grid,
            restore_bev_grid,
            reference_points_rebatch,
            bev_pillar_counts,
        )


@OBJECT_REGISTRY.register
class SingleFrameBEVFormerEncoderLayer(BEVFormerEncoderLayer):
    """BEVFormer encoder layer without temporal self-attention."""

    def forward(
        self,
        query: Tensor,
        key: Tensor = None,
        value: Tensor = None,
        query_pos: Tensor = None,
        ref_2d: Tensor = None,
        sa_spatial_shapes: Tensor = None,
        cross_spatial_shapes: Tensor = None,
        queries_rebatch_grid: Tensor = None,
        restore_bev_grid: Tensor = None,
        reference_points_rebatch: Tensor = None,
        bev_pillar_counts: Tensor = None,
    ):
        """Forward without temporal fusion: BEV self-attn -> SCA -> FFN."""
        query = self.sa(
            query,
            query,
            query_pos=query_pos,
            reference_points=ref_2d,
            spatial_shapes=sa_spatial_shapes,
        )
        query = self.sa_norm(query)
        return self._cross_attn_and_ffn(
            query,
            key,
            value,
            query_pos,
            cross_spatial_shapes,
            queries_rebatch_grid,
            restore_bev_grid,
            reference_points_rebatch,
            bev_pillar_counts,
        )


# Deprecated aliases for backward-compatible configs.
OBJECT_REGISTRY.alias("SingleBEVFormerEncoder")(SingleFrameBEVFormerEncoder)
OBJECT_REGISTRY.alias("SingleBEVFormerEncoderLayer")(
    SingleFrameBEVFormerEncoderLayer
)
