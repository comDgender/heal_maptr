# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
import logging
import math
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.model_helpers import fx_wrap
from horizon_plugin_pytorch.nn.quantized import FloatFunctional as FF
from horizon_plugin_pytorch.quantization import QuantStub
from torch import Tensor, nn

from heal.models.weight_init import constant_init, xavier_init

logger = logging.getLogger(__name__)

__all__ = [
    "HorizonMultiScaleDeformableAttention",
    "HorizonTemporalSelfAttention",
    "HorizonMultiScaleDeformableAttention3D",
    "HorizonSpatialCrossAttention",
    "HorizonMultiPointDeformableAttention",
]


class MultiScaleDeformableAttentionBase(nn.Module):
    """Base deformable attention: offsets + weights + multi-scale grid_sample.

    Predicts sampling offsets and attention weights from ``query``, samples
    ``value`` feature maps at deformed locations, then reduces across levels
    and sample points.

    Args:
        embed_dims: Hidden dimension (must divide ``num_heads``).
        num_heads: Number of attention heads.
        num_levels: Number of FPN levels in ``value``.
        num_points: Sample points per (head, level).
        grid_align_num: Pad query count for fixed ``grid_sample`` tile size.
        feats_size: Feature map (W, H) per level; used in deploy offset scaling.  # noqa: E501
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        grid_align_num: int = 8,
        feats_size: Sequence[Sequence[int]] = ((128, 128),),
    ) -> None:
        super().__init__()

        if embed_dims % num_heads != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_heads, "
                f"but got {embed_dims} and {num_heads}"
            )

        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points

        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2
        )
        self.attention_weights = nn.Linear(
            embed_dims, num_heads * num_levels * num_points
        )
        self.softmax = torch.nn.Softmax(dim=-1)

        self.quant_shape = QuantStub()
        self.norm_offset = FF()
        self.add_offset = FF()
        self.add1 = FF()
        self.mul1 = FF()
        self.stack_sampling = FF()
        self.mul_attention = FF()

        self.grid_align_num = grid_align_num
        self.reduce_sum = nn.Linear(
            self.grid_align_num * self.num_points * self.num_levels,
            self.grid_align_num,
            bias=False,
        )
        self.deploy = False
        self.feats_size = feats_size

    def init_weights(self) -> None:
        """Initialize for parameters of module."""
        constant_init(self.sampling_offsets, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.num_heads
        )
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0.0, bias=0.0)

        # init reduce_sum weight
        reduce_sum_weight = torch.zeros(
            self.reduce_sum.weight.size(), dtype=self.reduce_sum.weight.dtype
        )
        for i in range(self.grid_align_num):
            for j in range(self.num_levels):
                for k in range(self.num_points):
                    index = (
                        j * self.grid_align_num * self.num_points
                        + i * self.num_points
                        + k
                    )
                    reduce_sum_weight[i, index] = 1

        self.reduce_sum.weight = torch.nn.Parameter(
            reduce_sum_weight, requires_grad=False
        )

    @fx_wrap()
    def get_sampling_locations(
        self,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        sampling_offsets: Tensor,
    ) -> Tensor:
        """Get the sampling locations."""
        bs, num = reference_points.shape[:2]
        if not self.deploy:
            reference_points = reference_points.repeat(
                1, 1, self.num_heads, self.num_points
            ).reshape(
                bs, num, self.num_heads, self.num_levels * self.num_points * 2
            )
            offset_normalizer = self.quant_shape(
                (1 / spatial_shapes).repeat(1, self.num_points)
            ).flatten()
            sampling_offsets_tmp = self.norm_offset.mul(
                sampling_offsets, offset_normalizer[None, None, None, :]
            )
            sampling_locations = self.add_offset.add(
                reference_points, sampling_offsets_tmp
            )
            sampling_locations = self.add1.add_scalar(
                self.mul1.mul_scalar(sampling_locations, 2), -1
            )
        else:
            reference_points = self.add1.add_scalar(
                self.mul1.mul_scalar(reference_points, 2), -1
            )
            reference_points = reference_points.repeat(
                1, 1, self.num_heads, self.num_points
            ).reshape(
                bs, num, self.num_heads, self.num_levels * self.num_points * 2
            )
            sampling_locations = self.add_offset.add(
                reference_points, sampling_offsets
            )

        return sampling_locations

    @fx_wrap()
    def get_sampling_offsets(self, query: Tensor) -> Tensor:
        """Get the sampling offset."""
        bs, num_query = query.shape[:2]
        sampling_offsets = self.sampling_offsets(query).view(
            bs,
            num_query,
            self.num_heads,
            self.num_levels * self.num_points * 2,
        )
        return sampling_offsets

    @fx_wrap()
    def get_attention_weights(self, query: Tensor) -> Tensor:
        """Get the attention weight."""
        bs, num_query = query.shape[:2]
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_points * self.num_levels
        )
        attention_weights = self.softmax(attention_weights)
        attention_weights = (
            attention_weights.view(
                bs,
                -1,
                self.grid_align_num,
                self.num_heads,
                self.num_levels,
                self.num_points,
            )
            .permute(0, 3, 1, 4, 2, 5)
            .flatten(-3)
            .flatten(0, 1)
            .unsqueeze(1)
        )

        return attention_weights

    @fx_wrap()
    def multi_scale_deformable_attn_pytorch(
        self,
        value: Tensor,
        value_spatial_shapes: Tensor,
        sampling_locations: Tensor,
        attention_weights: Tensor,
    ) -> Tensor:
        """Get the multi scale deformable attention."""

        bs, _, num_heads, embed_dims = value.shape
        _, num_queries, num_heads, _ = sampling_locations.shape

        value_list = value.split(
            [int(H_ * W_) for W_, H_ in value_spatial_shapes], dim=1
        )
        sampling_grids = torch.split(
            sampling_locations, self.num_points * 2, dim=-1
        )
        sampling_value_list = []
        for level, (W_, H_) in enumerate(value_spatial_shapes):

            value_l_ = (
                value_list[level]
                .flatten(2)
                .transpose(1, 2)
                .reshape(
                    bs * num_heads, embed_dims, int(H_.item()), int(W_.item())
                )
            )
            sampling_grid_l_ = (
                sampling_grids[level].transpose(1, 2).flatten(0, 1)
            )
            sampling_grid_l_ = sampling_grid_l_.reshape(
                bs * self.num_heads,
                -1,
                self.grid_align_num * self.num_points,
                2,
            )
            sampling_value_l_ = F.grid_sample(
                value_l_,
                sampling_grid_l_,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            sampling_value_list.append(sampling_value_l_)

        sampling_value_list_outs = self.stack_sampling.cat(
            sampling_value_list, dim=-1
        )

        sampling_value_list_outs = self.mul_attention.mul(
            sampling_value_list_outs, attention_weights
        )

        output = self.reduce_sum(sampling_value_list_outs)
        output = output.view(bs, num_heads * embed_dims, num_queries)
        return output

    def forward(
        self,
        query: Tensor,
        value: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        **kwargs,
    ) -> Tensor:
        """Forward for base MultiScaleDeformableAttention."""

        sampling_offsets = self.get_sampling_offsets(query)
        sampling_locations = self.get_sampling_locations(
            reference_points, spatial_shapes, sampling_offsets
        )
        attention_weights = self.get_attention_weights(query)
        output = self.multi_scale_deformable_attn_pytorch(
            value, spatial_shapes, sampling_locations, attention_weights
        )
        return output

    def fix_weight_qscale(self) -> None:
        """Fix the qscale of conv weight when calibration or qat stage."""

        self.reduce_sum.weight_fake_quant.disable_observer()
        weight_scale_reduce_sum = torch.ones(
            self.reduce_sum.weight.shape[0],
            device=self.reduce_sum.weight.device,
        )
        weight_scale_reduce_sum[...] = (1.0) / 128.0
        self.reduce_sum.weight_fake_quant.set_qparams(weight_scale_reduce_sum)

    def switch_to_deploy(self):
        """Switch module to deploy."""
        old_offset_weight = self.sampling_offsets.weight.data.clone()
        old_offset_bias = self.sampling_offsets.bias.data.clone()

        old_offset_weight = old_offset_weight.reshape(
            self.num_heads, self.num_levels, self.num_points, 2, -1
        )
        old_offset_bias = old_offset_bias.reshape(
            self.num_heads, self.num_levels, self.num_points, 2
        )
        for idx, feat_size in enumerate(self.feats_size):
            w, h = feat_size
            old_offset_weight[:, idx, :, 0, :] = (
                old_offset_weight[:, idx, :, 0, :] * 2 / w
            )
            old_offset_weight[:, idx, :, 1, :] = (
                old_offset_weight[:, idx, :, 1, :] * 2 / h
            )
            old_offset_bias[:, idx, :, 0] = (
                old_offset_bias[:, idx, :, 0] * 2 / w
            )
            old_offset_bias[:, idx, :, 1] = (
                old_offset_bias[:, idx, :, 1] * 2 / h
            )
        new_offset_weight = old_offset_weight.flatten(0, -2)
        new_offset_bias = old_offset_bias
        self.sampling_offsets.weight.data = new_offset_weight
        self.sampling_offsets.bias.data = new_offset_bias.flatten()
        self.deploy = True


@OBJECT_REGISTRY.register
class HorizonMultiScaleDeformableAttention(MultiScaleDeformableAttentionBase):
    """The basic structure of HorizonMultiScaleDeformableAttention.

    Args:
        embed_dims: The embedding dimension of Attention.
        num_heads: Parallel attention heads.
        num_levels: The num of featuremap.
        num_points: The num points for each head sample.
        dropout: Probability of an element to be zeroed.
        batch_first: Whether the first dim is batch.
        grid_align_num: The align num for grid, align the grid shape of \
            gridsample operator to \
            [bs * numhead, -1, grid_align_num * numpoints, 2].
        feats_size: The Size of featmaps.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        dropout: float = 0.1,
        batch_first: bool = False,
        grid_align_num: int = 8,
        feats_size: Sequence[Sequence[int]] = ((128, 128),),
    ) -> None:
        super().__init__(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            grid_align_num=grid_align_num,
            feats_size=feats_size,
        )
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.add_pos = FF()
        self.add_res = FF()
        self.init_weights()

    def init_weights(self):
        """Initialize for parameters of module."""
        super().init_weights()
        xavier_init(self.value_proj, distribution="uniform", bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    @fx_wrap()
    def forward(
        self,
        query: Tensor,
        value: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        query_pos: Tensor = None,
        identity: Tensor = None,
        **kwargs,
    ) -> Tensor:
        """Forward for HorizonMultiScaleDeformableAttention."""

        if identity is None:
            identity = query
        if query_pos is not None:
            query = self.add_pos.add(query, query_pos)

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_value, _ = value.shape
        value = self.value_proj(value)
        value = value.reshape(bs, num_value, self.num_heads, -1)

        output = (
            super()
            .forward(query, value, reference_points, spatial_shapes, **kwargs)
            .transpose(1, 2)
            .contiguous()
        )
        output = self.output_proj(output)
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        output = self.add_res.add(self.dropout(output), identity)

        return output


@OBJECT_REGISTRY.register
class HorizonTemporalSelfAttention(MultiScaleDeformableAttentionBase):
    """Temporal Self-Attention (TSA): fuse prev and current BEV queries.

    Concatenates previous-frame BEV value tokens with current queries along
    the feature dim, runs deformable attention over a 2-queue BEV (prev/cur),
    then mean-pools back to ``len_bev`` queries via ``query_reduce_mean``.

    Args:
        embed_dims: Hidden dimension.
        num_heads: Attention heads.
        num_levels: BEV feature levels (typically 1).
        num_points: Deformable sample points per head/level.
        grid_align_num: Query padding for ``grid_sample``.
        num_bev_queue: Temporal queue size (2 = prev + current).
        reduce_align_num: Output align dim before mean pool.
        dropout: Dropout on output projection.
        feats_size: BEV (W, H) for deploy offset scaling.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        grid_align_num: int = 8,
        num_bev_queue: int = 2,
        reduce_align_num: int = 1,
        dropout: float = 0.1,
        feats_size: Sequence[Sequence[int]] = ((128, 128),),
    ) -> None:

        super().__init__(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            grid_align_num=grid_align_num,
            feats_size=feats_size,
        )
        self.dropout = nn.Dropout(dropout)
        self.num_bev_queue = num_bev_queue
        self.reduce_align_num = reduce_align_num

        self.sampling_offsets = nn.Linear(
            embed_dims * num_bev_queue,
            num_bev_queue * num_heads * num_levels * num_points * 2,
        )
        self.attention_weights = nn.Linear(
            embed_dims * num_bev_queue,
            num_bev_queue * num_heads * num_levels * num_points,
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.query_reduce_mean = nn.Conv2d(
            self.num_bev_queue * self.reduce_align_num,
            self.reduce_align_num,
            1,
            bias=False,
        )
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        # Functional wrappers kept for FX / quant graph connectivity.
        self.add_pos = FF()
        self.add_res = FF()
        self.cat_query_value = FF()

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize offset / attention / reduce-mean parameters."""
        super().init_weights()
        constant_init(self.sampling_offsets, 0.0)

        # Per-head unit directions on the circle, L-inf normalized, then
        # scaled by sample-point index (1..num_points). Broadcast over
        # (levels * bev_queue) so temporal slots share the same pattern.
        n_heads = self.num_heads
        angles = (
            torch.arange(n_heads, dtype=torch.float32) / float(n_heads)
        ) * (2.0 * math.pi)
        direction = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
        linf_norm = direction.abs().amax(dim=-1, keepdim=True)
        direction = direction / linf_norm

        n_level_slots = self.num_levels * self.num_bev_queue
        offset_bias = direction.view(n_heads, 1, 1, 2).expand(
            n_heads, n_level_slots, self.num_points, 2
        )
        point_scale = torch.arange(
            1, self.num_points + 1, dtype=torch.float32
        ).view(1, 1, -1, 1)
        offset_bias = (offset_bias * point_scale).contiguous()
        # Assign (not copy_): super().init_weights() may have replaced bias
        # with a size that omits num_bev_queue.
        self.sampling_offsets.bias.data = offset_bias.reshape(-1)

        constant_init(self.attention_weights, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)
        xavier_init(self.value_proj, distribution="uniform", bias=0.0)

        # Fixed 1x1 conv that averages matching align channels across the
        # BEV queue: out[i] = mean_j in[j * reduce_align_num + i].
        # Weight is Conv2d-shaped [C_out, C_in, 1, 1].
        avg_weight = torch.zeros(
            self.query_reduce_mean.weight.size(),
            dtype=self.query_reduce_mean.weight.dtype,
        )
        for align_idx in range(self.reduce_align_num):
            for queue_idx in range(self.num_bev_queue):
                col = queue_idx * self.reduce_align_num + align_idx
                avg_weight[align_idx, col] = 1.0 / self.num_bev_queue
        self.query_reduce_mean.weight = torch.nn.Parameter(
            avg_weight, requires_grad=False
        )

    @fx_wrap()
    def get_sampling_offsets(self, query: Tensor) -> Tensor:
        """Get the sampling offset."""
        bs, num_query = query.shape[:2]
        sampling_offsets = (
            self.sampling_offsets(query)
            .view(
                bs,
                num_query,
                self.num_heads,
                self.num_bev_queue,
                self.num_levels,
                self.num_points,
                2,
            )
            .permute(0, 3, 1, 2, 4, 5, 6)
            .flatten(0, 1)
            .flatten(-3)
        )
        return sampling_offsets

    @fx_wrap()
    def get_attention_weights(self, query: Tensor) -> Tensor:
        """Get the attention weight."""
        bs, num_query = query.shape[:2]
        attention_weights = self.attention_weights(query).view(
            bs,
            num_query,
            self.num_heads,
            self.num_bev_queue,
            self.num_points * self.num_levels,
        )
        attention_weights = self.softmax(attention_weights)
        attention_weights = attention_weights.view(
            bs,
            -1,
            self.grid_align_num,
            self.num_heads,
            self.num_bev_queue,
            self.num_levels,
            self.num_points,
        )
        attention_weights = (
            attention_weights.permute(0, 4, 3, 1, 2, 5, 6)
            .flatten(0, 2)
            .flatten(-3)
            .unsqueeze(1)
        )
        return attention_weights

    @fx_wrap()
    def forward(
        self,
        query: Tensor,
        value: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        query_pos: Tensor = None,
        identity: Tensor = None,
        **kwargs,
    ) -> Tensor:
        """Forward for HorizonTemporalSelfAttention."""

        if identity is None:
            identity = query
        if query_pos is not None:
            query = self.add_pos.add(query, query_pos)

        bs, num_query, _ = query.shape
        _, num_value, _ = value.shape

        # value: (bs, 2*len_bev, C) -> split prev queue and reshape for attn.
        value = value.reshape(bs, self.num_bev_queue, num_value, -1)
        prev_value = torch.split(value, (self.num_bev_queue - 1), dim=1)
        prev_value = prev_value[0].flatten(0, 1)
        # Condition offsets/weights on [prev_value || cur_query].
        query = self.cat_query_value.cat([prev_value, query], -1)

        value = self.value_proj(value)
        value = value.reshape(
            bs * self.num_bev_queue, num_value, self.num_heads, -1
        )

        output = super().forward(
            query, value, reference_points, spatial_shapes, **kwargs
        )
        output = output.reshape(
            bs, self.num_bev_queue * self.reduce_align_num, -1, num_query
        )
        output = self.query_reduce_mean(output).flatten(1, 2).permute(0, 2, 1)
        output = self.output_proj(output)
        output = self.add_res.add(self.dropout(output), identity)
        return output

    def fix_weight_qscale(self) -> None:
        """Fix the qscale of conv weight when calibration or qat stage."""
        super().fix_weight_qscale()
        self.query_reduce_mean.weight_fake_quant.disable_observer()
        weight_scale_query_reduce_mean = torch.ones(
            self.query_reduce_mean.weight.shape[0],
            device=self.query_reduce_mean.weight.device,
        )
        weight_scale_query_reduce_mean[...] = (1.0 / self.num_bev_queue) / 128
        self.query_reduce_mean.weight_fake_quant.set_qparams(
            weight_scale_query_reduce_mean
        )

    def switch_to_deploy(self):
        """Switch module to deploy."""
        old_offset_weight = self.sampling_offsets.weight.data.clone()
        old_offset_bias = self.sampling_offsets.bias.data.clone()

        old_offset_weight = old_offset_weight.reshape(
            self.num_heads,
            self.num_bev_queue,
            self.num_levels,
            self.num_points,
            2,
            -1,
        )
        old_offset_bias = old_offset_bias.reshape(
            self.num_heads,
            self.num_bev_queue,
            self.num_levels,
            self.num_points,
            2,
        )
        for idx, feat_size in enumerate(self.feats_size):
            w, h = feat_size
            old_offset_weight[:, :, idx, :, 0, :] = (
                old_offset_weight[:, :, idx, :, 0, :] * 2 / w
            )
            old_offset_weight[:, :, idx, :, 1, :] = (
                old_offset_weight[:, :, idx, :, 1, :] * 2 / h
            )
            old_offset_bias[:, :, idx, :, 0] = (
                old_offset_bias[:, :, idx, :, 0] * 2 / w
            )
            old_offset_bias[:, :, idx, :, 1] = (
                old_offset_bias[:, :, idx, :, 1] * 2 / h
            )
        new_offset_weight = old_offset_weight.flatten(0, -2)
        new_offset_bias = old_offset_bias
        self.sampling_offsets.weight.data = new_offset_weight
        self.sampling_offsets.bias.data = new_offset_bias.flatten()
        self.deploy = True


@OBJECT_REGISTRY.register
class HorizonMultiScaleDeformableAttention3D(
    MultiScaleDeformableAttentionBase
):
    """3D deformable attention for SCA (Z anchors per BEV pillar).

    ``reference_points`` encodes multiple image-plane refs per query (one per
    pillar Z anchor). Offsets are broadcast across Z anchors before sampling.

    Args:
        embed_dims: Hidden dimension.
        num_heads: Attention heads.
        num_levels: FPN levels in image ``value``.
        num_points: Sample points per (head, level, Z anchor group).
        grid_align_num: Query padding for ``grid_sample``.
        feats_size: Image feature (W, H) per level.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        grid_align_num: int = 8,
        feats_size: Sequence[Sequence[int]] = ((128, 128),),
    ) -> None:
        super().__init__(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            grid_align_num=grid_align_num,
            feats_size=feats_size,
        )
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        """Initialize for parameters of module."""
        super().init_weights()
        xavier_init(self.value_proj, distribution="uniform", bias=0.0)

    @fx_wrap()
    def get_sampling_locations(
        self,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        sampling_offsets: Tensor,
    ) -> Tensor:
        """Get the sampling locations."""
        num_Z_anchors = reference_points.shape[-1] // 2
        if not self.deploy:
            reference_points = reference_points.repeat(
                1, 1, self.num_heads, self.num_points // num_Z_anchors
            )
            offset_normalizer = self.quant_shape(
                (1 / spatial_shapes).repeat(1, self.num_points)
            ).flatten()
            sampling_offsets_tmp = self.norm_offset.mul(
                sampling_offsets, offset_normalizer[None, None, None, :]
            )
            sampling_locations = self.add_offset.add(
                reference_points, sampling_offsets_tmp
            )
            sampling_locations = self.add1.add_scalar(
                self.mul1.mul_scalar(sampling_locations, 2), -1
            )
        else:
            reference_points = self.add1.add_scalar(
                self.mul1.mul_scalar(reference_points, 2), -1
            )
            reference_points = reference_points.repeat(
                1, 1, self.num_heads, self.num_points // num_Z_anchors
            )
            sampling_locations = self.add_offset.add(
                reference_points, sampling_offsets
            )
        return sampling_locations

    @fx_wrap()
    def forward(
        self,
        query: Tensor,
        value: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        **kwargs,
    ) -> Tensor:
        """Forward for HorizonMultiScaleDeformableAttention3D."""

        if value is None:
            value = query

        bs, num_value, _ = value.shape

        value = self.value_proj(value)
        value = value.reshape(bs, num_value, self.num_heads, -1)
        output = super().forward(
            query, value, reference_points, spatial_shapes, **kwargs
        )
        return output


@OBJECT_REGISTRY.register
class HorizonSpatialCrossAttention(nn.Module):
    """Spatial Cross-Attention (SCA): multi-view image -> BEV aggregation.

    Rebatch visible BEV queries into per-camera dense grids, apply 3D
    deformable attention against that camera's image tokens, scatter results
    back to BEV pillars, and average when multiple cameras see the same pillar.

    Args:
        bev_h, bev_w: BEV grid size.
        deformable_attention: Inner 3D deformable module (typically
            ``HorizonMultiScaleDeformableAttention3D``).
        embed_dims: Hidden dimension.
        num_cams: Number of cameras.
        dropout: Dropout on output projection.
        max_camoverlap_num: Max stacked camera outputs per BEV pillar.
    """

    def __init__(
        self,
        bev_h: int,
        bev_w: int,
        deformable_attention: nn.Module,
        embed_dims: int = 256,
        num_cams: int = 6,
        dropout: int = 0.1,
        max_camoverlap_num: int = 2,
    ) -> None:
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = deformable_attention
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.max_camoverlap_num = max_camoverlap_num

        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.add_res = FF()
        self.add_query_pos = FF()
        self.mul_pillarweight = FF()
        self.sum_views = FF()

        self.query_reduce_sum = nn.Conv2d(
            self.embed_dims * self.max_camoverlap_num,
            self.embed_dims,
            1,
            bias=False,
            groups=self.embed_dims,
        )

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize for parameters of module."""

        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

        query_reduce_sum_weight = torch.ones(
            self.query_reduce_sum.weight.size(),
            dtype=self.query_reduce_sum.weight.dtype,
        )
        self.query_reduce_sum.weight = torch.nn.Parameter(
            query_reduce_sum_weight, requires_grad=False
        )

    def rebatch_attention_inputs(
        self,
        query: Tensor,
        queries_rebatch_grid: Tensor,
        reference_points_rebatch: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Pack BEV queries visible to each camera into a dense rebatch layout."""  # noqa: E501
        bs = query.shape[0]
        queries_rebatch = (
            query.unsqueeze(1)
            .repeat(1, self.num_cams, 1, 1)
            .reshape(
                bs * self.num_cams, self.bev_h, self.bev_w, self.embed_dims
            )
            .permute(0, 3, 1, 2)
        )
        queries_rebatch = F.grid_sample(
            queries_rebatch,
            queries_rebatch_grid,
            mode="nearest",
            align_corners=True,
        )
        queries_rebatch = queries_rebatch.flatten(-2).permute(0, 2, 1)
        reference_points_rebatch = reference_points_rebatch.flatten(
            -2
        ).unsqueeze(-2)
        return queries_rebatch, reference_points_rebatch

    def restore_outputs(
        self,
        restore_bev_grid: Tensor,
        queries_out: Tensor,
        counts: Tensor,
        bs: int,
        queries_rebatch_grid: Tensor,
    ):
        """Scatter per-camera SCA outputs back to BEV grid and average overlap."""  # noqa: E501
        queries_out = queries_out.reshape(
            bs, self.num_cams, self.embed_dims, -1
        )
        queries_out = queries_out.permute(0, 2, 1, 3)

        queries_out = queries_out.reshape(
            bs,
            self.embed_dims,
            self.num_cams * queries_rebatch_grid.shape[1],
            queries_rebatch_grid.shape[2],
        )
        bev_queries = F.grid_sample(
            queries_out, restore_bev_grid, mode="nearest", align_corners=True
        )
        bev_queries = bev_queries.reshape(bs, -1, self.bev_h, self.bev_w)
        slots = self.query_reduce_sum(bev_queries).flatten(-2).permute(0, 2, 1)

        slots = self.mul_pillarweight.mul(slots, counts)

        return slots

    @fx_wrap()
    def forward(
        self,
        query: Tensor,
        value: Tensor,
        spatial_shapes: Tensor,
        queries_rebatch_grid: Tensor,
        restore_bev_grid: Tensor,
        reference_points_rebatch: Tensor,
        bev_pillar_counts: Tensor,
        query_pos: Tensor = None,
        **kwargs,
    ) -> Tensor:
        """Forward SCA: rebatch -> deformable cross-attn -> restore -> residual."""  # noqa: E501
        inp_residual = query
        if query_pos is not None:
            query = self.add_query_pos.add(query, query_pos)

        bs, _, _ = query.size()
        _, l, _, _ = value.shape
        # Image value: (num_cam, bs, len, C) -> (bs*num_cam, len, C).
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims
        )

        (
            queries_rebatch_horizon,
            reference_points_rebatch,
        ) = self.rebatch_attention_inputs(
            query, queries_rebatch_grid, reference_points_rebatch
        )

        queries_out = self.deformable_attention(
            query=queries_rebatch_horizon,
            value=value,
            reference_points=reference_points_rebatch,
            spatial_shapes=spatial_shapes,
        )

        slots = self.restore_outputs(
            restore_bev_grid,
            queries_out,
            bev_pillar_counts,
            bs,
            queries_rebatch_grid,
        )

        slots = self.output_proj(slots)
        queries = self.add_res.add(self.dropout(slots), inp_residual)
        return queries

    def fix_weight_qscale(self) -> None:
        """Fix the qscale of conv weight when calibration or qat stage."""

        self.query_reduce_sum.weight_fake_quant.disable_observer()

        weight_scale = torch.ones(
            self.query_reduce_sum.weight.shape[0],
            device=self.query_reduce_sum.weight.device,
        )

        weight_scale[...] = 1.0 / 128.0
        self.query_reduce_sum.weight_fake_quant.set_qparams(weight_scale)
        if hasattr(self.deformable_attention, "fix_weight_qscale"):
            self.deformable_attention.fix_weight_qscale()


@OBJECT_REGISTRY.register
class HorizonMultiPointDeformableAttention(MultiScaleDeformableAttentionBase):
    """The basic structure of HorizonMultiPointDeformableAttention.

    Args:
        embed_dims: The embedding dimension of Attention.
        num_heads: Parallel attention heads.
        num_levels: The num of featuremap.
        num_points: The num points for each head sample.
        dropout: Probability of an element to be zeroed.
        batch_first: Whether the first dim is batch.
        grid_align_num: The align num for grid, align the grid shape of \
            gridsample operator to \
            [bs * numhead, -1, grid_align_num * numpoints, 2].
        feats_size: The Size of featmaps.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        dropout: float = 0.1,
        batch_first: bool = False,
        grid_align_num: int = 8,
        feats_size: Sequence[Sequence[int]] = ((128, 128),),
    ) -> None:
        super().__init__(
            embed_dims=embed_dims,
            num_heads=num_heads,
            num_levels=num_levels,
            num_points=num_points,
            grid_align_num=grid_align_num,
            feats_size=feats_size,
        )
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.add_pos = FF()
        self.add_res = FF()
        self.init_weights()

    def init_weights(self):
        """Initialize for parameters of module."""
        super().init_weights()
        xavier_init(self.value_proj, distribution="uniform", bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    @fx_wrap()
    def get_sampling_locations(
        self,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        sampling_offsets: Tensor,
    ) -> Tensor:
        """Get the sampling locations."""
        bs, num = reference_points.shape[:2]
        if not self.deploy:
            reference_points = reference_points.repeat(
                1, 1, self.num_heads, 1, 1
            ).reshape(
                bs, num, self.num_heads, self.num_levels * self.num_points * 2
            )
            offset_normalizer = self.quant_shape(
                (1 / spatial_shapes).repeat(1, self.num_points)
            ).flatten()
            sampling_offsets_tmp = self.norm_offset.mul(
                sampling_offsets, offset_normalizer[None, None, None, :]
            )
            sampling_locations = self.add_offset.add(
                reference_points, sampling_offsets_tmp
            )
            sampling_locations = self.add1.add_scalar(
                self.mul1.mul_scalar(sampling_locations, 2), -1
            )
        else:
            reference_points = self.add1.add_scalar(
                self.mul1.mul_scalar(reference_points, 2), -1
            )
            reference_points = reference_points.repeat(
                1, 1, self.num_heads, 1, 1
            ).reshape(
                bs, num, self.num_heads, self.num_levels * self.num_points * 2
            )
            sampling_locations = self.add_offset.add(
                reference_points, sampling_offsets
            )

        return sampling_locations

    @fx_wrap()
    def forward(
        self,
        query: Tensor,
        value: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        query_pos: Tensor = None,
        identity: Tensor = None,
        **kwargs,
    ) -> Tensor:
        """Forward for HorizonMultiPointDeformableAttention."""

        if identity is None:
            identity = query
        if query_pos is not None:
            query = self.add_pos.add(query, query_pos)

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_value, _ = value.shape
        value = self.value_proj(value)
        value = value.reshape(bs, num_value, self.num_heads, -1)

        output = (
            super()
            .forward(query, value, reference_points, spatial_shapes, **kwargs)
            .transpose(1, 2)
            .contiguous()
        )
        output = self.output_proj(output)
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        output = self.add_res.add(self.dropout(output), identity)

        return output
