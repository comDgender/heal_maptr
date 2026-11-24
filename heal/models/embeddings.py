# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import math

import torch
import torch.nn as nn
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.model_helpers import fx_wrap

__all__ = ["SinePositionalEncoding3D", "PositionEmbeddingSine"]


@OBJECT_REGISTRY.register
class SinePositionalEncoding3D(nn.Module):
    """Position encoding with sine and cosine functions for 3D space.

    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        num_feats: The feature dimension for each position along each axis.
            The final returned dimension for each axis is 2 * num_feats.
        temperature: The temperature used for scaling the position embedding.
        normalize: Whether to normalize the position embedding.
        scale: A scale factor that scales the position embedding.
            Only used when ``normalize`` is True.
        eps: A value added to the denominator for numerical stability.
        offset: Offset added to embed when doing normalization.
    """

    def __init__(
        self,
        num_feats,
        temperature=10000,
        normalize=False,
        scale=2 * math.pi,
        eps=1e-6,
        offset=0.0,
    ):
        super(SinePositionalEncoding3D, self).__init__()
        if normalize:
            assert isinstance(scale, (float, int)), (
                "when normalize is set,"
                "scale should be provided and in float or int type, "
                f"found {type(scale)}"
            )
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask):
        """Forward function for `SinePositionalEncoding3D`.

        Args:
            mask: ByteTensor mask. Non-zero values represent
                ignored positions, zero values mean valid.
                Shape [bs, n, h, w].

        Returns:
            pos: Returned position embedding with shape
                [bs, n, num_feats*3, h, w].
        """
        mask = mask.to(torch.int)
        not_mask = 1 - mask
        n_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            n_embed = (
                (n_embed + self.offset)
                / (n_embed[:, -1:, :, :] + self.eps)
                * self.scale
            )
            y_embed = (
                (y_embed + self.offset)
                / (y_embed[:, :, -1:, :] + self.eps)
                * self.scale
            )
            x_embed = (
                (x_embed + self.offset)
                / (x_embed[:, :, :, -1:] + self.eps)
                * self.scale
            )
        dim_t = torch.arange(
            self.num_feats, dtype=torch.float32, device=mask.device
        )
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)
        pos_n = n_embed[:, :, :, :, None] / dim_t
        pos_x = x_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t
        B, N, H, W = mask.size()
        pos_n = torch.stack(
            (pos_n[:, :, :, :, 0::2].sin(), pos_n[:, :, :, :, 1::2].cos()),
            dim=4,
        ).view(B, N, H, W, -1)
        pos_x = torch.stack(
            (pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()),
            dim=4,
        ).view(B, N, H, W, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()),
            dim=4,
        ).view(B, N, H, W, -1)
        pos = (
            torch.cat((pos_n, pos_y, pos_x), dim=4)
            .permute(0, 1, 4, 2, 3)
            .contiguous()
        )
        return pos


@OBJECT_REGISTRY.register
class PositionEmbeddingSine(nn.Module):
    """Position encoding with sine and cosine functions.

    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        num_pos_feats: The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature: The temperature used for scaling
            the position embedding. Default 10000.
        normalize: Whether to normalize the position
            embedding. Default False.
        scale: A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Default 2*pi.
    """

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: int = 10000,
        normalize: bool = False,
        scale: float = None,
        offset: float = 0.0,
    ):
        super(PositionEmbeddingSine, self).__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.offset = offset
        dim_t = torch.arange(
            self.num_pos_feats,
            dtype=torch.float32,
        )
        self.dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    @fx_wrap()
    def forward(self, mask):
        """Encode a ``(bs, h, w)`` mask into sine position features.

        Valid locations are ``False`` in ``mask``. Returns a tensor of shape
        ``(bs, 2 * num_pos_feats, h, w)`` with y/x channels concatenated.
        """
        valid = ~mask
        # Cumulative counts along H / W act as integer coordinates.
        coord_y = valid.cumsum(dim=1, dtype=torch.float32)
        coord_x = valid.cumsum(dim=2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            coord_y = (
                (coord_y + self.offset) / (coord_y[:, -1:, :] + eps)
            ) * self.scale
            coord_x = (
                (coord_x + self.offset) / (coord_x[:, :, -1:] + eps)
            ) * self.scale

        freqs = self.dim_t.to(device=mask.device)
        # (B, H, W, 1) / (num_pos_feats,) -> (B, H, W, num_pos_feats)
        ang_x = coord_x.unsqueeze(-1) / freqs
        ang_y = coord_y.unsqueeze(-1) / freqs

        def _sin_cos(angles: torch.Tensor) -> torch.Tensor:
            # Interleave sin(even) and cos(odd) along the feature axis.
            return torch.stack(
                (angles[..., 0::2].sin(), angles[..., 1::2].cos()),
                dim=-1,
            ).flatten(start_dim=-2)

        feat_x = _sin_cos(ang_x)
        feat_y = _sin_cos(ang_y)
        # Channel order matches DETR: y features then x features.
        return torch.cat((feat_y, feat_x), dim=-1).permute(0, 3, 1, 2)
