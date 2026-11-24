# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import torch
import torch.nn as nn
from hat_engine.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class LearnedPositionalEncoding(nn.Module):
    """Learnable 2D row/column position embeddings.

    Each spatial location ``(y, x)`` is represented by concatenating a
    learned row embedding and a learned column embedding, yielding a
    ``2 * num_feats``-channel map.

    Args:
        num_feats: Channels per axis (row or column).
        row_num_embed: Size of the row embedding table.
        col_num_embed: Size of the column embedding table.
    """

    def __init__(
        self,
        num_feats: int,
        row_num_embed: int = 50,
        col_num_embed: int = 50,
    ):
        super().__init__()
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """Build a ``(bs, 2*num_feats, h, w)`` position map from ``mask``.

        ``mask`` has shape ``(bs, h, w)``; only its spatial size and batch
        size are used (valid vs ignored bits do not affect the embedding).
        """
        batch, height, width = mask.shape
        device = mask.device

        col_ids = torch.arange(width, device=device)
        row_ids = torch.arange(height, device=device)
        # (W, C) / (H, C) -> broadcast to (H, W, C) then concat on channel.
        col_feat = (
            self.col_embed(col_ids).unsqueeze(0).expand(height, width, -1)
        )
        row_feat = (
            self.row_embed(row_ids).unsqueeze(1).expand(height, width, -1)
        )
        grid = torch.cat((col_feat, row_feat), dim=-1)
        # (H, W, 2C) -> (1, 2C, H, W) -> (B, 2C, H, W)
        pos = (
            grid.permute(2, 0, 1)
            .unsqueeze(0)
            .expand(batch, -1, -1, -1)
            .contiguous()
        )
        return pos
