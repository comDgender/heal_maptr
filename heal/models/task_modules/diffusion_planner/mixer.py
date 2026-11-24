# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import torch.nn as nn
from timm.models.layers import Mlp


class MixerBlock(nn.Module):
    """MLP-Mixer block with token-mixing and channel-mixing MLPs."""

    def __init__(self, tokens_mlp_dim, channels_mlp_dim, drop_path_rate):
        """Initialize MixerBlock.

        Args:
            tokens_mlp_dim: Hidden dim of the token-mixing MLP.
            channels_mlp_dim: Hidden dim of the channel-mixing MLP.
            drop_path_rate: Dropout rate applied inside both MLPs.
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(channels_mlp_dim)
        self.channels_mlp = Mlp(
            in_features=channels_mlp_dim,
            hidden_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )
        self.norm2 = nn.LayerNorm(channels_mlp_dim)
        self.tokens_mlp = Mlp(
            in_features=tokens_mlp_dim,
            hidden_features=tokens_mlp_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

    def forward(self, x):
        """Apply token-mixing then channel-mixing with residual connections.

        Args:
            x: Input tensor of shape (B, T, C).

        Returns:
            Output tensor with the same shape as input.
        """
        y = self.norm1(x)
        y = y.permute(0, 2, 1)
        y = self.tokens_mlp(y)
        y = y.permute(0, 2, 1)
        x = x + y
        y = self.norm2(x)
        return x + self.channels_mlp(y)
