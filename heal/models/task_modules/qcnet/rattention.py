# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import torch
import torch.nn as nn
from horizon_plugin_pytorch.quantization import QuantStub

from .utils import weight_init

__all__ = ["RAttentionLayer"]


class RAttentionLayer(nn.Module):
    """
    RAttentionLayer implements a relational attention mechanism.

    Args:
        hidden_dim: Dimensionality of the hidden layers.
        num_heads: Number of attention heads.
        head_dim: Dimensionality of each attention head.
        dropout: Dropout rate.
        bipartit: Whether the attention is bipartite.
        has_pos_emb: Whether to use positional embeddings.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        bipartite: bool,
        has_pos_emb: bool,
    ) -> None:
        super(RAttentionLayer, self).__init__()
        self._init_attention_meta(num_heads, head_dim, bipartite, has_pos_emb)
        fused_width = num_heads * head_dim
        self._install_qkv_projections(hidden_dim, fused_width, has_pos_emb)
        self._install_gating_head(hidden_dim, fused_width)
        self.attn_drop = nn.Dropout(dropout)
        self.ff_mlp = self._build_feedforward_mlp(hidden_dim, dropout)
        self._install_layer_norms(hidden_dim, bipartite, has_pos_emb)
        self.quant = QuantStub()
        self.apply(weight_init)

    def _init_attention_meta(
        self, num_heads: int, head_dim: int, bipartite: bool, has_pos_emb: bool
    ) -> None:
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.bipartite = bipartite
        self.has_pos_emb = has_pos_emb
        self.scale = head_dim**-0.5

    def _install_qkv_projections(
        self, hidden_dim: int, fused_width: int, use_rel_pos: bool
    ) -> None:
        self.to_q = nn.Linear(hidden_dim, fused_width)
        self.to_k = nn.Linear(hidden_dim, fused_width, bias=False)
        self.to_v = nn.Linear(hidden_dim, fused_width)
        if use_rel_pos:
            self.to_k_r = nn.Linear(hidden_dim, fused_width, bias=False)
            self.to_v_r = nn.Linear(hidden_dim, fused_width)

    def _install_gating_head(self, hidden_dim: int, fused_width: int) -> None:
        self.to_s = nn.Linear(hidden_dim, fused_width)
        self.to_g = nn.Linear(fused_width + hidden_dim, fused_width)
        self.to_out = nn.Linear(fused_width, hidden_dim)

    @staticmethod
    def _build_feedforward_mlp(
        hidden_dim: int, drop_rate: float
    ) -> nn.Sequential:
        expand_dim = hidden_dim * 4
        return nn.Sequential(
            nn.Linear(hidden_dim, expand_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(drop_rate),
            nn.Linear(expand_dim, hidden_dim),
        )

    def _install_layer_norms(
        self, hidden_dim: int, is_bipartite: bool, use_rel_pos: bool
    ) -> None:
        self.attn_prenorm_x_src = nn.LayerNorm(hidden_dim)
        if is_bipartite:
            self.attn_prenorm_x_dst = nn.LayerNorm(hidden_dim)
        if use_rel_pos:
            self.attn_prenorm_r = nn.LayerNorm(hidden_dim)
        self.ff_prenorm = nn.LayerNorm(hidden_dim)

    def _attn_block(
        self,
        x_src,
        x_dst,
        r,
        mask=None,
        extra_1dim=False,
    ):
        B = x_src.shape[0]

        if extra_1dim:
            lenq, lenk = x_dst.shape[2], x_src.shape[2]
            kdim1 = qdim = 1
            ex_dim = x_dst.shape[1]
            q = self.to_q(x_dst).view(
                B, ex_dim, lenq, qdim, self.num_heads, self.head_dim
            )  # [B，pl, 1, h, d]
            k = self.to_k(x_src).view(
                B, ex_dim, kdim1, lenk, self.num_heads, self.head_dim
            )  # [B，pl, pt, h, d]
            v = self.to_v(x_src).view(
                B, ex_dim, kdim1, lenk, self.num_heads, self.head_dim
            )  # [B，pl, pt, h, d]
            if self.has_pos_emb:
                rk = self.to_k_r(r).view(
                    B, ex_dim, lenq, lenk, self.num_heads, self.head_dim
                )
                rv = self.to_v_r(r).view(
                    B, ex_dim, lenq, lenk, self.num_heads, self.head_dim
                )
        else:
            if x_src.dim() == 4 and x_dst.dim() == 3:
                lenq, lenk = x_dst.shape[1], x_src.shape[2]
                kdim1 = lenq
                qdim = 1
            elif x_src.dim() == 3:
                kdim1 = qdim = 1
                lenq = x_dst.shape[1]
                lenk = x_src.shape[1]
            q = self.to_q(x_dst).view(
                B, lenq, qdim, self.num_heads, self.head_dim
            )  # [B，pl, 1, h, d]
            k = self.to_k(x_src).view(
                B, kdim1, lenk, self.num_heads, self.head_dim
            )  # [B，pl, pt, h, d]
            v = self.to_v(x_src).view(
                B, kdim1, lenk, self.num_heads, self.head_dim
            )  # [B，pl, pt, h, d]

            if self.has_pos_emb:
                rk = self.to_k_r(r).view(
                    B, lenq, lenk, self.num_heads, self.head_dim
                )
                rv = self.to_v_r(r).view(
                    B, lenq, lenk, self.num_heads, self.head_dim
                )
        if self.has_pos_emb:
            k = k + rk
            v = v + rv
        sim = q * k
        sim = sim.sum(dim=-1)
        sim = sim * self.scale  # [B, pl, pt, h]

        if mask is not None:
            sim = torch.where(
                mask.unsqueeze(-1),
                sim,
                self.quant(torch.tensor(-100.0).to(mask.device)),
            )
        attn = torch.softmax(sim, dim=-2)  # [B, pl, pt, h]

        attn = self.attn_drop(attn)
        out = v * attn.unsqueeze(-1)  # [B, pl, pt, h, d]
        out = torch.sum(out, dim=-3)  # [B, pl, h, d]

        if extra_1dim:
            inputs = out.view(B, ex_dim, -1, self.num_heads * self.head_dim)
        else:
            inputs = out.view(B, -1, self.num_heads * self.head_dim)
        x = torch.cat([inputs, x_dst], dim=-1)
        g = torch.sigmoid(self.to_g(x))
        agg = inputs + g * (self.to_s(x_dst) - inputs)

        return self.to_out(agg)

    def forward(self, x, r, mask=None, extra_dim=False):
        if isinstance(x, torch.Tensor):
            x_src = x_dst = self.attn_prenorm_x_src(x)
            x_dst = x_dst
        else:
            x_src, x_dst = x
            if self.bipartite:
                x_src = self.attn_prenorm_x_src(x_src)
                x_dst = self.attn_prenorm_x_dst(x_dst)
            else:
                x_src = self.attn_prenorm_x_src(x_src)
                x_dst = self.attn_prenorm_x_src(x_dst)
            x = x[1]

        attn = self._attn_block(
            x_src, x_dst, r, mask=mask, extra_1dim=extra_dim
        )  # [B, pl, h*d]
        x = x + attn
        x2 = self.ff_prenorm(x)
        x = x + self.ff_mlp(x2)

        return x
