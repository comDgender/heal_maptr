# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import torch.nn as nn
from hat_engine.registry import OBJECT_REGISTRY
from horizon_plugin_pytorch.quantization import QuantStub

from .fourier_embedding import FourierConvEmbedding
from .rattention import RAttentionLayer
from .utils import weight_init

__all__ = ["QCNetOEMapEncoder"]


@OBJECT_REGISTRY.register
class QCNetOEMapEncoder(nn.Module):
    """
    QCNetOEMapEncoder is a module for encoding map data.

    Args:
        input_dim: Dimension of the input data.
        hidden_dim: Dimension of the hidden layers.
        num_historical_steps: Number of historical time steps.
        num_freq_bands: Number of frequency bands for Fourier Embedding.
        num_layers: Number of layers in the model.
        num_heads: Number of attention heads.
        head_dim: Dimension of each attention head.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        num_historical_steps: int = 50,
        num_freq_bands: int = 64,
        num_layers: int = 1,
        num_heads: int = 8,
        head_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        input_dim_x_pt = 1
        input_dim_x_pl = 0
        input_dim_r_pt2pl = 3
        input_dim_r_pl2pl = 3

        # Compact builders; the produced submodules (and their state-dict
        # keys) are byte-for-byte equivalent to explicit construction.
        emb_kw = {"hidden_dim": hidden_dim, "num_freq_bands": num_freq_bands}

        def conv_emb(in_dim):
            return FourierConvEmbedding(input_dim=in_dim, **emb_kw)

        def rel_attn_stack(is_bipartite):
            return nn.ModuleList(
                RAttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=is_bipartite,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            )

        for emb_name, n_cls in (
            ("type_pl2pl_emb", 5),
            ("type_pt_emb", 17),
            ("side_pt_emb", 3),
            ("type_pl_emb", 4),
            ("int_pl_emb", 3),
        ):
            setattr(self, emb_name, nn.Embedding(n_cls, hidden_dim))

        self.x_pt_emb = conv_emb(input_dim_x_pt)
        self.x_pl_emb = conv_emb(input_dim_x_pl)
        self.r_pt2pl_emb = conv_emb(input_dim_r_pt2pl)
        self.r_pl2pl_emb = conv_emb(input_dim_r_pl2pl)
        self.pt2pl_layers = rel_attn_stack(True)
        self.pl2pl_layers = rel_attn_stack(False)

        self.map_point_magnitude_quant = QuantStub()

        self.quant_pt_type = QuantStub()
        self.quant_pt_side = QuantStub()
        self.quant_pl_type = QuantStub()
        self.quant_pl_int = QuantStub()
        self.type_pl2pl_quant = QuantStub()

        self.r_pt2pl_quant = QuantStub()
        self.r_pl2pl_quant = QuantStub()

        self.r_pt2pl_quant = nn.ModuleList(
            [QuantStub() for i in range(input_dim_r_pt2pl)]
        )
        self.r_pl2pl_quant = nn.ModuleList(
            [QuantStub() for i in range(input_dim_r_pl2pl)]
        )

        self.apply(weight_init)

    def forward(self, data: dict):
        data["map_point"]["magnitude_q"] = self.map_point_magnitude_quant(
            data["map_point"]["magnitude"]
        )
        x_pt = data["map_point"]["magnitude_q"]

        r_pl2pl = data["map_polygon"]["r_pl2pl"]
        r_pt2pl = data["map_polygon"]["r_pt2pl"]

        r_pl2pl = [self.r_pl2pl_quant[i](a) for i, a in enumerate(r_pl2pl)]
        r_pt2pl = [self.r_pt2pl_quant[i](a) for i, a in enumerate(r_pt2pl)]

        mask_pl2pl = data["map_polygon"]["mask_pl2pl"]
        pl2pl_near_idx = data["map_polygon"].get(
            "pl2pl_near_idx", None
        )  # [B, pl, pl_K]

        B, _, pl_N = r_pl2pl[0].shape[:3]

        x_pt_categorical_embs = self.quant_pt_type(
            self.type_pt_emb(data["map_point"]["pt_type"].long())
        ) + self.quant_pt_side(
            self.side_pt_emb(data["map_point"]["side"].long())
        )
        x_pl_categorical_embs = self.quant_pl_type(
            self.type_pl_emb(data["map_polygon"]["pl_type"].long())
        ) + self.quant_pl_int(
            self.int_pl_emb(data["map_polygon"]["is_intersection"].long())
        )

        x_pt = self.x_pt_emb(
            continuous_inputs=[
                x_pt.unsqueeze(1),
            ],
            categorical_embs=x_pt_categorical_embs.permute(0, 3, 1, 2),
        )  # [B, pl, pt, D]
        x_pt = x_pt.permute(0, 2, 3, 1)
        x_pl = x_pl_categorical_embs

        r_pt2pl = self.r_pt2pl_emb(
            continuous_inputs=r_pt2pl, categorical_embs=None
        )  # [B, pl, pt, 128]
        r_pt2pl = r_pt2pl.permute(0, 2, 3, 1)
        mask_pt2pl = data["map_point"]["mask"].bool()

        type_pl2pl = data["type_pl2pl"]
        type_pl2pl = self.type_pl2pl_quant(self.type_pl2pl_emb(type_pl2pl))
        r_pl2pl = self.r_pl2pl_emb(
            continuous_inputs=r_pl2pl,
            categorical_embs=type_pl2pl.permute(0, 3, 1, 2),
        )  # [B, pl, pl,  128]
        r_pl2pl = r_pl2pl.permute(0, 2, 3, 1)

        if pl2pl_near_idx is not None:
            x_pl_src = (
                x_pl.unsqueeze(2)
                .repeat(1, 1, pl_N, 1)
                .gather(
                    2,
                    pl2pl_near_idx.unsqueeze(-1)
                    .repeat(1, 1, 1, self.hidden_dim)
                    .long(),
                )
            )  # [B, pl, D]-->[B, pl, pl_K, D]
        else:
            x_pl_src = x_pl

        for i in range(self.num_layers):
            x_pl = self.pt2pl_layers[i](
                (x_pt, x_pl), r_pt2pl, mask=mask_pt2pl
            )  # [B, pl, 128]
            x_pl = self.pl2pl_layers[i](
                (x_pl_src, x_pl), r_pl2pl, mask=mask_pl2pl
            )
        return {
            "x_pt": x_pt,
            "x_pl": x_pl,
        }

    def set_qconfig(self):
        self.x_pl_emb.set_qconfig()
        self.x_pt_emb.set_qconfig()
        self.type_pl2pl_emb.qconfig = None
        self.type_pt_emb.qconfig = None
        self.side_pt_emb.qconfig = None
        self.type_pl_emb.qconfig = None
        self.int_pl_emb.qconfig = None
