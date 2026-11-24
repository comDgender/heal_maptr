# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import math

import torch
import torch.nn as nn
from hat_engine.registry import OBJECT_REGISTRY
from horizon_plugin_pytorch.quantization import QuantStub
from timm.models.layers import Mlp

from heal.models.task_modules.diffusion_planner.mixer import MixerBlock
from heal.models.task_modules.diffusion_planner.sde import SDE

__all__ = [
    "DiT",
]


@OBJECT_REGISTRY.register
class DiT(nn.Module):
    """Diffusion Transformer for trajectory denoising in the planner.

    It composes route encoding, timestep embedding and DiT blocks with
    cross-attention to predict trajectories under score or x_start
    parameterization.
    """

    def __init__(
        self,
        sde: SDE,
        route_encoder: nn.Module,
        depth,
        output_dim,
        hidden_dim=192,
        heads=6,
        dropout=0.1,
        mlp_ratio=4.0,
        model_type="x_start",
    ):
        """Initialize the DiT model.

        Args:
            sde: SDE providing the marginal probability std.
            route_encoder: Module encoding route lanes.
            depth: Number of DiT blocks.
            output_dim: Trajectory feature dimension.
            hidden_dim: Hidden feature dimension.
            heads: Number of attention heads.
            dropout: Attention dropout rate.
            mlp_ratio: MLP hidden-dim expansion ratio.
            model_type: Parameterization type, one of "score" or "x_start".
        """
        super().__init__()

        assert model_type in [
            "score",
            "x_start",
        ], f"Unknown model type: {model_type}"
        self._model_type = model_type
        self.hidden_dim = hidden_dim
        self.route_encoder = route_encoder
        self.agent_embedding = nn.Embedding(2, hidden_dim)

        self.preproj = Mlp(
            in_features=output_dim,
            hidden_features=512,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_dim, heads, dropout, mlp_ratio)
                for i in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_dim, output_dim)
        self.sde = sde
        self.marginal_prob_std = self.sde.marginal_prob_std

    @property
    def model_type(self):
        """Return the parameterization type of the model."""
        return self._model_type

    def forward(
        self,
        x,
        t,
        cross_c,
        route_lanes,
        route_lanes_speed_limit,
        route_lanes_has_speed_limit,
        neighbor_current_mask=None,
        current_states=None,
    ):
        """Forward pass of DiT.

        x: (B, P, output_dim) -> Embedded out of DiT
        t: (B,)
        cross_c: (B, N, D) -> Cross-Attention context
        route_lanes_speed_limit: (B, 1, 1)
        route_lanes_has_speed_limit: (B, 1, 1)
        """
        B, P, _ = x.shape

        x = self.preproj(x)

        x_embedding = torch.cat(
            [
                self.agent_embedding.weight[0][None, :],
                self.agent_embedding.weight[1][None, :].expand(P - 1, -1),
            ],
            dim=0,
        )  # (P, D)
        x_embedding = x_embedding[None, :, :].expand(B, -1, -1)  # (B, P, D)
        x = x + x_embedding

        route_encoding = self.route_encoder(
            route_lanes, route_lanes_speed_limit, route_lanes_has_speed_limit
        )
        y = route_encoding + self.t_embedder(t)

        attn_mask = torch.zeros((B, P), dtype=torch.bool, device=x.device)
        if neighbor_current_mask is not None:
            attn_mask[:, 1:] = neighbor_current_mask

        for block in self.blocks:
            x = block(x, cross_c, y, attn_mask)

        x = self.final_layer(x, y)

        if self._model_type == "score":
            return x / (self.marginal_prob_std(t)[:, None, None] + 1e-6)
        elif self._model_type == "x_start":
            return x
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")


def modulate(x, shift, scale, only_first=False):
    """Apply scale-shift modulation to token features.

    Args:
        x: Input features of shape (B, P, D).
        shift: Shift tensor broadcast over tokens.
        scale: Scale tensor broadcast over tokens.
        only_first: Whether to modulate only the first token.

    Returns:
        Modulated features of the same shape as x.
    """
    if only_first:
        x_first, x_rest = x[:, :1], x[:, 1:]
        x = torch.cat(
            [x_first * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), x_rest],
            dim=1,
        )
    else:
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    return x


def scale(x, scale, only_first=False):
    """Apply scale-only modulation to token features.

    Args:
        x: Input features of shape (B, P, D).
        scale: Scale tensor broadcast over tokens.
        only_first: Whether to modulate only the first token.

    Returns:
        Scaled features of the same shape as x.
    """
    if only_first:
        x_first, x_rest = x[:, :1], x[:, 1:]
        x = torch.cat([x_first * (1 + scale.unsqueeze(1)), x_rest], dim=1)
    else:
        x = x * (1 + scale.unsqueeze(1))

    return x


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        """Initialize the timestep embedder.

        Args:
            hidden_size: Output embedding dimension.
            frequency_embedding_size: Sinusoidal frequency dimension.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.t_freq_quant = QuantStub()

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Create sinusoidal timestep embeddings."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        """Embed scalar timesteps into vector representations.

        Args:
            t: Timestep tensor of shape (B,).

        Returns:
            Timestep embedding tensor of shape (B, hidden_size).
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = self.t_freq_quant(t_freq)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTBlock(nn.Module):
    """A DiT block with adaLN-Zero conditioning and Cross-Attention."""

    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0):
        """Initialize the DiT block.

        Args:
            dim: Feature dimension.
            heads: Number of attention heads.
            dropout: Attention dropout rate.
            mlp_ratio: MLP hidden-dim expansion ratio.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, heads, dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp1 = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        self.norm3 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, heads, dropout, batch_first=True
        )
        self.norm4 = nn.LayerNorm(dim)

        self.mlp2 = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x, cross_c, y, attn_mask):
        """Run the DiT block forward pass.

        Args:
            x: Token features of shape (B, P, D).
            cross_c: Cross-attention context of shape (B, N, D).
            y: Conditioning embedding of shape (B, D).
            attn_mask: Self-attention key padding mask of shape (B, P).

        Returns:
            Updated token features of shape (B, P, D).
        """
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(y).chunk(6, dim=1)

        modulated_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = (
            x
            + gate_msa.unsqueeze(1)
            * self.attn(
                modulated_x,
                modulated_x,
                modulated_x,
                key_padding_mask=attn_mask,
            )[0]
        )

        modulated_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp1(modulated_x)

        x = self.cross_attn(self.norm3(x), cross_c, cross_c)[0]
        x = self.mlp2(self.norm4(x))

        return x


class FinalLayer(nn.Module):
    """The final layer of DiT."""

    def __init__(self, hidden_size, output_size):
        """Initialize the final layer.

        Args:
            hidden_size: Input feature dimension.
            output_size: Output trajectory feature dimension.
        """
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size)
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4, bias=True),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, output_size, bias=True),
        )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, y):
        """Project token features to the output dimension.

        Args:
            x: Token features of shape (B, P, hidden_size).
            y: Conditioning embedding of shape (B, hidden_size).

        Returns:
            Output features of shape (B, P, output_size).
        """
        B, P, _ = x.shape

        shift, scale = self.adaLN_modulation(y).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.proj(x)
        return x


@OBJECT_REGISTRY.register
class RouteEncoder(nn.Module):
    """Encoder for route lanes with speed limit information."""

    def __init__(
        self,
        route_num,
        lane_len,
        drop_path_rate=0.3,
        hidden_dim=192,
        tokens_mlp_dim=32,
        channels_mlp_dim=64,
    ):
        """Initialize the route encoder.

        Args:
            route_num: Number of route lanes.
            lane_len: Number of points per lane.
            drop_path_rate: Drop path rate for the mixer blocks.
            hidden_dim: Output embedding dimension.
            tokens_mlp_dim: Token-mixing MLP dimension.
            channels_mlp_dim: Channel-mixing MLP dimension.
        """
        super().__init__()

        self._channel = channels_mlp_dim

        self.speed_limit_emb = nn.Linear(1, channels_mlp_dim)
        self.unknown_speed_emb = nn.Embedding(1, channels_mlp_dim)

        self.channel_pre_project = Mlp(
            in_features=4,
            hidden_features=channels_mlp_dim,
            out_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.token_pre_project = Mlp(
            in_features=route_num * lane_len,
            hidden_features=tokens_mlp_dim,
            out_features=tokens_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        self.Mixer = MixerBlock(
            tokens_mlp_dim, channels_mlp_dim, drop_path_rate
        )

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(
            in_features=channels_mlp_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )
        self.res_quant = QuantStub()
        self.unknown_speed_emb_quant = QuantStub()
        self.speed_limit_quant = QuantStub()

    def forward(self, x, route_lanes_speed_limit, route_lanes_has_speed_limit):
        """x: B, P, V, D."""
        # only x and x->x' vector,no boundary,no speed limit,no traffic light
        x = x[..., :4]

        B, P, V, _ = x.shape
        mask_v = (
            torch.sum(torch.ne(x[..., :4], 0).to(torch.int8), dim=-1).to(
                x.device
            )
            == 0
        )
        mask_p = torch.sum((~mask_v).to(torch.int8), dim=-1) == 0
        mask_b = torch.sum((~mask_p).to(torch.int8), dim=-1) == 0
        x = x.view(B, P * V, -1)

        valid_indices = ~mask_b.view(-1)

        mask = valid_indices.view(-1, 1, 1)
        x = x * mask

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.Mixer(x)

        x = torch.mean(x, dim=1)

        # Speed limit encoding
        route_lanes_speed_limit = route_lanes_speed_limit.view(B, 1)
        route_lanes_has_speed_limit = route_lanes_has_speed_limit.view(
            B, 1
        ).bool()

        route_lanes_speed_limit = self.speed_limit_quant(
            route_lanes_speed_limit
        )
        speed_limit_with_limit = (
            self.speed_limit_emb(route_lanes_speed_limit)
            * route_lanes_has_speed_limit
        )
        speed_limit_no_limit = self.unknown_speed_emb.weight.expand(B, -1)
        speed_limit_no_limit = self.unknown_speed_emb_quant(
            speed_limit_no_limit
        ) * (~route_lanes_has_speed_limit)
        speed_limit_embedding = speed_limit_with_limit + speed_limit_no_limit

        x = x + speed_limit_embedding
        x = self.emb_project(self.norm(x))

        x_result = x * mask.squeeze(-1)

        return x_result.view(B, -1)


@OBJECT_REGISTRY.register
class CurrentStageEncoder(nn.Module):
    """Encoder for current states of agents."""

    def __init__(
        self,
        time_len=1,
        agent_len=11,
        drop_path_rate=0.3,
        hidden_dim=192,
        depth=3,
        tokens_mlp_dim=64,
        channels_mlp_dim=128,
    ):
        """Initialize the current state encoder.

        Args:
            time_len: Temporal length of the input states.
            agent_len: Number of agents.
            drop_path_rate: Drop path rate for the mixer blocks.
            hidden_dim: Output embedding dimension.
            depth: Unused, kept for config compatibility.
            tokens_mlp_dim: Token-mixing MLP dimension.
            channels_mlp_dim: Channel-mixing MLP dimension.
        """
        super().__init__()

        self._hidden_dim = hidden_dim
        self._channel = channels_mlp_dim

        self.type_emb = nn.Linear(3, channels_mlp_dim)

        self.channel_pre_project = Mlp(
            in_features=4,
            hidden_features=channels_mlp_dim,
            out_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.token_pre_project = Mlp(
            in_features=time_len * agent_len,
            hidden_features=tokens_mlp_dim,
            out_features=tokens_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        self.Mixer = MixerBlock(
            tokens_mlp_dim, channels_mlp_dim, drop_path_rate
        )

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(
            in_features=channels_mlp_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

        self.res_quant = QuantStub()

    def forward(self, x):
        """x: B, P, V, D."""
        B, P, V, _ = x.shape
        mask_v = (
            torch.sum(torch.ne(x, 0).to(torch.int8), dim=-1).to(x.device) == 0
        )
        mask_p = torch.sum((~mask_v).to(torch.int8), dim=-1) == 0
        mask_b = torch.sum((~mask_p).to(torch.int8), dim=-1) == 0
        x = x.view(B, P * V, -1)

        valid_indices = ~mask_b.view(-1)
        x = x[valid_indices]

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.Mixer(x)

        x = torch.mean(x, dim=1)

        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B, x.shape[-1]), device=x.device)
        x_result = self.res_quant(x_result)
        x_result[valid_indices] = x  # Fill in valid parts

        return x_result.view(B, -1)
