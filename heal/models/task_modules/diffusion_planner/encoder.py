# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import torch
import torch.nn as nn
from hat_engine.models.base_modules.conv_module import ConvModule2d
from hat_engine.registry import OBJECT_REGISTRY
from horizon_plugin_pytorch.quantization import QuantStub
from timm.layers import DropPath
from timm.models.layers import Mlp
from torch.quantization import DeQuantStub

from heal.models.task_modules.diffusion_planner.mixer import MixerBlock


@OBJECT_REGISTRY.register
class DiffusionPlannerEncoder(nn.Module):
    """Diffusion planner encoder composing sub-encoders and fusion.

    It runs neighbor, static, lane and BEV sub-encoders, concatenates their
    token encodings with position embeddings, and fuses them.
    """

    def __init__(
        self,
        neighbor_encoder: nn.Module = None,
        static_encoder: nn.Module = None,
        lane_encoder: nn.Module = None,
        fusion_encoder: nn.Module = None,
        hidden_dim: int = 192,
        token_num: int = 50,
        bev_encoder: nn.Module = None,
        use_static_encoder: bool = True,
        num_bev_inputs: int = 1,
        quantize_bev_features: bool = True,
    ):
        """Initialize the planner encoder.

        Args:
            neighbor_encoder: Encoder for neighbor agent histories.
            static_encoder: Encoder for static objects.
            lane_encoder: Encoder for lane polylines.
            fusion_encoder: Fusion module applied over concatenated tokens.
            hidden_dim: Hidden embedding dimension.
            token_num: Base token number before adding BEV tokens.
            bev_encoder: Optional BEV encoder.
            use_static_encoder: Whether the static encoder branch is enabled.
            num_bev_inputs: Number of BEV feature maps fed to BEV encoder.
            quantize_bev_features: Whether to quantize BEV feature inputs.

        Raises:
            ValueError: If static encoder is enabled but not provided.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.base_token_num = token_num
        self.token_num = token_num
        self.use_static_encoder = use_static_encoder
        self.quantize_bev_features = quantize_bev_features

        self.neighbor_encoder = neighbor_encoder
        self.static_encoder = static_encoder
        self.lane_encoder = lane_encoder
        self.bev_encoder = bev_encoder
        self.num_bev_inputs = num_bev_inputs

        self.fusion = fusion_encoder

        # QuantStubs — only create for encoders that exist
        if self.use_static_encoder and static_encoder is not None:
            self.static_quant = QuantStub()
        else:
            self.static_quant = None
        if lane_encoder is not None:
            self.lanes_quant = QuantStub()
        else:
            self.lanes_quant = None
        if bev_encoder is not None:
            assert self.num_bev_inputs in [1, 3, 4]
            if self.num_bev_inputs == 1:
                self.bev_quant = QuantStub()
            elif self.num_bev_inputs == 3:
                self.bev_quant = QuantStub()
                self.dynamic_feature_quant = QuantStub()
                self.static_feature_quant = QuantStub()
            else:
                self.bev_quant = QuantStub()
                self.dynamic_feature_quant = QuantStub()
                self.static_feature_quant = QuantStub()
                self.occ_feature_quant = QuantStub()
        self.encoder_pos_quant = QuantStub()
        self.dequant = DeQuantStub()

        # position embedding encode x, y, cos, sin, type
        self.pos_emb = nn.Linear(7, hidden_dim)

        if bev_encoder is not None:
            self.token_num = token_num + bev_encoder.bev_token_num

        if self.use_static_encoder and self.static_encoder is None:
            raise ValueError("`static_encoder` must be provided when enabled.")

    def _has_bev_inputs(self, inputs):
        """Check if BEV inputs are present for pluggable BEV encoding.

        Args:
            inputs: Input dict possibly containing BEV features.

        Returns:
            True if BEV encoder exists and "base_bev" is in inputs.
        """
        if self.bev_encoder is None:
            return False
        return "base_bev" in inputs

    def forward(self, inputs, quantized_neighbor_agents_past=None):
        """Run sub-encoders and fuse token encodings.

        Args:
            inputs: Dict with static objects, lanes and BEV feature maps.
            quantized_neighbor_agents_past: Quantized neighbor agent history
                tensor consumed by the neighbor encoder.

        Returns:
            Dict with fused token encoding under key "encoding".

        Raises:
            RuntimeError: If no encoder input is provided to infer batch size.
        """

        encoder_outputs = {}

        B = None
        encoding_list = []
        mask_list = []
        pos_list = []

        # --- Neighbor (agent) encoder ---
        if (
            self.neighbor_encoder is not None
            and quantized_neighbor_agents_past is not None
        ):
            neighbors = quantized_neighbor_agents_past
            B = neighbors.shape[0]
            (
                encoding_neighbors,
                neighbors_mask,
                neighbor_pos,
            ) = self.neighbor_encoder(neighbors)
            encoding_list.append(encoding_neighbors)
            mask_list.append(neighbors_mask)
            pos_list.append(neighbor_pos)

        # --- Static encoder ---
        if (
            self.use_static_encoder
            and self.static_encoder is not None
            and "static_objects" in inputs
        ):
            static = self.static_quant(inputs["static_objects"])
            if B is None:
                B = static.shape[0]
            encoding_static, static_mask, static_pos = self.static_encoder(
                static
            )
            encoding_list.append(encoding_static)
            mask_list.append(static_mask)
            pos_list.append(static_pos)

        # --- Lane encoder ---
        if self.lane_encoder is not None and "lanes" in inputs:
            lanes = self.lanes_quant(inputs["lanes"])
            if B is None:
                B = lanes.shape[0]
            encoding_lanes, lanes_mask, lane_pos = self.lane_encoder(lanes)
            encoding_list.append(encoding_lanes)
            mask_list.append(lanes_mask)
            pos_list.append(lane_pos)

        # --- BEV encoder (single or triple BEV input) ---
        if self._has_bev_inputs(inputs):
            base_bev = inputs["base_bev"]
            if B is None:
                B = base_bev.shape[0]

            if self.num_bev_inputs == 1:
                # BEVFusionEncoder: single unified BEV feature
                if self.quantize_bev_features:
                    base_bev = self.bev_quant(base_bev)
                encoding_bev, bev_mask, bev_pos = self.bev_encoder(base_bev)
            elif self.num_bev_inputs == 3:
                # BEVFusionEncoder: three separate BEV maps
                dynamic_bev = inputs["dynamic_bev"]
                static_bev = inputs["static_bev"]
                if self.quantize_bev_features:
                    base_bev_q = self.bev_quant(base_bev)
                    dynamic_bev_q = self.dynamic_feature_quant(dynamic_bev)
                    static_bev_q = self.static_feature_quant(static_bev)
                encoding_bev, bev_mask, bev_pos = self.bev_encoder(
                    base_bev_q, dynamic_bev_q, static_bev_q
                )
            else:
                # BEVFusionEncoder: four separate BEV maps
                dynamic_bev = inputs["dynamic_bev"]
                static_bev = inputs["static_bev"]
                occ_bev = inputs["occ_bev"]
                base_bev_q = self.bev_quant(base_bev)
                dynamic_bev_q = self.dynamic_feature_quant(dynamic_bev)
                static_bev_q = self.static_feature_quant(static_bev)
                # occ_bev_q = self.occ_feature_quant(occ_bev)
                encoding_bev, bev_mask, bev_pos = self.bev_encoder(
                    base_bev_q, dynamic_bev_q, static_bev_q, occ_bev
                )
            encoding_list.append(encoding_bev)
            mask_list.append(bev_mask)
            pos_list.append(bev_pos)

        if B is None:
            raise RuntimeError(
                "DiffusionPlannerEncoder: could not determine "
                "batch size. Ensure at least one encoder input "
                "is provided."
            )

        # --- Fuse all encodings ---
        token_num = sum(item.shape[1] for item in mask_list)
        encoding_input = torch.cat(encoding_list, dim=1)
        encoding_pos = torch.cat(pos_list, dim=1).view(B * token_num, -1)
        encoding_mask = torch.cat(mask_list, dim=1).view(-1)

        valid_indices = ~encoding_mask
        encoding_pos = encoding_pos * valid_indices.unsqueeze(-1)
        encoding_pos = self.pos_emb(encoding_pos)
        encoding_pos_result = encoding_pos * valid_indices.unsqueeze(-1)

        encoding_input = encoding_input + encoding_pos_result.view(
            B, token_num, -1
        )

        if len(encoding_list) == 1:
            encoding_result = encoding_input
        else:
            encoding_result = self.fusion(
                encoding_input, encoding_mask.view(B, token_num)
            )
        encoder_outputs["encoding"] = self.dequant(encoding_result)

        return encoder_outputs


@OBJECT_REGISTRY.register
class SelfAttentionBlock(nn.Module):
    """Self-attention block with multi-head attention and MLP."""

    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0):
        """Initialize the self-attention block.

        Args:
            dim: Embedding dimension.
            heads: Number of attention heads.
            dropout: Dropout and drop-path rate.
            mlp_ratio: Ratio of MLP hidden dim to embedding dim.
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, heads, dropout, batch_first=True
        )

        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=nn.GELU,
            drop=dropout,
        )

    def forward(self, x, mask):
        """Apply attention and MLP with residual connections.

        Args:
            x: Input token tensor of shape (B, N, dim).
            mask: Key padding mask, True for padded tokens.

        Returns:
            Output token tensor of the same shape as input.
        """
        x = x + self.drop_path(
            self.attn(self.norm1(x), x, x, key_padding_mask=mask)[0]
        )
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


@OBJECT_REGISTRY.register
class AgentFusionEncoder(nn.Module):
    """Encoder for neighbor agent trajectories."""

    def __init__(
        self,
        time_len,
        drop_path_rate=0.3,
        hidden_dim=192,
        depth=3,
        tokens_mlp_dim=64,
        channels_mlp_dim=128,
    ):
        """Initialize the agent fusion encoder.

        Args:
            time_len: Number of history time steps per agent.
            drop_path_rate: Drop-path rate for mixer blocks.
            hidden_dim: Output embedding dimension.
            depth: Number of mixer blocks.
            tokens_mlp_dim: Token-mixing MLP dimension.
            channels_mlp_dim: Channel-mixing MLP dimension.
        """
        super().__init__()

        self._hidden_dim = hidden_dim
        self._channel = channels_mlp_dim

        self.type_emb = nn.Linear(3, channels_mlp_dim)

        self.channel_pre_project = Mlp(
            in_features=8 + 1,
            hidden_features=channels_mlp_dim,
            out_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.token_pre_project = Mlp(
            in_features=time_len,
            hidden_features=tokens_mlp_dim,
            out_features=tokens_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        self.blocks = nn.ModuleList(
            [
                MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)
                for i in range(depth)
            ]
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
        self.mask_qant = QuantStub()
        self.type_pos_quant = QuantStub()

    def forward(self, x):
        """x: B, P, V, D (x, y, cos, sin, vx, vy, w, l, type(3))."""
        B, P, V, _ = x.shape
        neighbor_type = x[:, :, -1, 8:]
        x = x[..., :8]

        geometry_features = x[:, :, -1, :4].clone()  # Shape: B, P, 4
        neighbor_type_pos = torch.zeros((B, P, 3), device=x.device)
        neighbor_type_pos = self.type_pos_quant(neighbor_type_pos)
        neighbor_type_pos[..., 0:1] = 1.0
        pos = torch.cat((geometry_features, neighbor_type_pos), dim=-1)

        mask_v = (
            torch.sum(torch.ne(x[..., :8], 0).to(torch.int8), dim=-1).to(
                x.device
            )
            == 0
        )
        mask_p = torch.sum((~mask_v).to(torch.int8), dim=-1) == 0
        x = torch.cat(
            [x, self.mask_qant((~mask_v).float().unsqueeze(-1))], dim=-1
        )
        x = x.view(B * P, V, -1)

        valid_indices = ~mask_p.view(-1)

        # masked version for bpu
        mask = valid_indices.view(-1, 1, 1)
        x_masked = x * mask

        x_masked = self.channel_pre_project(x_masked)
        x_masked = x_masked.permute(0, 2, 1)
        x_masked = self.token_pre_project(x_masked)
        x_masked = x_masked.permute(0, 2, 1)
        for block in self.blocks:
            x_masked = block(x_masked)
        # pooling
        x_masked = torch.mean(x_masked, dim=1)

        neighbor_type = neighbor_type.view(B * P, -1)
        neighbor_type_mask = neighbor_type * (mask.squeeze(-1))
        neighbor_type_mask = self.type_emb(
            neighbor_type_mask
        )  # Type embedding for valid data
        x_masked = x_masked + neighbor_type_mask

        x_masked = self.emb_project(self.norm(x_masked))

        x_result = x_masked * mask.squeeze(-1)

        return (
            x_result.view(B, P, -1),
            mask_p.reshape(B, -1),
            pos.view(B, P, -1),
        )


@OBJECT_REGISTRY.register
class StaticFusionEncoder(nn.Module):
    """Encoder for static objects."""

    def __init__(self, dim, drop_path_rate=0.3, hidden_dim=192, device="cuda"):
        """Initialize the static fusion encoder.

        Args:
            dim: Input feature dimension per static object.
            drop_path_rate: Dropout rate of the projection MLP.
            hidden_dim: Output embedding dimension.
            device: Target device for generated tensors.
        """
        super().__init__()

        self._hidden_dim = hidden_dim
        self.res_quant = QuantStub()
        self.type_pos_quant = QuantStub()
        self.projection = Mlp(
            in_features=dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

    def forward(self, x):
        """x: B, P, D (x, y, cos, sin, w, l, type(4))."""
        B, P, _ = x.shape

        geometry_features = x[:, :, :4].clone()  # Shape: B, P, 4
        neighbor_type_pos = torch.zeros((B, P, 3), device=x.device)
        neighbor_type_pos = self.type_pos_quant(neighbor_type_pos)
        neighbor_type_pos[..., 1:2] = 1.0
        pos = torch.cat((geometry_features, neighbor_type_pos), dim=-1)

        x_result = torch.zeros((B * P, self._hidden_dim), device=x.device)
        x_result = self.res_quant(x_result)
        mask_p = (
            torch.sum(torch.ne(x[..., :10], 0).to(torch.int8), dim=-1).to(
                x.device
            )
            == 0
        )

        valid_indices = ~mask_p.view(-1)

        if valid_indices.sum() > 0:
            mask = valid_indices.unsqueeze(-1)
            x_full = x.view(B * P, -1)
            x_masked = x_full * mask
            x_proj_full = self.projection(x_masked)
            x_result = x_proj_full * mask

        return x_result.view(B, P, -1), mask_p.view(B, P), pos.view(B, P, -1)


@OBJECT_REGISTRY.register
class LaneFusionEncoder(nn.Module):
    """Encoder for lane polylines."""

    def __init__(
        self,
        lane_len,
        drop_path_rate=0.3,
        hidden_dim=192,
        depth=3,
        tokens_mlp_dim=64,
        channels_mlp_dim=128,
        with_traffic_emb=True,
    ):
        """Initialize the lane fusion encoder.

        Args:
            lane_len: Number of points per lane polyline.
            drop_path_rate: Drop-path rate for mixer blocks.
            hidden_dim: Output embedding dimension.
            depth: Number of mixer blocks.
            tokens_mlp_dim: Token-mixing MLP dimension.
            channels_mlp_dim: Channel-mixing MLP dimension.
            with_traffic_emb: Whether to add traffic-light embeddings.
        """
        super().__init__()

        self._lane_len = lane_len
        self._channel = channels_mlp_dim

        self.with_traffic_emb = with_traffic_emb
        if self.with_traffic_emb:
            self.traffic_emb = nn.Linear(4, channels_mlp_dim)

        self.channel_pre_project = Mlp(
            in_features=4,
            hidden_features=channels_mlp_dim,
            out_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )
        self.token_pre_project = Mlp(
            in_features=lane_len,
            hidden_features=tokens_mlp_dim,
            out_features=tokens_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        self.blocks = nn.ModuleList(
            [
                MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)
                for i in range(depth)
            ]
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
        self.type_pos_quant = QuantStub()

    def forward(self, x):
        """x: B, P, V, D (x, y, x'-x, y'-y, x_left-x, ... , traffic(4))."""
        B, P, V, _ = x.shape
        if self.with_traffic_emb:
            traffic = x[:, :, 0, 8:]
        x = x[..., :4]

        geometry_features = x[
            :, :, int(self._lane_len / 2), :4
        ].clone()  # Shape: B, P, 4
        heading = torch.atan2(
            geometry_features[..., 3], geometry_features[..., 2]
        )
        geometry_features = torch.cat(
            (
                geometry_features[..., :2],
                torch.cos(heading).unsqueeze(-1),
                torch.sin(heading).unsqueeze(-1),
            ),
            dim=-1,
        )
        neighbor_type_pos = torch.zeros((B, P, 3), device=x.device)
        neighbor_type_pos = self.type_pos_quant(neighbor_type_pos)
        neighbor_type_pos[..., -1:] = 1.0
        pos = torch.cat((geometry_features, neighbor_type_pos), dim=-1)

        mask_v = (
            torch.sum(torch.ne(x[..., :8], 0).to(torch.int8), dim=-1).to(
                x.device
            )
            == 0
        )
        mask_p = torch.sum((~mask_v).to(torch.int8), dim=-1) == 0
        x = x.view(B * P, V, -1)

        valid_indices = ~mask_p.view(-1)

        mask = valid_indices.view(-1, 1, 1)
        x_masked = x * mask

        x_masked = self.channel_pre_project(x_masked)
        x_masked = x_masked.permute(0, 2, 1)
        x_masked = self.token_pre_project(x_masked)
        x_masked = x_masked.permute(0, 2, 1)
        for block in self.blocks:
            x_masked = block(x_masked)

        x_masked = torch.mean(x_masked, dim=1)

        if self.with_traffic_emb:
            # Process traffic lights directly for valid positions
            traffic = traffic.view(B * P, -1)
            traffic_mak = traffic * mask.squeeze(-1)
            traffic_light_embedding_mask = self.traffic_emb(traffic_mak)

            x_masked = x_masked + traffic_light_embedding_mask
        x_masked = self.emb_project(self.norm(x_masked))
        x_result = x_masked * mask.squeeze(-1)

        return (
            x_result.view(B, P, -1),
            mask_p.reshape(B, -1),
            pos.view(B, P, -1),
        )


@OBJECT_REGISTRY.register
class BEVFusionEncoder(nn.Module):
    """Pluggable BEV encoder.

    Each BEV map is first projected to a fixed per-map channel dim via 1x1
    Conv, then concatenated so that the total channel count is always
    ``bev_channels`` (i.e. ``hidden_dim`` by default). This keeps the
    downstream CNN independent of the number of BEV inputs.

    Input: any number of BEV maps, each (B, C, H, W)
    Output: (encoding, mask, pos) compatible with other encoders.
    """

    def __init__(
        self,
        bev_h: int,
        bev_w: int,
        num_bev_inputs: int = 3,
        bev_channels: int = 256,
        hidden_dim: int = 192,
        downsample_ratio: int = 8,
        drop_path_rate: float = 0.1,
        need_adaptive_pool: bool = True,
        reduce_channel_before_concat: bool = False,
        share_per_map_proj: bool = False,
        use_adapter: bool = False,
    ):
        super().__init__()
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.num_bev_inputs = num_bev_inputs
        self.downsample_ratio = downsample_ratio
        self.out_h = bev_h // downsample_ratio
        self.out_w = bev_w // downsample_ratio
        self.bev_token_num = self.out_h * self.out_w
        self.need_adaptive_pool = need_adaptive_pool
        self.res_quant = QuantStub()
        self.type_pos_quant = QuantStub()
        self.reduce_channel_before_concat = reduce_channel_before_concat
        self.use_adapter = use_adapter
        if num_bev_inputs == 1 or not reduce_channel_before_concat:
            in_channels = num_bev_inputs * bev_channels

            self.conv_layers = nn.Sequential(
                nn.Conv2d(
                    in_channels, hidden_dim, kernel_size=3, stride=2, padding=1
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(
                    hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(
                    hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
            )
        else:
            self.share_per_map_proj = share_per_map_proj
            if num_bev_inputs == 2:
                per_map_out = bev_channels // 2
            else:
                per_map_out = bev_channels // 2 - 32

            if share_per_map_proj:
                self.per_map_proj = nn.Conv2d(
                    bev_channels,
                    per_map_out,
                    kernel_size=1,
                )
            else:
                self.per_map_proj = nn.ModuleList(
                    [
                        nn.Conv2d(bev_channels, per_map_out, kernel_size=1)
                        for _ in range(num_bev_inputs)
                    ]
                )
            self.total_concat_channels = per_map_out * num_bev_inputs
            self.conv_layers = nn.Sequential(
                nn.Conv2d(
                    self.total_concat_channels,
                    hidden_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(
                    hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
                nn.Conv2d(
                    hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU(),
            )
        # Ensure output spatial matches out_h, out_w (handles non-power-of-2)
        if self.need_adaptive_pool:
            self.adaptive_pool = nn.AdaptiveAvgPool2d((self.out_h, self.out_w))

        if self.use_adapter:
            self.per_map_skip_add = nn.ModuleList(
                [nn.quantized.FloatFunctional() for _ in range(num_bev_inputs)]
            )
            self.per_map_adapter = nn.ModuleList(
                [
                    nn.Sequential(
                        ConvModule2d(
                            bev_channels,
                            bev_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            norm_layer=nn.BatchNorm2d(bev_channels),
                            act_layer=nn.ReLU(inplace=False),
                        ),
                        ConvModule2d(
                            bev_channels,
                            bev_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            norm_layer=nn.BatchNorm2d(bev_channels),
                            act_layer=None,
                        ),
                    )
                    for _ in range(num_bev_inputs)
                ]
            )

    def _get_bev_grid_positions(self, B, device):
        """Generate position embeddings for BEV grid."""
        y_coords = torch.linspace(-1, 1, self.out_h, device=device)
        x_coords = torch.linspace(-1, 1, self.out_w, device=device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        pos = torch.stack(
            [
                grid_x.flatten(),
                grid_y.flatten(),
                torch.ones(self.bev_token_num, device=device),
                torch.zeros(self.bev_token_num, device=device),
                torch.zeros(self.bev_token_num, device=device),
                torch.zeros(self.bev_token_num, device=device),
                torch.ones(self.bev_token_num, device=device),
            ],
            dim=-1,
        )
        pos = pos.unsqueeze(0).expand(B, -1, -1)
        return self.type_pos_quant(pos)

    def forward(self, *bev_inputs):
        """Args: *bev_inputs: any number of BEV maps, each (B, C, H, W)."""
        assert len(bev_inputs) > 0, "At least one BEV input is required"
        assert len(bev_inputs) == self.num_bev_inputs, (
            f"Expected {self.num_bev_inputs} BEV inputs, "
            f"got {len(bev_inputs)}"
        )
        B = bev_inputs[0].shape[0]
        device = bev_inputs[0].device

        if self.use_adapter:
            bev_inputs = tuple(
                skip_add.add(bev, adapter(bev))
                for bev, skip_add, adapter in zip(
                    bev_inputs, self.per_map_skip_add, self.per_map_adapter
                )
            )

        if self.num_bev_inputs == 1 or not self.reduce_channel_before_concat:
            bev_concat = torch.cat(bev_inputs, dim=1)  # B, N*C, H, W
        else:
            projected = []
            if self.share_per_map_proj:
                for bev in bev_inputs:
                    projected.append(self.per_map_proj(bev))
            else:
                for bev, proj in zip(bev_inputs, self.per_map_proj):
                    projected.append(proj(bev))
            bev_concat = torch.cat(projected, dim=1)
        bev_concat = self.res_quant(bev_concat)

        x = self.conv_layers(bev_concat)
        if self.need_adaptive_pool:
            x = self.adaptive_pool(x)  # B, hidden_dim, out_h, out_w
        encoding = x.flatten(2).permute(
            0, 2, 1
        )  # B, bev_token_num, hidden_dim

        # BEV has no padding, all valid
        mask = torch.zeros(
            B, self.bev_token_num, dtype=torch.bool, device=device
        )
        pos = self._get_bev_grid_positions(B, device)

        return encoding, mask, pos


@OBJECT_REGISTRY.register
class FusionEncoder(nn.Module):
    def __init__(
        self, hidden_dim=192, num_heads=6, drop_path_rate=0.3, depth=3
    ):
        super().__init__()

        dpr = drop_path_rate

        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(hidden_dim, num_heads, dropout=dpr)
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask):
        B, _ = mask.shape
        new_first_column = torch.full(
            (B, 1), False, dtype=torch.bool, device=mask.device
        )
        mask = torch.cat((new_first_column, mask[:, 1:]), dim=1)

        for b in self.blocks:
            x = b(x, mask)

        return self.norm(x)
