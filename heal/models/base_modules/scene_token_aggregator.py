# Copyright (c) Horizon Robotics. All rights reserved.
"""
Multi-View Scene Token Aggregator.

Pipeline:
    DINOv2 patch features
    → PixelShuffleReduction (2×2 spatial merge → channel ×4 → Linear)
    → view-position encoding + concat
    → cross-attention (learnable queries attend to all patches)
    → light self-attention transformer
    → scene tokens
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from hat_engine.registry import OBJECT_REGISTRY
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

__all__ = [
    "SceneProjector",
    "SceneTokenAggregator",
    "PixelShuffleReduction",
    "PlueckerRayEncoder",
]


class LightTransformerBlock(nn.Module):
    """Single pre-norm transformer block (self-attention + FFN)."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size)

        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        # Pre-norm FFN
        x = x + self.mlp(self.norm2(x))
        return x


@OBJECT_REGISTRY.register
class PixelShuffleReduction(nn.Module):
    """
    Spatial down-sampling via pixel-unshuffle + linear projection.

    Merges ``scale × scale`` spatial patches into one token with expanded
    channels, then projects back to ``hidden_size``.

    Input:  ``(B, H*W, C)``  + spatial dims ``(h, w)``
    Output: ``(B, H/s * W/s, C)``

    Args:
        hidden_size: Feature dim of input patches (e.g. 768 for DINOv2-base).
        scale: Spatial down-sample factor (default 2 → 4× token reduction).
    """

    def __init__(self, hidden_size: int = 768, scale: int = 2):
        super().__init__()
        self.scale = scale
        self.proj = nn.Linear(hidden_size * scale * scale, hidden_size)

    def forward(
        self,
        patch_features: torch.Tensor,
        h_patches: int,
        w_patches: int,
    ) -> Tuple[torch.Tensor, int, int]:
        """Reduce spatial patch tokens with a pixel-shuffle projection.

        Args:
            patch_features: ``(B, H*W, C)``
            h_patches: Spatial height in patches.
            w_patches: Spatial width in patches.

        Returns:
            reduced: ``(B, H'*W', C)`` where H'=H/scale, W'=W/scale.
            h_new: Reduced height.
            w_new: Reduced width.
        """
        s = self.scale
        B, _, C = patch_features.shape

        # Pad if not divisible by scale
        pad_h = (s - h_patches % s) % s
        pad_w = (s - w_patches % s) % s
        if pad_h > 0 or pad_w > 0:
            x = patch_features.view(B, h_patches, w_patches, C)
            x = nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            h_patches += pad_h
            w_patches += pad_w
            patch_features = x.view(B, h_patches * w_patches, C)

        h_new = h_patches // s
        w_new = w_patches // s

        # (B, H, W, C) → (B, H', s, W', s, C) → (B, H', W', s*s*C)
        x = patch_features.view(B, h_patches, w_patches, C)
        x = x.reshape(B, h_new, s, w_new, s, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(B, h_new * w_new, s * s * C)

        # Project back to hidden_size
        x = self.proj(x)
        return x, h_new, w_new


@OBJECT_REGISTRY.register
class PlueckerRayEncoder(nn.Module):
    """Project 6D Plücker ray coordinates to hidden_size via a small MLP."""

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(6, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, plucker: torch.Tensor) -> torch.Tensor:
        """Encode Plücker ray coordinates to hidden size.

        Args:
            plucker: ``(B, N, 6)`` Plücker ray coordinates.
        Returns:
            ``(B, N, hidden_size)``
        """
        plucker = plucker.to(dtype=self.mlp[0].weight.dtype)
        return self.norm(self.mlp(plucker))


@OBJECT_REGISTRY.register
class SceneProjector(nn.Module):
    """Project scene tokens from DINOv2 hidden size to the LLM embedding.

    Args:
        input_dim: DINOv2 feature dimension (768 for base, 1024 for large).
        output_dim: LLM hidden size (1024 for InternVL3.5-1B, 1536 for 2B).
        hidden_dim: Intermediate MLP width. Defaults to average of in/out.
    """

    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 1024,
        hidden_dim: Optional[int] = None,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (input_dim + output_dim) // 2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project scene tokens to the LLM embedding dimension.

        Args:
            x: ``(B, N, input_dim)``
        Returns:
            ``(B, N, output_dim)``
        """
        return self.mlp(x)


@OBJECT_REGISTRY.register
class SceneTokenAggregator(nn.Module):
    """
    Aggregates multi-view DINOv2 patch features into compact scene tokens.

    Pipeline:
        1. Add Plücker ray embeddings (requires camera calibration).
        2. Cross-attention: learnable scene queries attend to all view patches.
        3. Light Transformer (self-attention): refine scene tokens.

    Note: PixelShuffleReduction is applied upstream in VLMBackboneV2
    before features reach this module.

    Args:
        hidden_size: Feature dimension matching DINOv2 output.
        num_scene_tokens: Number of learnable output tokens.
        num_transformer_layers: Depth of the self-attention transformer.
        num_heads: Attention heads for both cross- and self-attention.
        dropout: Dropout probability.
        num_views: Expected number of camera views.
        effective_patch_size: Patch size in post-transform image space for
        rays.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_scene_tokens: int = 300,
        num_transformer_layers: int = 3,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_views: int = 3,
        effective_patch_size: int = 28,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_scene_tokens = num_scene_tokens
        self.num_views = num_views

        # Learnable scene query tokens
        self.scene_queries = nn.Parameter(
            torch.randn(1, num_scene_tokens, hidden_size) * 0.02
        )
        self.scene_queries_quant = QuantStub()

        # Cross-attention: scene queries (Q) ← all view patches (K, V)
        self.cross_attn_norm_q = nn.LayerNorm(hidden_size)
        self.cross_attn_norm_kv = nn.LayerNorm(hidden_size)
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )

        # Lightweight self-attention transformer for scene token refinement
        self.transformer_blocks = nn.ModuleList(
            [
                LightTransformerBlock(hidden_size, num_heads, dropout=dropout)
                for _ in range(num_transformer_layers)
            ]
        )

        # Plücker ray embedding is required (no learnable view-embedding
        # fallback).
        self.effective_patch_size = effective_patch_size
        self.ray_encoder = PlueckerRayEncoder(hidden_size)
        self.view_feats_quants = nn.ModuleList(
            [QuantStub() for _ in range(num_views)]
        )
        self.ray_embed_quants = nn.ModuleList(
            [QuantStub() for _ in range(num_views)]
        )
        self.dequant = DeQuantStub()

    # ------------------------------------------------------------------
    # Plücker ray computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_plucker_rays(
        intrinsics: torch.Tensor,
        rotation: torch.Tensor,
        translation: torch.Tensor,
        h_patches: int,
        w_patches: int,
        patch_size: int = 14,
    ) -> torch.Tensor:
        """
        Compute Plücker ray coordinates for each patch in ego/lidar frame.

        Args:
            intrinsics:  ``(B, 3, 3)`` camera intrinsic matrix K.
            rotation:    ``(B, 3, 3)`` sensor-to-lidar rotation R.
            translation: ``(B, 3)``   camera origin in lidar frame.
            h_patches:   Number of patches along height.
            w_patches:   Number of patches along width.
            patch_size:  Pixel size of each patch (default 14 for DINOv2).

        Returns:
            ``(B, h_patches*w_patches, 6)`` Plücker vectors ``(d, o × d)``.
        """
        # Force float32: linalg.inv does not support low-precision dtypes
        intrinsics = intrinsics.float()
        rotation = rotation.float()
        translation = translation.float()

        B = intrinsics.shape[0]
        device = intrinsics.device

        # Patch centre pixel coordinates (post-transform image space).
        # Match HAT: use float half-patch (patch_size / 2), not integer //.
        ys = (
            torch.arange(h_patches, device=device, dtype=torch.float32)
            * patch_size
            + patch_size / 2
        )
        xs = (
            torch.arange(w_patches, device=device, dtype=torch.float32)
            * patch_size
            + patch_size / 2
        )
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)
        ones = torch.ones_like(grid_x)
        pts = torch.stack([grid_x, grid_y, ones], dim=-1)  # (H, W, 3)
        pts = pts.view(1, -1, 3).expand(B, -1, -1)  # (B, N, 3)

        # K from data is in *original* image pixel space, but ``pts`` are
        # patch centres in *post-transform* space (after aspect-preserving
        # resize + pad). Without rescaling, K⁻¹ @ pts yields rays with the
        # wrong FOV. Estimate scale from principal point vs post size
        # (same as HAT SceneTokenAggregator.compute_plucker_rays).
        post_W = float(w_patches * patch_size)
        post_H = float(h_patches * patch_size)
        cx = intrinsics[:, 0, 2].clamp(min=1.0)
        cy = intrinsics[:, 1, 2].clamp(min=1.0)
        s = torch.maximum(post_W / (2.0 * cx), post_H / (2.0 * cy))  # (B,)
        K_scaled = intrinsics.clone()
        K_scaled[:, 0, :] = K_scaled[:, 0, :] * s.view(B, 1)
        K_scaled[:, 1, :] = K_scaled[:, 1, :] * s.view(B, 1)

        # Back-project through K⁻¹: pixel → camera direction
        K_inv = torch.linalg.inv(K_scaled)  # (B, 3, 3)
        d_cam = torch.bmm(pts, K_inv.transpose(1, 2))  # (B, N, 3)

        # Rotate to ego/lidar frame and normalise
        d_ego = torch.bmm(d_cam, rotation.transpose(1, 2))  # (B, N, 3)
        d_ego = F.normalize(d_ego, dim=-1)

        # Ray origin = camera position in ego frame
        o = translation.unsqueeze(1).expand_as(d_ego)  # (B, N, 3)

        # Plücker coordinates: (direction, moment = origin × direction)
        moment = torch.cross(o, d_ego, dim=-1)  # (B, N, 3)
        return torch.cat([d_ego, moment], dim=-1)  # (B, N, 6)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _add_ray_pos_encoding(
        self,
        view_features_list: List[torch.Tensor],
        ray_embeds_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Add Plücker ray position embeddings then concatenate all views.

        Args:
            view_features_list: List of ``(B, N_i, hidden_size)`` tensors,
                one per camera view.
            ray_embeds_list: List of ``(B, N_i, hidden_size)`` ray
                embeddings per view.

        Returns:
            ``(B, sum(N_i), hidden_size)`` concatenated tensor.
        """
        encoded = []
        for view_idx, feat in enumerate(view_features_list):
            if view_idx >= self.num_views:
                raise ValueError(
                    f"View index {view_idx} exceeds configured num_views "
                    f"{self.num_views}."
                )
            encoded.append(feat + ray_embeds_list[view_idx])
        return torch.cat(encoded, dim=1)  # (B, total_patches, hidden_size)

    def forward(
        self,
        view_features_list: List[torch.Tensor],
        spatial_dims_list: Optional[List[Tuple[int, int]]] = None,
        camera_params_list: Optional[List[Optional[Dict]]] = None,
    ) -> torch.Tensor:
        """Aggregate multi-view patch features into scene tokens.

        Args:
            view_features_list: Patch features from each camera view,
                each shaped ``(B, N_patches_i, hidden_size)``.
                PixelShuffleReduction (if any) is already applied upstream.
            spatial_dims_list: ``[(h_patches, w_patches), ...]`` per view
                (post-shuffle dims). Required for Plücker rays.
            camera_params_list: List of dicts per view with keys
                ``"intrinsics"`` ``(B,3,3)``, ``"rotation"`` ``(B,3,3)``,
                ``"translation"`` ``(B,3)``. Required for Plücker rays.

        Returns:
            scene_tokens: ``(B, num_scene_tokens, hidden_size)``
        """
        batch_size = view_features_list[0].shape[0]

        if camera_params_list is None:
            raise ValueError(
                "camera_params_list is required for Plücker ray embedding. "
                "All datasets must provide camera calibration."
            )
        if spatial_dims_list is None:
            raise ValueError(
                "spatial_dims_list is required for Plücker ray embedding"
            )

        ray_embeds_list = []
        for view_idx, (feat, (h, w), params) in enumerate(
            zip(view_features_list, spatial_dims_list, camera_params_list)
        ):
            if params is None:
                raise ValueError(
                    f"camera_params_list[{view_idx}] is None. "
                    "Every view must have camera calibration."
                )
            plucker = self.compute_plucker_rays(
                params["intrinsics"],
                params["rotation"],
                params["translation"],
                h,
                w,
                patch_size=self.effective_patch_size,
            )  # (B, N, 6)
            plucker = plucker.to(feat.dtype)

            ray_emb = self.ray_encoder(plucker)  # (B, N, D)
            ray_embeds_list.append(ray_emb)

        view_features_list = [
            self.view_feats_quants[i](feat)
            for i, feat in enumerate(view_features_list)
        ]
        ray_embeds_list = [
            self.ray_embed_quants[i](emb)
            for i, emb in enumerate(ray_embeds_list)
        ]
        all_patches = self._add_ray_pos_encoding(
            view_features_list, ray_embeds_list
        )

        # Step 3: cross-attention – scene queries attend to all patches
        queries = self.scene_queries_quant(
            self.scene_queries.expand(batch_size, -1, -1)
        )
        scene_tokens_ca, _ = self.cross_attn(
            self.cross_attn_norm_q(queries),  # Q (pre-norm)
            self.cross_attn_norm_kv(all_patches),  # K
            all_patches,  # V
        )
        scene_tokens = queries + scene_tokens_ca  # residual

        # Step 4: refine with light transformer
        for block in self.transformer_blocks:
            scene_tokens = block(scene_tokens)

        return self.dequant(scene_tokens)
