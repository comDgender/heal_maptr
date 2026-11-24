# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
"""BEVFormer view transformer: image features -> dense BEV embedding.

Responsibilities:

1. **Reference points**: 3D pillar anchors in BEV and 2D refs for TSA; project
   3D refs to each camera for SCA (``point_sampling``).
2. **Rebatch grids**: pack visible BEV pillars per camera into a dense virtual
   grid for efficient SCA (``queries_rebatch_grid`` / ``restore_bev_grid``).
3. **Temporal fusion**: warp ``prev_bev`` with ego motion, iterate over
   ``queue_length`` frames during training (``get_fusion_bev``).
4. **BEV encoder**: learnable BEV query embedding + ``BEVFormerEncoder``.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.model_helpers import fx_wrap
from horizon_plugin_pytorch.quantization import QuantStub
from torch import Tensor, nn
from torch.amp import autocast
from torch.quantization import DeQuantStub

from heal.models.task_modules.bevformer.utils import (
    gen_coords,
    get_min_max_coords,
)

logger = logging.getLogger(__name__)

__all__ = ["BevFormerViewTransformer", "SingleFrameBevFormerViewTransformer"]


@OBJECT_REGISTRY.register
class BevFormerViewTransformer(nn.Module):
    """Lift multi-view FPN features to a dense BEV feature map.

    Wraps ``BEVFormerEncoder`` with BEV query/pos embeddings, reference-point
    generation, camera projection, and optional temporal queue fusion.

    Args:
        pc_range: Point cloud / VCS bounds [x_min, y_min, z_min, x_max, ...].
        num_points_in_pillar: Z anchors per BEV pillar for 3D reference points.
        bev_h, bev_w: BEV grid resolution.
        embed_dims: Transformer hidden dimension.
        queue_length: Number of temporal frames in training queue.
        max_camoverlap_num: Max cameras observing the same BEV pillar.
        virtual_bev_h, virtual_bev_w: Dense grid size for SCA rebatch.
        encoder: ``BEVFormerEncoder`` module.
        positional_encoding: Sinusoidal / learned BEV positional encoding.
        numcam: Number of surround cameras.
        in_indices: Which FPN levels feed the encoder.
        is_compile: Export/deploy mode (external prev_bev + grids).
        single_bev: Disable temporal queue (single-frame variant).
        use_lidar2img: Use lidar2img instead of ego2img for projection.
    """

    def __init__(
        self,
        pc_range: List[float],
        num_points_in_pillar: int,
        bev_h: int,
        bev_w: int,
        embed_dims: int,
        queue_length: int,
        max_camoverlap_num: int,
        virtual_bev_h: int,
        virtual_bev_w: int,
        encoder: nn.Module,
        positional_encoding: nn.Module,
        numcam: int = 6,
        in_indices: List[int] = (2,),
        is_compile: bool = False,
        single_bev: bool = False,
        use_lidar2img: bool = False,
    ):
        super(BevFormerViewTransformer, self).__init__()
        self.in_indices = in_indices
        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.bev_size = (self.bev_w, self.bev_h)
        self.grid_resolution = (
            self.real_w / self.bev_w,
            self.real_h / self.bev_h,
        )
        self.embed_dims = embed_dims
        self.is_compile = is_compile
        self.queue_length = queue_length
        self.numcam = numcam
        self.encoder = encoder
        self.positional_encoding = positional_encoding

        self.bev_embedding = nn.Embedding(
            self.bev_h * self.bev_w, self.embed_dims
        )

        # QuantStubs mark tensor boundaries for Horizon QAT / HBIR export.
        self.quant_bev_pos = QuantStub()
        self.quant_bev_query = QuantStub()
        self.quant_prev_bev = QuantStub()
        self.quant_hybird_ref_2d = QuantStub()
        self.quant_norm_coords = QuantStub()
        self.dequant = DeQuantStub()
        self.quant_queries_rebatch_grid = QuantStub()
        self.quant_restore_bev_grid = QuantStub()
        self.quant_reference_points_rebatch = QuantStub()
        self.quant_bev_pillar_counts = QuantStub()

        self.single_bev = single_bev
        self.use_lidar2img = use_lidar2img
        self.global_pose_key = (
            "lidar2global" if use_lidar2img else "ego2global"
        )
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            self.global_pose_key: None,
        }

        self.max_camoverlap_num = max_camoverlap_num
        self.virtual_bev_h = virtual_bev_h
        self.virtual_bev_w = virtual_bev_w

    def get_reference_points_cam(self, data):
        B = data["img"].shape[0] // self.numcam
        _, ref3d = self.export_reference_points(B, device=data["img"].device)
        (
            reference_points_rebatch,
            queries_rebatch_grid,
            restore_bev_grid,
            bev_pillar_counts,
        ) = self.point_sampling(
            ref3d, self.pc_range, data["seq_meta"][0], data["img"].shape[2:]
        )
        return (
            queries_rebatch_grid,
            restore_bev_grid,
            reference_points_rebatch,
            bev_pillar_counts,
        )

    def get_deploy_prev(self, data, prev_frame_info):
        prev_bev = prev_frame_info["prev_bev"]
        prev_meta = prev_frame_info
        if prev_bev is None:
            prev_bev = torch.zeros(
                (1, self.bev_h * self.bev_w, self.embed_dims),
                dtype=torch.float32,
            )
            prev_meta = None
        if prev_meta is not None:
            pre_scene = prev_meta["scene_token"]
            cur_scene = data["seq_meta"][0]["meta"][0]["scene"]
            if pre_scene != cur_scene:
                prev_bev = torch.zeros(
                    (1, self.bev_h * self.bev_w, self.embed_dims),
                    dtype=torch.float32,
                )
            prev_meta["scene_token"] = [prev_meta["scene_token"]]

        norm_coords = self.export_bev_transform_points(
            data["seq_meta"][0], prev_meta
        )
        return prev_bev, norm_coords

    @fx_wrap()
    def gen_reference_points(
        self,
        H: int,
        W: int,
        Z: int = 8,
        num_points_in_pillar: int = 4,
        dim: str = "3d",
        bs: int = 1,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Get the reference points used in SCA and TSA."""

        # reference points in 3D space.
        if dim == "3d":
            zs = (
                torch.linspace(
                    0.5,
                    Z - 0.5,
                    num_points_in_pillar,
                    dtype=dtype,
                    device=device,
                )
                .view(-1, 1, 1)
                .expand(num_points_in_pillar, H, W)
                / Z
            )
            xs = (
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
                .view(1, 1, W)
                .expand(num_points_in_pillar, H, W)
                / W
            )
            # Pillar cell centers along H, broadcast to (D, H, W).
            y_axis = (torch.arange(H, dtype=dtype, device=device) + 0.5) / H
            ys = y_axis.view(1, H, 1).expand(num_points_in_pillar, H, W)
            # (D, H, W, 3) -> (B, D, H*W, 3)
            ref_3d = torch.stack([xs, ys, zs], dim=-1).reshape(
                num_points_in_pillar, H * W, 3
            )
            return ref_3d.unsqueeze(0).repeat(bs, 1, 1, 1)

        # reference points on 2D bev plane.
        elif dim == "2d":
            # Normalized query coords at BEV cell centers (ij mesh order).
            y_norm = (torch.arange(H, dtype=dtype, device=device) + 0.5) / H
            x_norm = (torch.arange(W, dtype=dtype, device=device) + 0.5) / W
            grid_y, grid_x = torch.meshgrid(y_norm, x_norm, indexing="ij")
            ref_2d = torch.stack(
                (grid_x.reshape(1, -1), grid_y.reshape(1, -1)), dim=-1
            )
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    @autocast("cuda", enabled=False)
    @fx_wrap()
    def point_sampling(
        self,
        reference_points: Tensor,
        pc_range: List[float],
        img_metas: Dict,
        im_shape: Tuple[int],
    ) -> Tuple[Tensor, Tensor]:
        """Project 3D BEV reference points to image plane and build SCA grids.

        For each camera, collects visible BEV pillars, packs them into a
        ``virtual_bev_h x virtual_bev_w`` grid (``queries_rebatch_grid``), and
        builds ``restore_bev_grid`` to scatter multi-cam results back to BEV.
        ``bev_pillar_counts`` stores per-pillar averaging weights.
        """

        reference_points = reference_points.to(torch.float32)
        if self.use_lidar2img:
            ego2img = img_metas["lidar2img"]
        else:
            ego2img = img_metas["ego2img"]
        ego2img = np.asarray(ego2img)
        ego2img = reference_points.new_tensor(ego2img)  # (B, N, 4, 4)
        # Map normalized BEV pillars into metric coordinates, then project.
        pts = reference_points.clone()
        extent = pts.new_tensor(
            [
                pc_range[3] - pc_range[0],
                pc_range[4] - pc_range[1],
                pc_range[5] - pc_range[2],
            ]
        )
        origin = pts.new_tensor([pc_range[0], pc_range[1], pc_range[2]])
        pts = pts * extent + origin
        pts_h = torch.cat([pts, torch.ones_like(pts[..., :1])], dim=-1)

        # Layout: depth samples x batch x queries, broadcast over cameras.
        pts_h = pts_h.permute(1, 0, 2, 3)
        D, B, num_query = pts_h.shape[:3]
        num_cam = ego2img.shape[1]
        pts_h = (
            pts_h.reshape(D, B, 1, num_query, 4, 1)
            .expand(D, B, num_cam, num_query, 4, 1)
            .clone()
        )
        cam2img = (
            ego2img.reshape(1, B, num_cam, 1, 4, 4)
            .expand(D, B, num_cam, num_query, 4, 4)
            .to(torch.float32)
        )
        cam_pts = torch.matmul(cam2img, pts_h.to(torch.float32)).squeeze(-1)

        eps = 1e-5
        depth = cam_pts[..., 2:3]
        on_front = depth > eps
        # Perspective divide, then normalize pixel coords into [0, 1].
        xy = cam_pts[..., :2] / depth.clamp(min=eps)
        xy = torch.stack(
            (xy[..., 0] / im_shape[1], xy[..., 1] / im_shape[0]), dim=-1
        )
        in_frame = (
            (xy[..., 0:1] > 0.0)
            & (xy[..., 0:1] < 1.0)
            & (xy[..., 1:2] > 0.0)
            & (xy[..., 1:2] < 1.0)
        )
        bev_mask = torch.nan_to_num(on_front & in_frame)
        reference_points_cam = xy.clamp(min=-2.1, max=2.1)
        # Camera-major layout expected by the rebatch loops below.
        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        bev_mask_ori = bev_mask.clone()
        max_len = self.virtual_bev_h * self.virtual_bev_w
        # Per-camera dense grid: maps virtual cell -> BEV pillar index.
        queries_rebatch_grid = reference_points_cam.new_zeros(
            [B * self.numcam, self.virtual_bev_h, self.virtual_bev_w, 2]
        )
        for camera_idx, mask_per_img_bs in enumerate(bev_mask):
            for bs_id, mask_per_img in enumerate(mask_per_img_bs):
                temp_grid = (
                    torch.zeros(
                        (max_len, 2),
                        device=queries_rebatch_grid.device,
                        dtype=torch.float32,
                    )
                    - 1.5
                )
                index_query_per_img = (
                    mask_per_img.sum(-1).nonzero().squeeze(-1)
                )
                num_bev_points = index_query_per_img.shape[0]

                camera_idx_tensor_x = index_query_per_img % self.bev_w
                camera_idx_tensor_y = index_query_per_img // self.bev_w

                index_grid = torch.stack(
                    [
                        camera_idx_tensor_x / (self.bev_w - 1),
                        camera_idx_tensor_y / (self.bev_h - 1),
                    ],
                    dim=-1,
                )
                index_grid = index_grid * 2 - 1
                temp_grid[:num_bev_points] = index_grid
                temp_grid = temp_grid.reshape(
                    self.virtual_bev_h, self.virtual_bev_w, 2
                )
                queries_rebatch_grid[bs_id * self.numcam + camera_idx] = (
                    temp_grid
                )

        reference_points_rebatch = (
            reference_points_cam.flatten(-2)
            .permute(1, 0, 3, 2)
            .flatten(0, 1)
            .reshape(B * self.numcam, D * 2, self.bev_h, self.bev_w)
        )

        reference_points_rebatch = (
            F.grid_sample(
                reference_points_rebatch,
                queries_rebatch_grid,
                mode="nearest",
                align_corners=True,
            )
            .flatten(-2)
            .permute(0, 2, 1)
            .reshape(B * self.numcam, max_len, D, 2)
        )
        bev_mask = bev_mask.permute(1, 0, 2, 3)
        # Scatter grid: maps BEV pillar -> (cam_id, virtual_cell) for restore.
        restore_bev_grid = (
            reference_points_cam.new_zeros(
                B, self.max_camoverlap_num * self.bev_h, self.bev_w, 2
            )
            - 1.5
        )
        for bs_id, bev_mask_ in enumerate(bev_mask):
            bev_pillar_num_map = torch.zeros(
                (self.bev_h, self.bev_w), device=bev_mask_.device
            )
            count = bev_mask_.sum(-1) > 0
            camera_idxs, bev_pillar_idxs = torch.where(count)
            camera_idx_offset = 0

            for cam_id in range(self.numcam):
                camera_idx = torch.where(camera_idxs == cam_id)
                bev_pillar_idx_cam = bev_pillar_idxs[camera_idx[0]]
                num_camera_idx = len(camera_idx[0])

                camera_idx_tmp = camera_idx[0] - camera_idx_offset
                camare_tmp_idx_x = camera_idx_tmp % self.virtual_bev_w
                camare_tmp_idx_y = camera_idx_tmp // self.virtual_bev_w
                grid_x = camare_tmp_idx_x
                grid_y = cam_id * self.virtual_bev_h + camare_tmp_idx_y

                bev_pillar_idx_cam_x = bev_pillar_idx_cam % self.bev_w
                bev_pillar_idx_cam_y = bev_pillar_idx_cam // self.bev_w
                bev_pillar_num_map_tmp = bev_pillar_num_map[
                    bev_pillar_idx_cam_y, bev_pillar_idx_cam_x
                ]
                grid_h = (
                    bev_pillar_num_map_tmp * self.bev_h + bev_pillar_idx_cam_y
                ).to(torch.int64)
                grid_w = (bev_pillar_idx_cam_x).to(torch.int64)

                restore_bev_grid[bs_id, grid_h, grid_w, 0] = grid_x / (
                    self.virtual_bev_w - 1
                )
                restore_bev_grid[bs_id, grid_h, grid_w, 1] = grid_y / (
                    self.numcam * self.virtual_bev_h - 1
                )

                bev_pillar_num_map[
                    bev_pillar_idx_cam_y, bev_pillar_idx_cam_x
                ] = (
                    bev_pillar_num_map[
                        bev_pillar_idx_cam_y, bev_pillar_idx_cam_x
                    ]
                    + 1
                )
                camera_idx_offset = camera_idx_offset + num_camera_idx

        restore_bev_grid = restore_bev_grid * 2 - 1

        bev_pillar_counts = bev_mask_ori.sum(-1) > 0
        bev_pillar_counts = bev_pillar_counts.permute(1, 2, 0).sum(-1)
        bev_pillar_counts = torch.clamp(bev_pillar_counts, min=1.0)
        bev_pillar_counts = 1.0 / bev_pillar_counts[..., None]

        return (
            reference_points_rebatch,
            queries_rebatch_grid,
            restore_bev_grid,
            bev_pillar_counts,
        )

    @fx_wrap()
    def export_reference_points(
        self, bs: int, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> Tuple[Tensor, Tensor]:
        """Export reference points."""
        ref_3d = self.gen_reference_points(
            self.bev_h,
            self.bev_w,
            self.pc_range[5] - self.pc_range[2],
            self.num_points_in_pillar,
            dim="3d",
            bs=bs,
            device=device,
            dtype=dtype,
        )
        ref_2d = self.gen_reference_points(
            self.bev_h, self.bev_w, dim="2d", bs=bs, device=device, dtype=dtype
        )

        return ref_2d, ref_3d

    @fx_wrap()
    def set_eval(self, seq_feats: List[List[Tensor]]) -> List[List[Tensor]]:
        """Set model to eval mode."""
        self.eval()
        return seq_feats

    @fx_wrap()
    def set_train(self, seq_feats: List[List[Tensor]]) -> List[List[Tensor]]:
        """Set model to train mode."""
        self.train()
        return seq_feats

    def get_cur_scene_token(self, data: List[Dict], i: int) -> List[str]:
        """Get scene tokens."""
        return [each["scene"] for each in data[i]["meta"]]

    @fx_wrap()
    def update_pre_info(
        self,
        seq_meta: List[Dict],
        bev_embed: Tensor,
        i: int,
        seq_feats: List[List[Tensor]],
    ) -> List[List[Tensor]]:
        """Update prev bev info."""
        self.prev_frame_info["scene_token"] = self.get_cur_scene_token(
            seq_meta, i
        )
        self.prev_frame_info[self.global_pose_key] = seq_meta[i][
            self.global_pose_key
        ]
        self.prev_frame_info["prev_bev"] = self.dequant(bev_embed.detach())
        return seq_feats

    def get_fusion_bev(
        self, data: Dict, seq_feats: List[List[Tensor]]
    ) -> Tensor:
        """Fuse ``queue_length`` frames: older frames update ``prev_bev`` first."""  # noqa: E501
        seq_meta = data["seq_meta"]
        im_shape = data["img"].shape[2:]
        if self.single_bev:
            assert self.queue_length == 1
            return self.get_bev_embed(seq_meta, 0, seq_feats, im_shape)
        for i in range(self.queue_length - 1, 0, -1):
            seq_feats = self.set_eval(seq_feats)
            bev_emb = self.get_bev_embed(seq_meta, i, seq_feats, im_shape)
            # Propagate BEV state to the next (earlier) frame in the queue.
            seq_feats = self.update_pre_info(seq_meta, bev_emb, i, seq_feats)
            seq_feats = self.set_train(seq_feats)

        bev_emb = self.get_bev_embed(seq_meta, 0, seq_feats, im_shape)
        bev_emb = self.update_pre_info(seq_meta, bev_emb, 0, bev_emb)
        return bev_emb

    def get_bev_embed(
        self,
        seq_meta: List[Dict],
        i: int,
        seq_feats: List[List[Tensor]],
        im_shape: Tuple[int, int],
    ) -> Tensor:
        """Get bev feat."""
        feats = seq_feats[i]
        bs = feats[0].shape[0]
        device = feats[0].device

        prev_bev, prev_meta = self.get_prev_bev(i, bs, device, seq_meta[i])
        ref2d, ref3d = self.export_reference_points(bs, device=device)
        (
            reference_points_rebatch,
            queries_rebatch_grid,
            restore_bev_grid,
            bev_pillar_counts,
        ) = self.point_sampling(ref3d, self.pc_range, seq_meta[i], im_shape)

        hybird_ref_2d = self.get_fusion_ref(ref2d)
        norm_coords = self.export_bev_transform_points(
            seq_meta[i], prev_meta
        ).to(device)

        bev_emb = self.bev_encoder(
            feats,
            prev_bev,
            hybird_ref_2d,
            norm_coords,
            queries_rebatch_grid,
            restore_bev_grid,
            reference_points_rebatch,
            bev_pillar_counts,
        )

        return bev_emb

    @fx_wrap()
    def get_fusion_ref(self, ref_2d: Tensor) -> Tensor:
        """Build hybrid 2D refs for TSA: [shifted_prev, current] stacked."""
        shift_ref_2d = ref_2d.clone()
        bs, len_bev, num_bev_level, _ = ref_2d.shape
        hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(
            bs * 2, len_bev, num_bev_level, 2
        )
        return hybird_ref_2d

    @fx_wrap()
    def get_prev_bev(
        self, idx: int, bs: int, device: torch.device, cur_meta: Dict
    ) -> Tuple[Tensor, Dict]:
        """Get prev bevfeat and metas."""
        if idx == self.queue_length - 1 and self.queue_length != 1:
            prev_bev = torch.zeros(
                (bs, self.bev_h * self.bev_w, self.embed_dims),
                dtype=torch.float32,
                device=device,
            )
            prev_meta = None
        else:
            prev_bev = self.prev_frame_info["prev_bev"]
            prev_meta = self.prev_frame_info
            if prev_bev is None:
                prev_bev = torch.zeros(
                    (bs, self.bev_h * self.bev_w, self.embed_dims),
                    dtype=torch.float32,
                    device=device,
                )
                prev_meta = None
            else:
                prev_bev = prev_bev.to(device)

        if prev_meta is not None:
            pre_scene = prev_meta["scene_token"]
            for i in range(bs):
                if pre_scene[i] != cur_meta["meta"][i]["scene"]:
                    prev_bev[i] = torch.zeros(
                        (self.bev_h * self.bev_w, self.embed_dims),
                        dtype=torch.float32,
                        device=device,
                    )
        return prev_bev, prev_meta

    @fx_wrap()
    def export_bev_transform_points(
        self, cur_meta: Dict, pre_meta: Dict
    ) -> Tensor:
        """Get normed coords for bevfeat transformer."""
        bs = len(cur_meta["meta"])
        real_range = (
            self.pc_range[0],
            self.pc_range[1],
            self.pc_range[3],
            self.pc_range[4],
        )

        coords = gen_coords(self.bev_size, self.pc_range)

        bev_min_x, bev_max_x, bev_min_y, bev_max_y = get_min_max_coords(
            real_range,
            self.grid_resolution,
        )

        bs_new_coords = []
        if pre_meta is not None:
            pre_scene = pre_meta["scene_token"]
            for i in range(bs):
                if pre_scene[i] != cur_meta["meta"][i]["scene"]:
                    new_coords = coords.clone()
                else:
                    new_coords = self.get_refine_coords(
                        coords, cur_meta, pre_meta, i
                    )
                bs_new_coords.append(new_coords)
        else:
            new_coords = coords.clone().repeat(bs, 1, 1, 1)
            bs_new_coords.append(new_coords)

        bs_new_coords = torch.cat(bs_new_coords, dim=0)
        bs_new_coords[..., 0] = (bs_new_coords[..., 0] - bev_min_x) / (
            (bev_max_x - bev_min_x)
        )
        bs_new_coords[..., 1] = (bs_new_coords[..., 1] - bev_min_y) / (
            (bev_max_y - bev_min_y)
        )

        norm_coords = (bs_new_coords * 2 - 1).to(torch.float32)
        return norm_coords

    def forward(
        self,
        feats: List[Tensor],
        data: Dict,
    ) -> Tensor:
        """Forward: training uses temporal queue; compile uses pre-built inputs."""  # noqa: E501
        if not self.is_compile:
            # Training: split queue dim, fuse history, encode current frame.
            seq_feats = self.get_seq_feats(feats)
            bev_emb = self.get_fusion_bev(data, seq_feats)
        else:
            # Deploy: prev_bev / grids supplied by BevFormerIrInfer.process_input.  # noqa: E501
            B = feats[0].shape[0] // self.numcam
            ref2d, _ = self.export_reference_points(B, device=feats[0].device)
            feats = self.get_deploy_feats(feats)
            hybird_ref_2d = self.get_fusion_ref(ref2d)
            prev_bev = data["prev_bev"]
            prev_bev_ref = data["prev_bev_ref"]
            queries_rebatch_grid = data["queries_rebatch_grid"]
            restore_bev_grid = data["restore_bev_grid"]
            reference_points_rebatch = data["reference_points_rebatch"]
            bev_pillar_counts = data["bev_pillar_counts"]

            bev_emb = self.bev_encoder(
                feats,
                prev_bev,
                hybird_ref_2d,
                prev_bev_ref,
                queries_rebatch_grid,
                restore_bev_grid,
                reference_points_rebatch,
                bev_pillar_counts,
            )
        return bev_emb

    @fx_wrap()
    def get_deploy_feats(self, feats: List[Tensor]) -> List[Tensor]:
        """Get deploy feats for view transformer inputs."""
        feats = [feats[i] for i in self.in_indices]
        for idx, feat in enumerate(feats):
            _, c, h, w = feat.shape
            feats[idx] = feat.view(-1, self.numcam, c, h, w)
        return feats

    @fx_wrap()
    def get_seq_feats(self, feats: List[Tensor]) -> List[List[Tensor]]:
        """Get seq feats for view transformer inputs."""
        feats = [feats[i] for i in self.in_indices]
        seq_feats = [[] for _ in range(self.queue_length)]
        for _, feat in enumerate(feats):
            _, c, h, w = feat.shape
            tmp_feat = feat.view(self.queue_length, -1, self.numcam, c, h, w)
            for i in range(self.queue_length):
                feat_i = tmp_feat[i]
                seq_feats[i].append(feat_i)
        return seq_feats

    @fx_wrap()
    def get_bev_pos(self, bs: int, device: torch.device) -> Tensor:
        """Get bev positional encoding."""
        bev_mask_pos = torch.zeros(
            (bs, self.bev_h, self.bev_w), device=device
        ).to(torch.float32)
        bev_pos = self.positional_encoding(bev_mask_pos).to(torch.float32)
        return bev_pos

    def bev_encoder(
        self,
        mlvl_feats: List[Tensor],
        prev_bev: Tensor,
        hybird_ref_2d: Tensor,
        norm_coords: Tensor,
        queries_rebatch_grid: Tensor,
        restore_bev_grid: Tensor,
        reference_points_rebatch: Tensor,
        bev_pillar_counts: Tensor,
    ) -> Tensor:
        """Run BEVFormerEncoder: query embed + prev BEV warp + encoder stack."""  # noqa: E501
        bs = mlvl_feats[0].shape[0]
        device = mlvl_feats[0].device
        bev_query = self.bev_embedding.weight
        bev_query = bev_query.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = self.get_bev_pos(bs, device)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        bev_query = self.quant_bev_query(bev_query)
        bev_pos = self.quant_bev_pos(bev_pos)
        hybird_ref_2d = self.quant_hybird_ref_2d(hybird_ref_2d)
        prev_bev = self.quant_prev_bev(prev_bev)
        norm_coords = self.quant_norm_coords(norm_coords)

        # Ego-motion align previous BEV to current frame (bilinear grid_sample).  # noqa: E501
        tmp_prev_bev = prev_bev.reshape(
            bs, self.bev_h, self.bev_w, self.embed_dims
        ).permute(0, 3, 1, 2)
        prev_bev = F.grid_sample(
            tmp_prev_bev, norm_coords, "bilinear", "zeros", True
        )
        prev_bev = prev_bev.reshape(
            bs, self.embed_dims, self.bev_h * self.bev_w
        ).permute(0, 2, 1)

        queries_rebatch_grid = self.quant_queries_rebatch_grid(
            queries_rebatch_grid
        )
        restore_bev_grid = self.quant_restore_bev_grid(restore_bev_grid)
        reference_points_rebatch = self.quant_reference_points_rebatch(
            reference_points_rebatch
        )
        bev_pillar_counts = self.quant_bev_pillar_counts(bev_pillar_counts)

        bev_embed = self.encoder(
            bev_query,
            mlvl_feats,
            bev_pos,
            prev_bev,
            hybird_ref_2d,
            queries_rebatch_grid,
            restore_bev_grid,
            reference_points_rebatch,
            bev_pillar_counts,
        )
        return bev_embed

    @fx_wrap()
    def get_refine_coords(
        self,
        coords: Tensor,
        img_metas: Dict,
        img_metas_pre: Dict,
        idx: int,
    ):
        """Get refine coords."""
        cur_e2g = img_metas[self.global_pose_key][idx]
        cur_e2g = np.array(cur_e2g)
        prev_e2g = img_metas_pre[self.global_pose_key][idx]
        prev_e2g = np.array(prev_e2g)
        prev_g2e = np.linalg.inv(prev_e2g)
        wrap_m = prev_g2e @ cur_e2g
        wrap_m = wrap_m[None, ...]
        wrap_r_t = wrap_m[:, :2, :3]
        # Extract the translation component
        trans = np.eye(3)[None, :, :]
        trans[:, :2, :3] = wrap_r_t
        output = trans.transpose((0, 2, 1))
        wrap_r1 = torch.tensor(output, dtype=torch.float32)
        ones = torch.ones((1, self.bev_h, self.bev_w, 1), dtype=torch.float32)
        coords_3d = torch.cat([coords, ones], dim=-1)
        new_coords_all = []
        batch = wrap_r1.shape[0]
        for i in range(batch):
            new_coord = torch.matmul(coords_3d, wrap_r1[i].double()).float()
            new_coords_all.append(new_coord)
        new_coords_all = torch.cat(new_coords_all)
        return new_coords_all[..., :2]


@OBJECT_REGISTRY.register
class SingleFrameBevFormerViewTransformer(BevFormerViewTransformer):
    """BevFormer view transformer without temporal BEV fusion (single frame)."""  # noqa: E501

    def get_bev_embed(
        self,
        feats: List[List[Tensor]],
        ref2d: Tensor,
        queries_rebatch_grid: Tensor,
        restore_bev_grid: Tensor,
        reference_points_rebatch: Tensor,
        bev_pillar_counts: Tensor,
    ) -> Tensor:
        """Get bev feat."""
        bs = feats[0].shape[0]
        device = feats[0].device

        bev_query = self.bev_embedding.weight
        bev_query = bev_query.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = self.get_bev_pos(bs, device)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        bev_query = self.quant_bev_query(bev_query)
        bev_pos = self.quant_bev_pos(bev_pos)
        ref2d = self.quant_hybird_ref_2d(ref2d)
        queries_rebatch_grid = self.quant_queries_rebatch_grid(
            queries_rebatch_grid
        )
        restore_bev_grid = self.quant_restore_bev_grid(restore_bev_grid)
        reference_points_rebatch = self.quant_reference_points_rebatch(
            reference_points_rebatch
        )
        bev_pillar_counts = self.quant_bev_pillar_counts(bev_pillar_counts)

        bev_embed = self.encoder(
            bev_query,
            feats,
            bev_pos,
            ref2d,
            queries_rebatch_grid,
            restore_bev_grid,
            reference_points_rebatch,
            bev_pillar_counts,
        )
        return bev_embed

    def forward(
        self,
        feats: List[Tensor],
        data: Dict,
    ) -> Tensor:
        """Forward bevformer viewtransformer."""
        feats = self.get_seq_feats(feats)[0]
        if not self.is_compile:
            img_meta = data["seq_meta"][0]
            im_shape = data["img"].shape[2:]
            bs = feats[0].shape[0]
            device = feats[0].device

            ref2d, ref3d = self.export_reference_points(bs, device=device)
            (
                reference_points_rebatch,
                queries_rebatch_grid,
                restore_bev_grid,
                bev_pillar_counts,
            ) = self.point_sampling(ref3d, self.pc_range, img_meta, im_shape)
            bev_emb = self.get_bev_embed(
                feats=feats,
                ref2d=ref2d,
                reference_points_rebatch=reference_points_rebatch,
                queries_rebatch_grid=queries_rebatch_grid,
                restore_bev_grid=restore_bev_grid,
                bev_pillar_counts=bev_pillar_counts,
            )
        else:
            B = feats[0].shape[0]
            ref2d, _ = self.export_reference_points(B, device=feats[0].device)
            queries_rebatch_grid = data["queries_rebatch_grid"]
            restore_bev_grid = data["restore_bev_grid"]
            reference_points_rebatch = data["reference_points_rebatch"]
            bev_pillar_counts = data["bev_pillar_counts"]
            bev_emb = self.get_bev_embed(
                feats=feats,
                ref2d=ref2d,
                reference_points_rebatch=reference_points_rebatch,
                queries_rebatch_grid=queries_rebatch_grid,
                restore_bev_grid=restore_bev_grid,
                bev_pillar_counts=bev_pillar_counts,
            )
        return bev_emb


# Deprecated alias for backward-compatible configs.
OBJECT_REGISTRY.alias("SingleBevFormerViewTransformer")(
    SingleFrameBevFormerViewTransformer
)
