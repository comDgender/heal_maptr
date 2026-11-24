# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
# Source code reference to OpenMMLab.
from typing import List, Optional

import torch
from hat_engine.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class CenterPointBBoxCoder:
    """Bbox coder for CenterPoint.

    Args:
        pc_range: Range of point cloud.
        out_size_factor: Downsample factor of the model.
        voxel_size: Size of voxel.
        post_center_range: Limit of the center.
            Default: None.
        max_num: Max number to be kept. Default: 100.
        score_threshold: Threshold to filter boxes
            based on score. Default: None.
    """

    def __init__(
        self,
        pc_range: List[float],
        out_size_factor: int,
        voxel_size: List[float],
        post_center_range: Optional[List[float]] = None,
        max_num: Optional[int] = 100,
        score_threshold: Optional[float] = None,
    ):

        self.pc_range = pc_range
        self.out_size_factor = out_size_factor
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold

    def _gather_feat(
        self,
        feats: torch.Tensor,
        inds: torch.Tensor,
        feat_masks: torch.Tensor = None,
    ):
        """Given feats and indexes, returns the gathered feats.

        Args:
            feats: Features to be transposed and gathered
                with the shape of [B, W*H, C].
            inds: Indexes with the shape of [B, N].
            feat_masks: Mask of the feats.
                Default: None.

        Returns:
            feats: Gathered feats.
        """
        dim = feats.size(2)
        inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), dim)
        feats = feats.gather(1, inds)
        if feat_masks is not None:
            feat_masks = feat_masks.unsqueeze(2).expand_as(feats)
            feats = feats[feat_masks]
            feats = feats.view(-1, dim)
        return feats

    def _gather_decode_feat(
        self, feat: torch.Tensor, inds: torch.Tensor, batch: int, last_dim: int
    ) -> torch.Tensor:
        return self._transpose_and_gather_feat(feat, inds).view(
            batch, self.max_num, last_dim
        )

    def _decode_bev_coord(
        self, coord: torch.Tensor, batch: int, axis: int
    ) -> torch.Tensor:
        return (
            coord.view(batch, self.max_num, 1)
            * self.out_size_factor
            * self.voxel_size[axis]
            + self.pc_range[axis]
        )

    def _spatial_coords_from_indices(
        self, cell_ids: torch.Tensor, width: int
    ) -> tuple:
        ys = (cell_ids // width).float()
        xs = (cell_ids % width).float()
        return ys, xs

    def _topk(self, scores: torch.Tensor, K: int = 80):
        """Get indexes based on scores.

        Args:
            scores: scores with the shape of [B, N, W, H].
            K: Number to be kept. Defaults to 80.

        Returns:
            tuple[torch.Tensor]:
                - topk_score: Selected scores with the shape of [B, K].
                - topk_inds: Selected indexes with the shape of [B, K].
                - topk_clses: Selected classes with the shape of [B, K].
                - topk_ys: Selected y coord with the shape of [B, K].
                - topk_xs: Selected x coord with the shape of [B, K].
        """
        batch, num_cls, height, width = scores.size()
        grid_size = height * width

        cls_scores, cls_cell_ids = torch.topk(
            scores.reshape(batch, num_cls, -1), K
        )
        cell_ids = cls_cell_ids % grid_size
        cls_ys, cls_xs = self._spatial_coords_from_indices(cell_ids, width)

        flat_scores = cls_scores.reshape(batch, -1)
        best_scores, rank = torch.topk(flat_scores, K)
        cls_labels = (rank // K).int()

        def _pick_candidates(tensor: torch.Tensor) -> torch.Tensor:
            return self._gather_feat(
                tensor.reshape(batch, -1, 1), rank
            ).reshape(batch, K)

        return (
            best_scores,
            _pick_candidates(cell_ids),
            cls_labels,
            _pick_candidates(cls_ys),
            _pick_candidates(cls_xs),
        )

    def _transpose_and_gather_feat(
        self, feat: torch.Tensor, ind: torch.Tensor
    ):
        """Given feats and indexes, returns the transposed and gathered feats.

        Args:
            feat: Features to be transposed and gathered
                with the shape of [B, C, W, H].
            ind: Indexes with the shape of [B, N].

        Returns:
            torch.Tensor: Transposed and gathered feats.
        """
        n_batch, channels, feat_h, feat_w = feat.shape
        flattened = feat.permute(0, 2, 3, 1).reshape(
            n_batch, feat_h * feat_w, channels
        )
        return self._gather_feat(flattened, ind)

    def encode(self):
        pass

    def decode(
        self,
        heat: torch.Tensor,
        rot_sine: torch.Tensor,
        rot_cosine: torch.Tensor,
        hei: torch.Tensor,
        dim: torch.Tensor,
        vel: torch.Tensor,
        reg: torch.Tensor = None,
        task_id: int = -1,
    ):
        """Decode bboxes.

        Args:
            heat: Heatmap with the shape of [B, N, W, H].
            rot_sine: Sine of rotation with the shape of
                [B, 1, W, H].
            rot_cosine: Cosine of rotation with the shape of
                [B, 1, W, H].
            hei: Height of the boxes with the shape
                of [B, 1, W, H].
            dim: Dim of the boxes with the shape of
                [B, 3, W, H].
            vel: Velocity with the shape of [B, 2, W, H].
            reg: Regression value of the boxes in
                2D with the shape of [B, 2, W, H]. Default: None.
            task_id: Index of task. Default: -1.

        Returns:
            list[dict]: Decoded boxes.
        """
        batch, cat, _, _ = heat.size()

        scores, inds, clses, ys, xs = self._topk(heat, K=self.max_num)

        if reg is not None:
            reg = self._transpose_and_gather_feat(reg, inds)
            reg = reg.view(batch, self.max_num, 2)
            xs = xs.view(batch, self.max_num, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, self.max_num, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, self.max_num, 1) + 0.5
            ys = ys.view(batch, self.max_num, 1) + 0.5

        rot_sine = self._gather_decode_feat(rot_sine, inds, batch, 1)
        rot_cosine = self._gather_decode_feat(rot_cosine, inds, batch, 1)
        rot = torch.atan2(rot_sine, rot_cosine)

        hei = self._gather_decode_feat(hei, inds, batch, 1)
        dim = self._gather_decode_feat(dim, inds, batch, 3)

        # class label
        clses = clses.view(batch, self.max_num).float()
        scores = scores.view(batch, self.max_num)

        xs = self._decode_bev_coord(xs, batch, 0)
        ys = self._decode_bev_coord(ys, batch, 1)

        box_parts = [xs, ys, hei, dim, rot]
        if vel is not None:
            box_parts.append(self._gather_decode_feat(vel, inds, batch, 2))
        final_box_preds = torch.cat(box_parts, dim=2)

        score_mask = None
        if self.score_threshold is not None:
            score_mask = scores > self.score_threshold

        if self.post_center_range is None:
            raise NotImplementedError(
                "Need to reorganize output as a batch, only "
                "support post_center_range is not None for now!"
            )

        center_bounds = torch.tensor(
            self.post_center_range,
            device=heat.device,
            dtype=final_box_preds.dtype,
        )
        xyz = final_box_preds[..., :3]
        in_range_mask = (xyz >= center_bounds[:3]).all(dim=2)
        in_range_mask &= (xyz <= center_bounds[3:]).all(dim=2)

        predictions_dicts = []
        for batch_idx in range(batch):
            keep = in_range_mask[batch_idx]
            if score_mask is not None:
                keep = keep & score_mask[batch_idx]
            predictions_dicts.append(
                {
                    "bboxes": final_box_preds[batch_idx, keep],
                    "scores": scores[batch_idx, keep],
                    "labels": clses[batch_idx, keep],
                }
            )

        return predictions_dicts
