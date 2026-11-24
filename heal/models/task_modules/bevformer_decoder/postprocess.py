# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

"""BEVFormer inference post-processing (top-k decode + range filter).

Takes last decoder layer cls/bbox outputs, selects top-``max_num`` scores
across all classes, denormalizes 3D boxes, and filters by score threshold
and ``post_center_range``.
"""

import logging
from typing import Dict, List

import torch
from hat_engine.registry import OBJECT_REGISTRY
from torch import Tensor, nn
from torch.amp import autocast

from .utils import denormalize_bbox

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class BevFormerProcess(nn.Module):
    """Decode and filter BEVFormer 3D detections for eval / visualization.

    Output per sample: ``(N, 11)`` tensor
    ``[cx, cy, cz, w, l, h, yaw, vx, vy, score, label]``.

    Args:
        pc_range: VCS bounds (passed through for config consistency).
        post_center_range: ``[x_min, y_min, z_min, x_max, y_max, z_max]`` filter.  # noqa: E501
        max_num: Top-k detections kept before filtering.
        score_threshold: Min score; auto-relaxed if no box passes.
        num_classes: Number of detection classes.
    """

    def __init__(
        self,
        pc_range: List[float],
        post_center_range: List[float] = None,
        max_num: int = 100,
        score_threshold: float = None,
        num_classes: int = 10,
    ):
        super(BevFormerProcess, self).__init__()
        self.pc_range = pc_range
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def decode_single(self, cls_scores: Tensor, bbox_preds: Tensor) -> Tensor:
        """Top-k multi-class NMS-free decode + center-range filter for one sample."""  # noqa: E501
        max_num = self.max_num
        cls_scores = cls_scores.sigmoid()
        # Flatten (num_query, num_classes) and take global top-k scores.
        scores, flat_idx = cls_scores.reshape(-1).topk(max_num)
        class_ids = flat_idx % self.num_classes
        query_ids = flat_idx // self.num_classes
        boxes = denormalize_bbox(bbox_preds[query_ids])

        # Optional score gate; relax threshold if the mask is empty.
        score_mask = None
        if self.score_threshold is not None:
            thr = self.score_threshold
            score_mask = scores > thr
            while score_mask.sum() == 0:
                thr *= 0.9
                if thr < 0.01:
                    score_mask = scores > -1
                    break
                score_mask = scores >= thr

        if self.post_center_range is None:
            raise NotImplementedError(
                "Need to reorganize output as a batch, only "
                "support post_center_range is not None for now!"
            )

        center_lim = torch.as_tensor(
            self.post_center_range, device=scores.device
        )
        self.post_center_range = center_lim
        keep = (boxes[..., :3] >= center_lim[:3]).all(1) & (
            boxes[..., :3] <= center_lim[3:]
        ).all(1)
        if self.score_threshold:
            keep = keep & score_mask

        predictions = torch.cat(
            [
                boxes[keep],
                scores[keep].unsqueeze(-1),
                class_ids[keep].unsqueeze(-1),
            ],
            dim=-1,
        ).detach()
        return predictions

    @autocast("cuda", enabled=False)
    def forward(self, preds_dicts: Dict) -> List:
        """Use last decoder layer only; return list of per-sample prediction tensors."""  # noqa: E501
        all_cls_scores = preds_dicts["all_cls_scores"][-1]
        all_bbox_preds = preds_dicts["all_bbox_preds"][-1]
        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(
                self.decode_single(all_cls_scores[i], all_bbox_preds[i])
            )
        return predictions_list


@OBJECT_REGISTRY.register
class BevFormer2DProcess(BevFormerProcess):
    def denormalize_bbox_2d(
        self, normalized_bboxes: torch.Tensor
    ) -> torch.Tensor:
        """Denormalize bbox."""

        # center in the bev
        cx = normalized_bboxes[..., 0:1]
        cy = normalized_bboxes[..., 1:2]

        # size
        w = normalized_bboxes[..., 2:3]
        bl = normalized_bboxes[..., 3:4]

        w = w.exp()
        bl = bl.exp()
        if normalized_bboxes.size(-1) == 6:
            # rotation
            rot_sine = normalized_bboxes[..., 4:5]

            rot_cosine = normalized_bboxes[..., 5:6]
            rot = torch.atan2(rot_sine, rot_cosine)
            denormalized_bboxes = torch.cat([cx, cy, w, bl, rot], dim=-1)
        elif normalized_bboxes.size(-1) == 4:
            denormalized_bboxes = torch.cat([cx, cy, w, bl], dim=-1)
        else:
            # rotation
            rot_sine = normalized_bboxes[..., 4:5]

            rot_cosine = normalized_bboxes[..., 5:6]
            rot = torch.atan2(rot_sine, rot_cosine)
            vx = normalized_bboxes[:, 6:7]
            vy = normalized_bboxes[:, 7:8]
            denormalized_bboxes = torch.cat(
                [cx, cy, w, bl, rot, vx, vy], dim=-1
            )
        return denormalized_bboxes

    def decode_single(self, cls_scores: Tensor, bbox_preds: Tensor) -> Tensor:
        """Decode bboxes."""

        max_num = self.max_num
        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.reshape(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_preds = bbox_preds[bbox_index]

        final_box_preds = self.denormalize_bbox_2d(bbox_preds)
        final_scores = scores
        final_preds = labels

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
            tmp_score = self.score_threshold
            while thresh_mask.sum() == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = final_scores > -1
                    break
                thresh_mask = final_scores >= tmp_score

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=scores.device
            )
            mask = (
                final_box_preds[..., :2] >= self.post_center_range[:2]
            ).all(1)
            mask &= (
                final_box_preds[..., :2] <= self.post_center_range[3:5]
            ).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes2d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            predictions = torch.cat(
                [boxes2d, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1
            ).detach()

        else:
            raise NotImplementedError(
                "Need to reorganize output as a batch, only "
                "support post_center_range is not None for now!"
            )
        return predictions
