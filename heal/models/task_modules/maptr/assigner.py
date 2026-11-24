# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import logging
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from hat_engine.registry import OBJECT_REGISTRY
from torch import Tensor, nn

from heal.models.task_modules.maptr.criterion import (
    bbox_cxcywh_to_xyxy,
    bbox_xyxy_to_cxcywh,
)

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

logger = logging.getLogger(__name__)

__all__ = [
    "MapTRAssigner",
]


def normalize_2d_bbox(bboxes, pc_range):

    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    cxcywh_bboxes = bbox_xyxy_to_cxcywh(bboxes)
    cxcywh_bboxes[..., 0:1] = cxcywh_bboxes[..., 0:1] - pc_range[0]
    cxcywh_bboxes[..., 1:2] = cxcywh_bboxes[..., 1:2] - pc_range[1]
    factor = bboxes.new_tensor([patch_w, patch_h, patch_w, patch_h])

    normalized_bboxes = cxcywh_bboxes / factor
    return normalized_bboxes


def normalize_2d_pts(pts, pc_range):
    patch_h = pc_range[4] - pc_range[1]
    patch_w = pc_range[3] - pc_range[0]
    new_pts = pts.clone()
    new_pts[..., 0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[..., 1:2] = pts[..., 1:2] - pc_range[1]
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts


def denormalize_2d_bbox(bboxes, pc_range):

    bboxes = bbox_cxcywh_to_xyxy(bboxes)
    bboxes[..., 0::2] = (
        bboxes[..., 0::2] * (pc_range[3] - pc_range[0]) + pc_range[0]
    )
    bboxes[..., 1::2] = (
        bboxes[..., 1::2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    )

    return bboxes


def denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[..., 0:1] = (
        pts[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    )
    new_pts[..., 1:2] = (
        pts[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    )
    return new_pts


@OBJECT_REGISTRY.register
class MapTRAssigner(nn.Module):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        cls_cost: The classification cost module. Default: None.
        pts_cost: The points cost module. Default: None.
        pc_range: The point cloud range. Default: None.
        pred_absolute_points: Whether to predict absolute points.
            Default: False.
    """

    def __init__(
        self,
        cls_cost: nn.Module = None,
        pts_cost: nn.Module = None,
        pc_range: List[float] = None,
        pred_absolute_points: bool = False,
    ):
        super(MapTRAssigner, self).__init__()
        self.cls_cost = cls_cost
        self.pts_cost = pts_cost
        self.pc_range = pc_range
        self.pred_absolute_points = pred_absolute_points

    def assign(
        self,
        bbox_pred: Tensor,
        cls_pred: Tensor,
        pts_pred: Tensor,
        gt_bboxes: Tensor,
        gt_labels: Tensor,
        gt_pts: Tensor,
        gt_bboxes_ignore: Tensor = None,
        eps: Union[int, float] = 1e-7,
    ) -> Tuple[int, Tensor, None, Tensor, Tensor]:
        """Match map queries to GT polylines via cls + point costs.

        Pipeline: initialize slots as ignore (-1) -> build pairwise cost from
        classification and ordered point distances -> run Hungarian matching
        on CPU -> fill background (0), then write 1-based GT indices for
        matched queries. ``order_index`` records which GT point-order won
        the min over ``num_orders``.

        Args:
            bbox_pred: Normalized query boxes ``(cx, cy, w, h)`` in ``[0, 1]``,
                shape ``[num_query, 4]`` (shape-checked only; unused in cost).
            cls_pred: Classification logits, ``[num_query, num_class]``.
            pts_pred: Predicted polyline points, ``[num_query, num_pts, 2]``.
            gt_bboxes: Absolute GT boxes ``(x1, y1, x2, y2)``, ``[num_gt, 4]``.
            gt_labels: GT class ids, shape ``(num_gt,)``.
            gt_pts: GT polylines with order permutations,
                ``[num_gt, num_orders, num_pts_per_gtline, 2]``.
            gt_bboxes_ignore: Must be ``None`` (ignored boxes unsupported).
            eps: Reserved numeric epsilon (unused in the current path).

        Returns:
            num_gts: GT count for this sample.
            assigned_gt_inds: Per-query GT index (0 = bg, >0 = 1-based GT).
            None: Placeholder for API compatibility.
            assigned_labels: Per-query GT label (-1 if unmatched).
            order_index: Best GT point-order index per (query, gt) pair.
        """
        assert (
            gt_bboxes_ignore is None
        ), "Only case when gt_bboxes_ignore is None is supported."
        assert (
            bbox_pred.shape[-1] == 4
        ), "Only support bbox pred shape is 4 dims"
        num_gts = gt_bboxes.shape[0]
        num_queries = bbox_pred.shape[0]

        # Default: unmatched / ignore sentinel (-1).
        assigned_gt_inds = bbox_pred.new_full(
            (num_queries,), -1, dtype=torch.long
        )
        assigned_labels = bbox_pred.new_full(
            (num_queries,), -1, dtype=torch.long
        )
        if num_gts == 0 or num_queries == 0:
            if num_gts == 0:
                assigned_gt_inds.fill_(0)
            return num_gts, assigned_gt_inds, None, assigned_labels, None

        # Pairwise costs: classification + ordered point distances.
        cls_cost = self.cls_cost(cls_pred, gt_labels)

        _, num_orders, num_pts_per_gtline, num_coords = gt_pts.shape
        normalized_gt_pts = normalize_2d_pts(gt_pts, self.pc_range)
        num_pts_per_predline = pts_pred.size(1)
        if num_pts_per_predline != num_pts_per_gtline:
            pts_pred_interpolated = F.interpolate(
                pts_pred.permute(0, 2, 1),
                size=(num_pts_per_gtline),
                mode="linear",
                align_corners=True,
            )
            pts_pred_interpolated = pts_pred_interpolated.permute(
                0, 2, 1
            ).contiguous()
        else:
            pts_pred_interpolated = pts_pred
        if self.pred_absolute_points:
            pts_pred_interpolated = normalize_2d_pts(
                pts_pred_interpolated, self.pc_range
            )
        # num_q, num_pts, 2 <-> num_gt, num_pts, 2
        pts_cost_ordered = self.pts_cost(
            pts_pred_interpolated, normalized_gt_pts
        )
        pts_cost_ordered = pts_cost_ordered.view(
            num_queries, num_gts, num_orders
        )
        pts_cost, order_index = torch.min(pts_cost_ordered, 2)

        # weighted sum of above costs
        cost = cls_cost + pts_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()

        # Guard against NaN/Inf in the cost matrix: a single bad batch (e.g.
        # an out-of-range pts prediction with pred_absolute_points=True) can
        # feed NaN/Inf through cls_cost/pts_cost, which scipy rejects with
        # "matrix contains invalid numeric entries". Mirror the guard already
        # used by BevFormerHungarianAssigner3D. NaN -> 0 (neutral cost),
        # +/-Inf -> clipped to a large finite value so the matching still
        # proceeds instead of crashing the whole run.
        cost = np.where(np.isneginf(cost), -1e8, cost)
        cost = np.where(np.isinf(cost), 1e8, cost)
        cost = np.where(np.isnan(cost), 0.0, cost)

        if linear_sum_assignment is None:
            raise ImportError(
                'Please run "pip install scipy" ' "to install scipy first."
            )
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            bbox_pred.device
        )
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            bbox_pred.device
        )

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return num_gts, assigned_gt_inds, None, assigned_labels, order_index
