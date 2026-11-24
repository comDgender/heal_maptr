# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

"""BEVFormer 3D detection Hungarian matching (query-to-GT assignment).

Used by ``BevFormerCriterion`` to assign each object query to a GT box or
background before computing cls/reg losses. Cost = cls_cost + reg_cost;
``linear_sum_assignment`` finds the optimal one-to-one matching.
"""

import logging
from typing import Tuple

import numpy as np
import torch
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.package_helper import require_packages
from torch import Tensor, nn

from .utils import normalize_bbox, normalize_bbox_2d

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class BevFormerHungarianAssigner3D(nn.Module):
    """Hungarian assigner for 3D DETR-style detection on BEV.

    Matches ``num_query`` predictions to ``num_gt`` boxes by minimizing
    ``cls_cost + reg_cost``. Returns ``assigned_gt_inds`` where 0 = background,
    k > 0 = matched to GT index (k-1).

    Args:
        cls_cost: Classification cost module (e.g. FocalLossCost).
        reg_cost: Regression cost module (e.g. BBox3DL1Cost on first 8 dims).
    """

    @require_packages("scipy", raise_msg="Please `pip3 install scipy`")
    def __init__(
        self,
        cls_cost: nn.Module,
        reg_cost: nn.Module,
    ):
        super(BevFormerHungarianAssigner3D, self).__init__()
        self.cls_cost = cls_cost
        self.reg_cost = reg_cost

    def assign(
        self,
        bbox_pred: Tensor,
        cls_pred: Tensor,
        gt_bboxes: Tensor,
        gt_labels: Tensor,
    ) -> Tuple[Tensor]:
        """Assign each query to a GT index (1-based) or background (0)."""
        n_gt = gt_bboxes.shape[0]
        n_query = bbox_pred.shape[0]

        # Default: unmatched / ignore sentinel.
        match_ids = bbox_pred.new_full((n_query,), -1, dtype=torch.long)
        if n_gt == 0 or n_query == 0:
            if n_gt == 0:
                match_ids.fill_(0)
            return match_ids

        # Pairwise costs: classification + L1 on normalized 8-dim box codes.
        cost_cls = self.cls_cost(cls_pred, gt_labels)
        gt_norm = normalize_bbox(gt_bboxes)
        cost_reg = self.reg_cost(bbox_pred[:, :8], gt_norm[:, :8])
        cost_mat = (cost_cls + cost_reg).detach().cpu().numpy()

        # Sanitize non-finite entries before the Hungarian solver.
        cost_mat = np.nan_to_num(cost_mat, nan=0.0, posinf=1e8, neginf=-1e8)
        query_idx, gt_idx = linear_sum_assignment(cost_mat)
        query_idx = torch.as_tensor(
            query_idx, dtype=torch.long, device=bbox_pred.device
        )
        gt_idx = torch.as_tensor(
            gt_idx, dtype=torch.long, device=bbox_pred.device
        )

        # Background everywhere, then overwrite matched queries (1-based GT).
        match_ids.fill_(0)
        match_ids[query_idx] = gt_idx + 1
        return match_ids


@OBJECT_REGISTRY.register
class BevFormerHungarianAssigner2D(nn.Module):
    """The basic structure of BevFormerHungarianAssigner2D.

    Args:
        cls_cost: classification cost module.
        reg_cost: regression cost module.
        iou_cost: iou cost module.
    """

    @require_packages("scipy", raise_msg="Please `pip3 install scipy`")
    def __init__(
        self,
        cls_cost: nn.Module,
        reg_cost: nn.Module,
    ):
        super(BevFormerHungarianAssigner2D, self).__init__()
        self.cls_cost = cls_cost
        self.reg_cost = reg_cost

    def assign(
        self,
        bbox_pred: Tensor,
        cls_pred: Tensor,
        gt_bboxes: Tensor,
        gt_labels: Tensor,
    ) -> Tuple[Tensor]:
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full(
            (num_bboxes,), -1, dtype=torch.long
        )
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return assigned_gt_inds

        # 2. compute the weighted costs
        # classification and bboxcost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        # regression L1 cost
        normalized_gt_bboxes = normalize_bbox_2d(gt_bboxes)

        reg_cost = self.reg_cost(bbox_pred, normalized_gt_bboxes)

        # weighted sum of above two costs
        cost = cls_cost + reg_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()

        cost = np.where(np.isneginf(cost), -1e8, cost)
        cost = np.where(np.isinf(cost), 1e8, cost)
        cost = np.where(np.isnan(cost), 0.0, cost)

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

        return assigned_gt_inds
