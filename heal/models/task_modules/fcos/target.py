# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import List, Optional, Sequence, Tuple

import torch
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.apply_func import multi_apply
from torch import Tensor, nn

from heal.core.box_utils import bbox_overlaps

__all__ = [
    "DynamicFcosTarget",
    "get_points",
    "distance2bbox",
]

INF = 1e8


def distance2bbox(
    points: Tensor, distance: Tensor, max_shape: Optional[Tuple] = None
) -> Tensor:
    """Decode ltrb distances to xyxy bboxes.

    Args:
        points: Reference points with shape `(..., 2)` in `(x, y)` format.
        distance: Distances with shape `(..., 4)` in `(l, t, r, b)` format.
        max_shape: Optional image shape `(h, w)` used to clip bbox coordinates.

    Returns:
        Decoded bounding boxes with shape ``(..., 4)`` in
        ``(x1, y1, x2, y2)`` format.
    """
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)


def get_points(
    feat_sizes: List[Tuple],
    strides: List[int],
    dtype: torch.dtype,
    device: torch.device,
) -> List[Tensor]:
    """Generate FCOS center points for each feature level.

    Args:
        feat_sizes: Feature map sizes per level as `(h, w)`.
        strides: Stride per level aligned with `feat_sizes`.
        dtype: Target tensor dtype.
        device: Target tensor device.

    Returns:
        Per-level point tensors, each with shape `(h*w, 2)`.
    """

    def _get_points_single(feat_size, stride):
        h, w = feat_size
        x_range = torch.arange(w, dtype=dtype, device=device)
        y_range = torch.arange(h, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = (
            torch.stack(
                (x.reshape(-1) * stride, y.reshape(-1) * stride), dim=-1
            )
            + stride // 2
        )
        return points

    return [
        _get_points_single(feat_sizes[i], strides[i])
        for i in range(len(feat_sizes))
    ]


@OBJECT_REGISTRY.register
class DynamicFcosTarget(nn.Module):
    """Generate FCOS cls/box targets via simOTA assignment (YOLO-X style)."""

    def __init__(
        self,
        strides: Sequence[int],
        topK: int,
        loss_cls: nn.Module,
        loss_reg: nn.Module,
        cls_out_channels: int,
        background_label: int,
        bbox_relu: bool = False,
    ):
        """Initialize dynamic FCOS target assigner.

        Args:
            strides: Strides of prediction levels.
            topK: Max candidate locations used by dynamic-k matching.
            loss_cls: Classification loss module for matching cost.
            loss_reg: Regression loss module for matching cost.
            cls_out_channels: Number of class channels in prediction.
            background_label: Label id used for background.
            bbox_relu: Whether bbox predictions are ReLUed before assignment.
        """
        super(DynamicFcosTarget, self).__init__()
        self.strides = strides
        self.topK = topK
        self.loss_cls = loss_cls
        self.loss_reg = loss_reg
        self.cls_out_channels = cls_out_channels
        self.background_label = background_label
        self.bbox_relu = bbox_relu

    def _centerness_target(self, pos_bbox_targets: Tensor) -> Tensor:
        """Compute centerness targets from positive ltrb distances.

        Args:
            pos_bbox_targets: Positive bbox targets with shape `(N, 4)`.

        Returns:
            Centerness targets with shape `(N,)`.
        """
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]
        ) * (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    def _get_iou(self, flatten_bbox_preds, bbox_targets, points):
        """Compute aligned IoU between decoded preds and targets.

        Args:
            flatten_bbox_preds: Predicted ltrb distances.
            bbox_targets: Target ltrb distances.
            points: Reference points used for decoding.

        Returns:
            IoU tensor aligned by row/column positions.
        """
        decoded_targets = distance2bbox(points, bbox_targets)
        decoded_preds = distance2bbox(points, flatten_bbox_preds)
        return bbox_overlaps(decoded_preds, decoded_targets, is_aligned=True)

    def _get_cost(
        self, cls_preds, cls_targets, bbox_preds, bbox_targets, points
    ):
        """Build matching cost from classification and regression losses.

        Args:
            cls_preds: Classification predictions.
            cls_targets: Classification targets.
            bbox_preds: Predicted bbox distances.
            bbox_targets: Target bbox distances.
            points: Reference points for bbox decoding.

        Returns:
            Combined matching cost tensor.
        """
        cls_loss = list(self.loss_cls(cls_preds, cls_targets.long()).values())[
            0
        ]
        cls_loss = cls_loss.sum(2)
        bbox_preds_decoded = distance2bbox(points, bbox_preds)
        bbox_targets_decoded = distance2bbox(points, bbox_targets)
        reg_loss = list(
            self.loss_reg(bbox_preds_decoded, bbox_targets_decoded).values()
        )[0]
        return cls_loss + reg_loss

    def _get_target_single(
        self,
        gt_bboxes,
        gt_labels,
        flatten_cls_scores,
        flatten_bbox_preds,
        strides,
        points,
        device,
    ):
        """Assign FCOS targets for one image with dynamic-k matching.

        Args:
            gt_bboxes: Ground-truth boxes of shape `(M, 4)`.
            gt_labels: Ground-truth labels of shape `(M,)`.
            flatten_cls_scores: Flattened class predictions `(N, C)`.
            flatten_bbox_preds: Flattened bbox predictions `(N, 4)`.
            strides: Per-location stride tensor `(N, 4)`.
            points: Flattened location points `(N, 2)`.
            device: Device used for temporary tensors.

        Returns:
            Tuple of ``(cls_targets, bbox_targets, centerness_targets,
            label_weights)``.
        """
        num_gts = gt_bboxes.shape[0]
        if num_gts == 0:
            bbox_targets = torch.zeros_like(flatten_bbox_preds, device=device)
            cls_targets = (
                torch.ones((flatten_cls_scores.shape[0]), device=device)
                * self.background_label
            )
            centerness_targets = torch.zeros(
                (flatten_bbox_preds.shape[0]), device=device
            )
            label_weights = torch.ones_like(
                cls_targets, dtype=torch.float, device=device
            )
            return (
                cls_targets.long(),
                bbox_targets,
                centerness_targets,
                label_weights,
            )

        # Expand GT labels so each location can evaluate all GTs.
        gt_labels = gt_labels.view(1, -1).repeat(
            flatten_cls_scores.shape[0], 1
        )
        bbox_targets = torch.zeros(
            (flatten_bbox_preds.shape[0], num_gts, 4), device=device
        )
        # Build ltrb distances from each location point to every GT box.
        bbox_targets[..., 0:2] = points[:, None, :] - gt_bboxes[None, :, 0:2]
        bbox_targets[..., 2:4] = gt_bboxes[None, :, 2:4] - points[:, None, :]

        inside_gt_bbox_mask = (bbox_targets.min(-1)[0] > 0).int()

        ignore_labels = gt_labels < 0
        ignore_labels[inside_gt_bbox_mask != 1] = 0
        inside_gt_bbox_mask[ignore_labels] = -1
        bbox_targets /= strides[:, None, :]
        points_norm = points / strides[..., 0:2]
        points_norm = points_norm.view(-1, 1, 2).repeat(1, num_gts, 1)
        flatten_bbox_preds_exp = flatten_bbox_preds.view(-1, 1, 4).repeat(
            1, num_gts, 1
        )
        flatten_cls_scores_exp = flatten_cls_scores.view(
            -1, 1, self.cls_out_channels
        ).repeat(1, num_gts, 1)

        # Candidate quality for each (location, gt) pair.
        ious = self._get_iou(flatten_bbox_preds_exp, bbox_targets, points_norm)
        ious[inside_gt_bbox_mask != 1] = 0.0
        cost = self._get_cost(
            flatten_cls_scores_exp,
            gt_labels,
            flatten_bbox_preds_exp,
            bbox_targets,
            points_norm,
        )
        cost[inside_gt_bbox_mask != 1] = INF
        cost = cost.permute(1, 0).contiguous()

        # Dynamic-k matching:
        # 1) Use top-k IoUs to estimate how many positives each GT should keep.
        # 2) Select lowest-cost locations for each GT.
        # 3) Resolve conflicts so one location matches at most one GT.
        n_candidate_k = min(self.topK, ious.size(0))
        topk_ious, _ = torch.topk(ious, n_candidate_k, dim=0)
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)
        matching_matrix = torch.zeros_like(
            cost, device=flatten_bbox_preds.device
        )
        for gt_idx in range(num_gts):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1.0
        matching_gt = matching_matrix.sum(0)
        if (matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, matching_gt > 1], dim=0)
            matching_matrix[:, matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, matching_gt > 1] = 1.0
        matching_matrix = matching_matrix.permute(1, 0).contiguous()
        matching_matrix[inside_gt_bbox_mask != 1] = 0.0
        gt_labels[matching_matrix == 0] = self.background_label
        argmin = gt_labels.argmin(1).view(-1)

        # Convert multi-GT tensors back to one target per location.
        cls_targets = gt_labels[torch.arange(gt_labels.shape[0]), argmin]
        bbox_targets = bbox_targets[
            torch.arange(bbox_targets.shape[0]), argmin
        ]
        centerness_targets = ious[torch.arange(bbox_targets.shape[0]), argmin]
        label_weights = torch.ones_like(gt_labels, dtype=torch.float)
        label_weights[inside_gt_bbox_mask == -1] = 0
        label_weights, _ = label_weights.min(-1)
        return cls_targets, bbox_targets, centerness_targets, label_weights

    def forward(self, label, pred, *args):
        """Generate training targets for FCOS losses.

        Args:
            label: Label dict containing `gt_bboxes` and `gt_classes`.
            pred: Prediction tuple `(cls_scores, bbox_preds, centernesses)`.
            *args: Unused extra arguments for interface compatibility.

        Returns:
            Three dicts for cls, giou and centerness loss inputs.
        """
        cls_scores, bbox_preds, centernesses = pred
        if self.bbox_relu:
            bbox_preds = [nn.functional.relu(i) for i in bbox_preds]

        gt_bboxes_list = label["gt_bboxes"]
        gt_labels_list = label["gt_classes"]
        num_imgs = len(gt_bboxes_list)
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(
                num_imgs, -1, self.cls_out_channels
            )
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for centerness in centernesses
        ]
        device = bbox_preds[0].device
        dtype = bbox_preds[0].dtype
        flatten_strides = [
            torch.tensor([s, s, s, s], device=device)
            .view(-1, 4)
            .repeat(flatten_bbox_pred.shape[1], 1)
            for flatten_bbox_pred, s in zip(flatten_bbox_preds, self.strides)
        ]
        flatten_strides = torch.cat(flatten_strides)
        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_centerness = torch.cat(flatten_centerness, dim=1)
        feat_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = get_points(feat_sizes, self.strides, dtype, device)
        concat_points = torch.cat(all_level_points, dim=0)

        with torch.no_grad():
            (
                cls_targets_list,
                bbox_targets_list,
                centerness_targets_list,
                label_weights_list,
            ) = multi_apply(
                self._get_target_single,
                gt_bboxes_list,
                gt_labels_list,
                flatten_cls_scores,
                flatten_bbox_preds,
                strides=flatten_strides,
                points=concat_points,
                device=device,
            )
        cls_targets = torch.cat(cls_targets_list)
        bbox_targets = torch.cat(bbox_targets_list)
        centerness_targets = torch.cat(centerness_targets_list)
        label_weights = torch.cat(label_weights_list)
        pos_inds = (
            (
                (cls_targets >= 0)
                & (cls_targets != self.background_label)
                & (label_weights > 0)
            )
            .nonzero()
            .reshape(-1)
        )
        num_pos = len(pos_inds)
        cls_avg_factor = max(num_pos, 1.0)
        if num_pos == 0:
            pos_inds = flatten_cls_scores.new_zeros((1,)).long()
        pos_bbox_targets = bbox_targets[pos_inds]
        if num_pos == 0:
            centerness_weight = torch.tensor([0.0], device=device)
            bbox_weight = torch.tensor([0.0], device=device)
            bbox_avg_factor = None
        else:
            centerness_weight = None
            bbox_weight = self._centerness_target(pos_bbox_targets)
            bbox_avg_factor = bbox_weight.sum()

        flatten_cls_scores = flatten_cls_scores.view(-1, self.cls_out_channels)
        flatten_bbox_preds = flatten_bbox_preds.view(-1, 4)
        flatten_centerness = flatten_centerness.view(-1)
        cls_target = {
            "pred": flatten_cls_scores,
            "target": cls_targets.long(),
            "weight": label_weights,
            "avg_factor": cls_avg_factor,
        }
        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_points = torch.zeros((2), device=device)
        pos_decoded_targets = distance2bbox(pos_points, pos_bbox_targets)
        pos_decoded_preds = distance2bbox(pos_points, pos_bbox_preds)
        giou_target = {
            "pred": pos_decoded_preds,
            "target": pos_decoded_targets,
            "weight": bbox_weight,
            "avg_factor": bbox_avg_factor,
        }
        pos_centerness = flatten_centerness[pos_inds]
        pos_centerness_targets = centerness_targets[pos_inds]
        centerness_target = {
            "pred": pos_centerness,
            "target": pos_centerness_targets,
            "weight": centerness_weight,
        }
        return cls_target, giou_target, centerness_target
