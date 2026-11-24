# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import Tuple

import torch
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.apply_func import multi_apply
from torch import nn

from heal.models.task_modules.fcos.target import get_points

INF = 1e8


@OBJECT_REGISTRY.register
class FCOS3DTarget(nn.Module):
    """Generate cls/reg targets for FCOS3D in training stage.

    Args:
        num_classes: Number of categories excluding the background category.
        background_label: Label ID of background.
        bbox_code_size: Dimensions of predicted bounding boxes.
        regress_ranges: Regress range of multiple level points.
        strides: Downsample factor of each feature map.
        pred_attrs: Whether to predict attributes.
        num_attrs: The number of attributes to be predicted.
        center_sampling: If true, use center sampling.
        center_sample_radius: Radius of center sampling.
        centerness_alpha: Parameter used to adjust the intensity attenuation.
        norm_on_bbox: If true, normalize the regression targets with strides.
    """

    def __init__(
        self,
        num_classes: int,
        background_label: int,
        bbox_code_size: int,
        regress_ranges: Tuple[Tuple[int, int]],
        strides: Tuple[int],
        pred_attrs: bool,
        num_attrs: int,
        center_sampling: bool,
        center_sample_radius: float = 1.5,
        centerness_alpha: float = 2.5,
        norm_on_bbox: bool = True,
    ):
        super(FCOS3DTarget, self).__init__()
        self.background_label = (
            num_classes if background_label is None else background_label
        )
        self.bbox_code_size = bbox_code_size
        self.regress_ranges = regress_ranges
        self.strides = strides
        self.attr_background_label = -1
        if pred_attrs:
            self.attr_background_label = num_attrs
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.centerness_alpha = centerness_alpha
        self.norm_on_bbox = norm_on_bbox

    def _get_target_single(
        self,
        gt_bboxes,
        gt_labels,
        gt_bboxes_3d,
        gt_labels_3d,
        centers2d,
        depths,
        attr_labels,
        points,
        regress_ranges,
        num_points_per_lvl,
    ):
        n_pts = points.size(0)
        n_gts = gt_labels.size(0)
        if not isinstance(gt_bboxes_3d, torch.Tensor):
            gt_bboxes_3d = gt_bboxes_3d.tensor.to(gt_bboxes.device)

        if n_gts == 0:
            bg2d = gt_labels.new_full((n_pts,), self.background_label)
            bg3d = gt_labels_3d.new_full((n_pts,), self.background_label)
            bg_attr = attr_labels.new_full(
                (n_pts,), self.attr_background_label
            )
            return (
                bg2d,
                gt_bboxes.new_zeros((n_pts, 4)),
                bg3d,
                gt_bboxes_3d.new_zeros((n_pts, self.bbox_code_size)),
                gt_bboxes_3d.new_zeros((n_pts,)),
                bg_attr,
            )

        # Local yaw relative to the camera ray through the 3D box center.
        gt_bboxes_3d[..., 6] = gt_bboxes_3d[..., 6] - torch.atan2(
            gt_bboxes_3d[..., 0], gt_bboxes_3d[..., 2]
        )

        # Broadcast GT fields over all points: (n_pts, n_gts, ...).
        reg_rng = regress_ranges.unsqueeze(1).expand(n_pts, n_gts, 2)
        boxes2d = gt_bboxes.unsqueeze(0).expand(n_pts, n_gts, 4)
        ctr2d = centers2d.unsqueeze(0).expand(n_pts, n_gts, 2)
        boxes3d = gt_bboxes_3d.unsqueeze(0).expand(
            n_pts, n_gts, self.bbox_code_size
        )
        depth_b = depths.view(1, n_gts, 1).expand(n_pts, n_gts, 1)

        px = points[:, 0].unsqueeze(1).expand(n_pts, n_gts)
        py = points[:, 1].unsqueeze(1).expand(n_pts, n_gts)

        dx = (px - ctr2d[..., 0]).unsqueeze(-1)
        dy = (py - ctr2d[..., 1]).unsqueeze(-1)
        targets_3d = torch.cat((dx, dy, depth_b, boxes3d[..., 3:]), dim=-1)

        # 2D ltrb offsets to the GT box edges.
        targets_2d = torch.stack(
            (
                px - boxes2d[..., 0],
                py - boxes2d[..., 1],
                boxes2d[..., 2] - px,
                boxes2d[..., 3] - py,
            ),
            dim=-1,
        )

        if not self.center_sampling:
            raise AssertionError(
                "Setting center_sampling to False "
                "has not been implemented for FCOS3D."
            )

        # Per-level sampling radius in image pixels: (n_pts, n_gts).
        sample_stride = ctr2d.new_zeros(n_pts, n_gts)
        cursor = 0
        for lvl, n_lvl in enumerate(num_points_per_lvl):
            nxt = cursor + n_lvl
            sample_stride[cursor:nxt] = (
                self.strides[lvl] * self.center_sample_radius
            )
            cursor = nxt

        cx, cy = ctr2d[..., 0], ctr2d[..., 1]
        center_box = torch.stack(
            (
                cx - sample_stride,
                cy - sample_stride,
                cx + sample_stride,
                cy + sample_stride,
            ),
            dim=-1,
        )
        # Positive if the point lies inside the expanded center box.
        inside_center = (
            torch.stack(
                (
                    px - center_box[..., 0],
                    py - center_box[..., 1],
                    center_box[..., 2] - px,
                    center_box[..., 3] - py,
                ),
                dim=-1,
            ).amin(dim=-1)
            > 0
        )

        max_ltrb = targets_2d.amax(dim=-1)
        inside_range = (max_ltrb >= reg_rng[..., 0]) & (
            max_ltrb <= reg_rng[..., 1]
        )

        # Assign each point to the nearest valid GT center (INF = invalid).
        dist = torch.linalg.vector_norm(targets_3d[..., :2], dim=-1)
        dist = dist.masked_fill(~inside_center, INF)
        dist = dist.masked_fill(~inside_range, INF)
        min_dist, gt_inds = dist.min(dim=1)

        labels = gt_labels[gt_inds]
        labels_3d = gt_labels_3d[gt_inds]
        attrs = attr_labels[gt_inds]
        bg_mask = min_dist == INF
        labels = labels.masked_fill(bg_mask, self.background_label)
        labels_3d = labels_3d.masked_fill(bg_mask, self.background_label)
        attrs = attrs.masked_fill(bg_mask, self.attr_background_label)

        pt_idx = torch.arange(n_pts, device=points.device)
        targets_2d = targets_2d[pt_idx, gt_inds]
        targets_3d = targets_3d[pt_idx, gt_inds]

        # Centerness decays with distance from the projected 3D center.
        rel_dist = torch.linalg.vector_norm(targets_3d[..., :2], dim=-1) / (
            1.414 * sample_stride[:, 0]
        )
        ctr_targets = torch.exp(-self.centerness_alpha * rel_dist)

        return (
            labels,
            targets_2d,
            labels_3d,
            targets_3d,
            ctr_targets,
            attrs,
        )

    def forward(
        self,
        cls_scores,
        bbox_preds,
        gt_bboxes_list,
        gt_labels_list,
        gt_bboxes_3d_list,
        gt_labels_3d_list,
        centers2d_list,
        depths_list,
        attr_labels_list,
    ):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        points = get_points(
            featmap_sizes,
            self.strides,
            bbox_preds[0].dtype,
            bbox_preds[0].device,
        )
        assert len(points) == len(self.regress_ranges)

        # Expand each level's regress range onto every point of that level.
        lvl_ranges = []
        for pts, rng in zip(points, self.regress_ranges):
            lvl_ranges.append(pts.new_tensor(rng).view(1, 2).expand_as(pts))
        all_ranges = torch.cat(lvl_ranges, dim=0)
        all_points = torch.cat(points, dim=0)
        pts_per_lvl = [pts.size(0) for pts in points]

        if attr_labels_list is None:
            attr_labels_list = [
                labels.new_full(labels.shape, self.attr_background_label)
                for labels in gt_labels_list
            ]

        (
            _,
            _,
            labels_3d_list,
            bbox_targets_3d_list,
            centerness_targets_list,
            attr_targets_list,
        ) = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            gt_bboxes_3d_list,
            gt_labels_3d_list,
            centers2d_list,
            depths_list,
            attr_labels_list,
            points=all_points,
            regress_ranges=all_ranges,
            num_points_per_lvl=pts_per_lvl,
        )

        # Split per-image targets back into FPN levels, then concat over batch.
        labels_3d_list = [
            lab.split(pts_per_lvl, dim=0) for lab in labels_3d_list
        ]
        bbox_targets_3d_list = [
            box.split(pts_per_lvl, dim=0) for box in bbox_targets_3d_list
        ]
        centerness_targets_list = [
            ctr.split(pts_per_lvl, dim=0) for ctr in centerness_targets_list
        ]
        attr_targets_list = [
            attr.split(pts_per_lvl, dim=0) for attr in attr_targets_list
        ]

        out_labels, out_boxes, out_ctr, out_attr = [], [], [], []
        for lvl, stride in enumerate(self.strides):
            out_labels.append(
                torch.cat([per_img[lvl] for per_img in labels_3d_list], dim=0)
            )
            out_ctr.append(
                torch.cat(
                    [per_img[lvl] for per_img in centerness_targets_list],
                    dim=0,
                )
            )
            out_attr.append(
                torch.cat(
                    [per_img[lvl] for per_img in attr_targets_list], dim=0
                )
            )
            boxes_lvl = torch.cat(
                [per_img[lvl] for per_img in bbox_targets_3d_list], dim=0
            )
            if self.norm_on_bbox:
                boxes_lvl = boxes_lvl.clone()
                boxes_lvl[:, :2] = boxes_lvl[:, :2] / stride
            out_boxes.append(boxes_lvl)

        return out_labels, out_boxes, out_ctr, out_attr
