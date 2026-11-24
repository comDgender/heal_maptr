# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.apply_func import limit_period
from hat_engine.utils.model_helpers import fx_wrap

from heal.core.box_utils import bbox_overlaps, center_to_minmax_2d
from heal.models.task_modules.lidar.box_coders import GroundBox3dCoder

__all__ = ["LidarTargetAssigner"]


@OBJECT_REGISTRY.register
class LidarTargetAssigner(nn.Module):
    """TargetAssigner for Lidar.

    Args:
        box_coder: BoxCoder.
        class_names: Class names.
        positive_fraction: Positive fraction.
        sample_size: Sample size.
    """

    def __init__(
        self,
        box_coder: GroundBox3dCoder,
        class_names: List[str],
        positive_fraction: int = None,
        sample_size: int = 512,
    ):

        super(LidarTargetAssigner, self).__init__()
        self._box_coder = box_coder
        self._sample_size = sample_size
        self._class_names = class_names
        if positive_fraction is None or positive_fraction < 0:
            self._positive_fraction = None
        else:
            self._positive_fraction = positive_fraction

    @property
    def box_coder(self):
        """3D boxCoder."""
        return self._box_coder

    @fx_wrap(skip_compile=True)
    def forward(
        self,
        anchors_list: List[torch.Tensor],
        matched_thresholds: List[float],
        unmatched_thresholds: List[float],
        annos: Dict,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """Forward pass, generate targets.

        Args:
            anchors_list: List of anchors.
            match_thresholds: Match thresholds of IoU.
            unmatch_thresholds: Unmatch thresholds of IoU.
            annos: Annotations of ground truth.
            device: The device on which the target will be generated.
        Returns:
            bbox_targets: BBox targets.
            cls_labels: Classification label for bbox.
            reg_weights: Regression weights for each bbox.
        """
        return self.assign_targets(
            anchors_list,
            matched_thresholds,
            unmatched_thresholds,
            annos,
            device,
        )

    def assign_targets(
        self,
        anchors_list: List[torch.Tensor],
        matched_thresholds: List[float],
        unmatched_thresholds: List[float],
        annos: Dict,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """Generate targets.

        Args:
            anchors_list: List of anchors.
            match_thresholds: Match thresholds of IoU.
            unmatch_thresholds: Unmatch thresholds of IoU.
            annos: Annotations of ground truth.
            device: The device on which the target will be generated.
        Returns:
            bbox_targets: BBox targets.
            cls_labels: Classification label for bbox.
            reg_weights: Regression weights for each bbox.
        """
        batch_size = len(annos)

        bbox_targets = []
        cls_labels = []
        reg_weights = []

        for idx in range(batch_size):
            gt_boxes = torch.tensor(
                annos[idx]["gt_boxes"], dtype=torch.float32, device=device
            )
            gt_classes = torch.tensor(annos[idx]["gt_classes"], device=device)
            gt_names = annos[idx]["gt_names"]
            targets_dict = self.assign_per_class(
                classes_names=self._class_names,
                anchors_list=anchors_list,
                matched_thresholds=matched_thresholds,
                unmatched_thresholds=unmatched_thresholds,
                gt_boxes=gt_boxes,
                gt_classes=gt_classes,
                gt_names=gt_names,
            )
            bbox_targets.append(targets_dict["bbox_targets"])
            cls_labels.append(targets_dict["labels"])
            reg_weights.append(targets_dict["reg_weights"])

        bbox_targets = torch.stack(bbox_targets, dim=0)
        cls_labels = torch.stack(cls_labels, dim=0)
        reg_weights = torch.stack(reg_weights, dim=0)

        return bbox_targets, cls_labels, reg_weights

    def assign_per_class(
        self,
        classes_names,
        anchors_list,
        matched_thresholds,
        unmatched_thresholds,
        gt_boxes,
        gt_classes,
        gt_names,
    ):
        """Assign targets for each class.

        Args:
            classes_names: Class names.
            anchors_list: List of anchors.
            match_thresholds: Match thresholds of IoU.
            unmatch_thresholds: Unmatch thresholds of IoU.
            gt_boxes: Ground truth boxes.
            gt_classes: Ground truth classes.
            gt_names: Names of Ground truth.
        Returns:
            labels: Bbox classification label.
            bbox_targets: Bbox.
            reg_weights: Regression weights for each bbox.
        """

        def similarity_fn(anchors, gt_boxes):
            anchors_rbv = anchors[:, [0, 1, 3, 4, -1]]
            gt_boxes_rbv = gt_boxes[:, [0, 1, 3, 4, -1]]
            return self.nearest_iou_similarity(anchors_rbv, gt_boxes_rbv)

        def box_encoding_fn(boxes, anchors):
            return self._box_coder.encode(boxes, anchors)

        targets_list = []
        for class_name, anchors, match_threshold, unmatch_threshold in zip(
            classes_names,
            anchors_list,
            matched_thresholds,
            unmatched_thresholds,
        ):
            mask = torch.tensor(
                [c == class_name for c in gt_names], dtype=torch.bool
            )
            feature_map_size = anchors.shape[:3]
            # num_loc = anchors.shape[-2]

            targets = self.create_targets_single(
                anchors.reshape(-1, self.box_coder.code_size),
                torch.nan_to_num(gt_boxes[mask]),
                similarity_fn,
                box_encoding_fn,
                gt_classes=gt_classes[mask],
                matched_threshold=match_threshold,
                unmatched_threshold=unmatch_threshold,
                positive_fraction=self._positive_fraction,
                sample_size=self._sample_size,
                norm_by_num_examples=False,
                box_code_size=self.box_coder.code_size,
            )
            targets_list.append(targets)

        targets_dict = {
            "labels": [t["box_cls_labels"] for t in targets_list],
            "bbox_targets": [t["bbox_reg_targets"] for t in targets_list],
            "reg_weights": [t["reg_weights"] for t in targets_list],
        }
        targets_dict["bbox_targets"] = torch.cat(
            [
                v.reshape(*feature_map_size, -1, self.box_coder.code_size)
                for v in targets_dict["bbox_targets"]
            ],
            dim=-2,
        ).view(-1, self.box_coder.code_size)
        targets_dict["labels"] = torch.cat(
            [v.reshape(*feature_map_size, -1) for v in targets_dict["labels"]],
            dim=-1,
        ).view(-1)
        targets_dict["reg_weights"] = torch.cat(
            [
                v.reshape(*feature_map_size, -1)
                for v in targets_dict["reg_weights"]
            ],
            dim=-1,
        ).view(-1)

        return targets_dict

    @property
    def num_anchors_per_location(self):
        """Get number of anchors per location."""
        num = 0
        for a_generator in self._anchor_generators:
            num += a_generator.num_anchors_per_localization
        return num

    @property
    def box_ndim(self):
        """Dimension of box."""
        return self.box_coder.code_size

    def create_targets_single(
        self,
        all_anchors: torch.Tensor,
        gt_boxes: torch.Tensor,
        similarity_fn: Callable,
        box_encoding_fn: Callable,
        gt_classes: Optional[torch.Tensor] = None,
        matched_threshold: float = 0.6,
        unmatched_threshold: float = 0.45,
        positive_fraction: Optional[float] = None,
        sample_size: int = 300,
        norm_by_num_examples: bool = False,
        box_code_size: int = 7,
    ):
        """Create targets.

        Args:
            all_anchors: Anchor boxes, shape ``(A, box_ndim)``.
            gt_boxes: Ground-truth boxes, shape ``(G, box_ndim)``.
            similarity_fn: Callable ``(anchors, gt_boxes) -> (A, G)``
                similarity (e.g. IoU).
            box_encoding_fn: Callable ``(gt_boxes, anchors) -> encodings``
                that produces regression targets / offsets.
            prune_anchor_fn: Optional filter returning indices of anchors
                that remain valid for assignment.
            gt_classes: Per-GT class ids with shape ``(G,)``; values must
                be 1-based (background is not included).
            matched_threshold: Anchors with IoU at or above this value are
                marked positive.
            unmatched_threshold: Anchors with IoU below this value are
                marked negative / background.
            positive_fraction: Optional ``[0, 1]`` ratio. When set, sampling
                aims for this positive share among the batch;
                if there is not enough positives, it fills the rest with
                negatives.
            rpn_batch_size: int. sample size.
            norm_by_num_examples: bool. norm box_weight by number of examples.
        Returns:
            box_cls_labels: Bbox classification label.
            bbox_reg_targets: Bbox.
            reg_weights: Regression weights for each bbox.
        """

        total_anchors = all_anchors.shape[0]
        anchors = all_anchors
        num_inside = total_anchors
        # box_ndim = all_anchors.shape[1]
        if gt_classes is None:
            gt_classes = torch.ones(
                [gt_boxes.shape[0]], dtype=torch.int32, device=anchors.device
            )
        # Assignment codes: >0 positive class, 0 background, -1 ignore.
        labels = torch.full(
            (num_inside,), -1, dtype=torch.int32, device=anchors.device
        )
        gt_ids = torch.full(
            (num_inside,), -1, dtype=torch.int32, device=anchors.device
        )

        if len(gt_boxes) > 0:
            # Pairwise similarity matrix between anchors and GTs.
            anchor_by_gt_overlap = similarity_fn(anchors, gt_boxes)
            # Best GT index / score for every anchor.
            anchor_to_gt_argmax = torch.argmax(anchor_by_gt_overlap, dim=1)
            row_ids = torch.arange(num_inside, device=anchors.device)
            anchor_to_gt_max = anchor_by_gt_overlap[
                row_ids, anchor_to_gt_argmax
            ]
            # Best anchor index / score for every GT.
            gt_to_anchor_argmax = torch.argmax(anchor_by_gt_overlap, dim=0)
            col_ids = torch.arange(
                anchor_by_gt_overlap.shape[1], device=anchors.device
            )
            gt_to_anchor_max = anchor_by_gt_overlap[
                gt_to_anchor_argmax, col_ids
            ]
            # must remove gt which doesn't match any anchor.
            empty_gt_mask = gt_to_anchor_max == 0
            gt_to_anchor_max[empty_gt_mask] = -1
            # Find all anchors that share the max overlap amount
            # (this includes many ties)
            anchors_with_max_overlap = (
                anchor_by_gt_overlap == gt_to_anchor_max
            ).nonzero()[:, 0]
            # Force-match: assign every GT to its highest-overlap anchors.
            forced_gt = anchor_to_gt_argmax[anchors_with_max_overlap]
            labels[anchors_with_max_overlap] = gt_classes[forced_gt]
            gt_ids[anchors_with_max_overlap] = forced_gt.to(torch.int32)
            # Threshold match: IoU >= matched_threshold -> positive.
            pos_mask = anchor_to_gt_max >= matched_threshold
            matched_gt = anchor_to_gt_argmax[pos_mask]
            labels[pos_mask] = gt_classes[matched_gt]
            gt_ids[pos_mask] = matched_gt.to(torch.int32)
            bg_inds = torch.nonzero(
                anchor_to_gt_max < unmatched_threshold, as_tuple=False
            )[:, 0]
        else:
            bg_inds = torch.arange(num_inside, device=anchors.device)
        fg_inds = torch.nonzero(labels > 0, as_tuple=False)[:, 0]
        fg_max_overlap = None  # noqa F841
        if len(gt_boxes) > 0:
            fg_max_overlap = anchor_to_gt_max[fg_inds]  # noqa F841
        gt_pos_ids = gt_ids[fg_inds]  # noqa F841

        # Subsample positives / fill negatives when a fraction is requested.
        if positive_fraction is not None:
            max_fg = int(positive_fraction * sample_size)
            if len(fg_inds) > max_fg:
                n_disable = len(fg_inds) - max_fg
                # Indices into the current fg_inds list (same as legacy path).
                local_drop = torch.randperm(len(fg_inds))[:n_disable]
                labels[local_drop] = -1
                fg_inds = torch.nonzero(labels > 0, as_tuple=False)[:, 0]

            neg_budget = sample_size - (labels > 0).sum()
            if len(bg_inds) > neg_budget:
                kept_bg = bg_inds[
                    torch.randint(0, len(bg_inds), size=(neg_budget,))
                ]
                labels[kept_bg] = 0
        else:
            if len(gt_boxes) == 0:
                labels.fill_(0)
            else:
                labels[bg_inds] = 0
                # Force-matched anchors stay positive even if in the BG band.
                labels[anchors_with_max_overlap] = gt_classes[forced_gt]

        bbox_targets = anchors.new_zeros((num_inside, box_code_size))

        if len(gt_boxes) > 0:
            pos_gt = gt_boxes[anchor_to_gt_argmax[fg_inds]]
            pos_anchors = anchors[fg_inds]
            bbox_targets[fg_inds] = box_encoding_fn(
                pos_gt.float(), pos_anchors
            )

        reg_weights = anchors.new_zeros((num_inside,))
        # Uniform weight across valid (non-ignore) samples when normalizing.
        if norm_by_num_examples:
            num_examples = (labels >= 0).sum()  # neg + pos
            num_examples = num_examples if num_examples > 1.0 else 1.0
            reg_weights[labels > 0] = 1.0 / num_examples
        else:
            reg_weights[labels > 0] = 1.0

        ret_dict = {
            "box_cls_labels": labels,
            "bbox_reg_targets": bbox_targets,
            "reg_weights": reg_weights,
            # "assigned_anchors_overlap": fg_max_overlap,
            # "positive_gt_id": gt_pos_ids,
            # "assigned_anchors_inds": fg_inds,
        }

        # ret["assigned_anchors_inds"] = fg_inds
        return ret_dict

    def nearest_iou_similarity(self, boxes1, boxes2):
        """Compute matrix of (negated) sq distances.

        Args:
          boxlist1: BoxList holding N boxes.
          boxlist2: BoxList holding M boxes.

        Returns:
          A tensor with shape [N, M] representing negated pairwise
          squared distance.
        """
        boxes1_bv = self._rbbox2d_to_near_bbox(boxes1)
        boxes2_bv = self._rbbox2d_to_near_bbox(boxes2)
        ret = bbox_overlaps(boxes1_bv, boxes2_bv)
        return ret

    def _rbbox2d_to_near_bbox(
        self,
        in_boxes: torch.Tensor,
        box_mode: str = "wlh",
        rect: bool = False,
    ):
        """Convert rotated bbox to nearest 'standing' or 'lying' bbox.

        Args:
            inboxes: [N, 5(x, y, w, l, ry)] or [N, 7(x,y,z,w,l,h,ry)]
        Returns:
            outboxes: [N, 4(xmin, ymin, xmax, ymax)] bboxes
        """

        rots = in_boxes[..., -1]
        # limit ry in range np.abs([-np.pi/2., np.pi/2.])
        rots_0_pi_div_2 = torch.abs(limit_period(rots, 0.5, np.pi))
        # this line aims to rotate the box to a vertial or horizonal
        # direction with abs(angle) less than 45'.
        cond = (rots_0_pi_div_2 > np.pi / 4).unsqueeze(-1)
        in_boxes_center = torch.where(
            cond, in_boxes[:, [0, 1, 3, 2]], in_boxes[:, :4]
        )  # if True, change w and l; otherwise keep the same;
        out_boxes = torch.zeros(
            [in_boxes.shape[0], 4],
            dtype=in_boxes.dtype,
            device=in_boxes.device,
        )
        out_boxes[:, :4] = center_to_minmax_2d(
            in_boxes_center[:, :2], in_boxes_center[:, 2:]
        )
        return out_boxes
