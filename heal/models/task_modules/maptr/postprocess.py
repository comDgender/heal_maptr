# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import logging
from typing import Dict, List, Optional

import torch
from hat_engine.registry import OBJECT_REGISTRY
from torch import Tensor, nn
from torch.amp import autocast

from heal.core.box_utils import box_center_to_corner

logger = logging.getLogger(__name__)


def denormalize_2d_bbox(bboxes, pc_range):

    bboxes = box_center_to_corner(bboxes)
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


class MapTRNMSFreeCoder(object):
    """Bbox coder for NMS-free detector.

    Args:
        pc_range: Range of point cloud.
        post_center_range: Limit of the center.
            Default: None.
        max_num: Max number to be kept. Default: 100.
        score_threshold: Threshold to filter boxes based on score.
            Default: None.
        num_classes: Number of classes. Default: 10.
    """

    def __init__(
        self,
        pc_range: List[float],
        post_center_range: List[float] = None,
        max_num: int = 100,
        score_threshold: float = None,
        num_classes: int = 10,
        pred_absolute_points: bool = False,
    ):
        self.pc_range = pc_range
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes
        self.pred_absolute_points = pred_absolute_points

    def encode(self):
        pass

    def decode_single(
        self, cls_scores: Tensor, bbox_preds: Tensor, pts_preds: Tensor
    ) -> Dict:
        """Decode one sample: top-k queries -> denorm boxes / pts + filters."""
        k = self.max_num
        prob = cls_scores.sigmoid().reshape(-1)
        scores, flat_idx = prob.topk(k)
        # flat index = query * num_classes + class
        class_ids = flat_idx % self.num_classes
        query_ids = flat_idx // self.num_classes
        box_raw = bbox_preds[query_ids]
        pts_raw = pts_preds[query_ids]

        if self.pred_absolute_points:
            boxes = box_raw
            pts = pts_raw
        else:
            boxes = denormalize_2d_bbox(box_raw, self.pc_range)
            pts = denormalize_2d_pts(pts_raw, self.pc_range)

        # Optional score gate; relax threshold if nothing survives.
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
        in_range = (boxes[..., :4] >= center_lim[:4]).all(1) & (
            boxes[..., :4] <= center_lim[4:]
        ).all(1)
        if self.score_threshold:
            in_range = in_range & score_mask

        return {
            "bboxes": boxes[in_range],
            "scores": scores[in_range],
            "labels": class_ids[in_range],
            "pts": pts[in_range],
        }

    def decode(self, preds_dicts):
        cls_all = preds_dicts["all_cls_scores"][-1]
        box_all = preds_dicts["all_bbox_preds"][-1]
        pts_all = preds_dicts["all_pts_preds"][-1]
        n_batch = cls_all.size(0)
        return [
            self.decode_single(cls_all[i], box_all[i], pts_all[i])
            for i in range(n_batch)
        ]


@OBJECT_REGISTRY.register
class MapTRPostProcess(nn.Module):
    """Post-processing module for MapTR.

    Args:
        pc_range: Range of the point cloud. Default: None.
        post_center_range: Limit of the center. Default: None.
        max_num: Maximum number of boxes to keep. Default: 100.
        score_threshold: Threshold to filter boxes based on score.
            Default: None.
        num_classes: Number of classes. Default: 10.
    """

    def __init__(
        self,
        pc_range: Optional[List[float]] = None,
        post_center_range: Optional[List[float]] = None,
        max_num: int = 100,
        score_threshold: Optional[float] = None,
        num_classes: int = 10,
        pred_absolute_points: bool = False,
    ):
        super(MapTRPostProcess, self).__init__()

        self.bbox_coder = MapTRNMSFreeCoder(
            pc_range,
            post_center_range=post_center_range,
            max_num=max_num,
            score_threshold=score_threshold,
            num_classes=num_classes,
            pred_absolute_points=pred_absolute_points,
        )
        self.pc_range = pc_range

    @autocast("cuda", enabled=False)
    def forward(self, preds_dicts, img_metas):
        if "one2many_outs" in preds_dicts:
            preds_dicts.pop("one2many_outs")
        if "pv_seg" in preds_dicts:
            preds_dicts.pop("pv_seg")
        if "dense_depth" in preds_dicts:
            preds_dicts.pop("dense_depth")
        for k, v in preds_dicts.items():
            if v is not None:
                preds_dicts[k] = v.float()

        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        bbox_results = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds["bboxes"]
            # bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            # code_size = bboxes.shape[-1]
            # bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds["scores"]
            labels = preds["labels"]
            pts = preds["pts"]

            result_dict = {
                "boxes_3d": bboxes.to("cpu"),
                "scores_3d": scores.cpu(),
                "labels_3d": labels.cpu(),
                "pts_3d": pts.to("cpu"),
            }

            bbox_results.append(result_dict)

        # import pdb;pdb.set_trace()
        return bbox_results
