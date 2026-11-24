# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from hat_engine.registry import OBJECT_REGISTRY

from heal.core.box3d_utils import xywhr2xyxyr
from heal.core.nms.box3d_nms import nms_bev
from .bbox_coders import CenterPointBBoxCoder

__all__ = ["CenterPointPostProcess"]


@OBJECT_REGISTRY.register
class CenterPointPostProcess(nn.Module):
    """CenterPoint post-process with rotate BEV NMS for BEV vtv2."""

    def __init__(
        self,
        tasks: Optional[List[dict]] = None,
        norm_bbox: bool = True,
        bbox_coder: Optional[CenterPointBBoxCoder] = None,
        score_threshold: float = 0.0,
        post_center_limit_range: Optional[List[float]] = None,
        out_size_factor: int = 1,
        pre_max_size: int = 1000,
        post_max_size: int = 83,
        nms_thr: float = 0.2,
        box_size: Optional[int] = 9,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = [len(t["class_names"]) for t in tasks]
        self.norm_bbox = norm_bbox
        self.bbox_coder = bbox_coder
        self.post_center_limit_range = post_center_limit_range
        self.score_threshold = score_threshold
        self.nms_thr = nms_thr
        self.pre_max_size = pre_max_size
        self.post_max_size = post_max_size
        self.box_size = box_size
        self.out_size_factor = out_size_factor

    @torch.no_grad()
    def forward(self, preds_dicts, meta=None):
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            num_class_with_bg = self.num_classes[task_id]
            batch_heatmap = preds_dict["heatmap"].sigmoid()
            batch_reg = preds_dict["reg"]
            batch_hei = preds_dict["height"]
            batch_dim = (
                torch.exp(preds_dict["dim"])
                if self.norm_bbox
                else preds_dict["dim"]
            )
            batch_rots = preds_dict["rot"][:, 0].unsqueeze(1)
            batch_rotc = preds_dict["rot"][:, 1].unsqueeze(1)
            batch_vel = preds_dict["vel"] if "vel" in preds_dict else None
            temp = self.bbox_coder.decode(
                batch_heatmap,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                task_id=task_id,
            )
            batch_reg_preds = [box["bboxes"] for box in temp]
            batch_cls_preds = [box["scores"] for box in temp]
            batch_cls_labels = [box["labels"] for box in temp]
            rets.append(
                self.get_task_detections(
                    num_class_with_bg,
                    batch_cls_preds,
                    batch_reg_preds,
                    batch_cls_labels,
                )
            )

        num_samples = len(rets[0])
        ret_list = []
        for i in range(num_samples):
            ret = {}
            for k in rets[0][i].keys():
                if k == "bboxes":
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
                elif k == "scores":
                    ret[k] = torch.cat([ret[i][k] for ret in rets])
                elif k == "labels":
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    ret[k] = torch.cat([ret[i][k].int() for ret in rets])
            ret_list.append(ret)
        return ret_list

    def get_task_detections(
        self,
        num_class_with_bg: int,
        batch_cls_preds: List[torch.Tensor],
        batch_reg_preds: List[torch.Tensor],
        batch_cls_labels: List[torch.Tensor],
    ):
        predictions_dicts = []
        post_center_range = self.post_center_limit_range
        if len(post_center_range) > 0:
            post_center_range = torch.tensor(
                post_center_range,
                dtype=batch_reg_preds[0].dtype,
                device=batch_reg_preds[0].device,
            )

        for _, (box_preds, cls_preds, cls_labels) in enumerate(
            zip(batch_reg_preds, batch_cls_preds, batch_cls_labels)
        ):
            if num_class_with_bg == 1:
                top_scores = cls_preds.squeeze(-1)
                top_labels = torch.zeros(
                    cls_preds.shape[0],
                    device=cls_preds.device,
                    dtype=torch.long,
                )
            else:
                top_labels = cls_labels.long()
                top_scores = cls_preds.squeeze(-1)

            if self.score_threshold > 0.0:
                thresh = torch.tensor(
                    [self.score_threshold], device=cls_preds.device
                ).type_as(cls_preds)
                top_scores_keep = top_scores >= thresh
                top_scores = top_scores.masked_select(top_scores_keep)

            if top_scores.shape[0] != 0:
                if self.score_threshold > 0.0:
                    box_preds = box_preds[top_scores_keep]
                    top_labels = top_labels[top_scores_keep]

                boxes = box_preds[:, [0, 1, 4, 3, 6]]
                boxes[:, -1] = -boxes[:, -1] - np.pi / 2
                boxes_for_nms = xywhr2xyxyr(boxes)
                selected = nms_bev(
                    boxes_for_nms,
                    top_scores,
                    thresh=self.nms_thr,
                    pre_max_size=self.pre_max_size,
                    post_max_size=self.post_max_size,
                )
            else:
                selected = []

            selected_boxes = box_preds[selected]
            selected_labels = top_labels[selected]
            selected_scores = top_scores[selected]

            if selected_boxes.shape[0] != 0:
                final_box_preds = selected_boxes
                final_scores = selected_scores
                final_labels = selected_labels
                if post_center_range is not None:
                    mask = (
                        final_box_preds[:, :3] >= post_center_range[:3]
                    ).all(1)
                    mask &= (
                        final_box_preds[:, :3] <= post_center_range[3:]
                    ).all(1)
                    predictions_dict = {
                        "bboxes": final_box_preds[mask],
                        "scores": final_scores[mask],
                        "labels": final_labels[mask],
                    }
                else:
                    predictions_dict = {
                        "bboxes": final_box_preds,
                        "scores": final_scores,
                        "labels": final_labels,
                    }
            else:
                dtype = batch_reg_preds[0].dtype
                device = batch_reg_preds[0].device
                predictions_dict = {
                    "bboxes": torch.zeros(
                        [0, self.box_size],
                        dtype=dtype,
                        device=device,
                    ),
                    "scores": torch.zeros([0], dtype=dtype, device=device),
                    "labels": torch.zeros(
                        [0], dtype=top_labels.dtype, device=device
                    ),
                }

            predictions_dicts.append(predictions_dict)
        return predictions_dicts
