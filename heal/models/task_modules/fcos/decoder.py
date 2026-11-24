# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import Any, Dict, Optional, Sequence

import torch
from hat_engine.registry import OBJECT_REGISTRY

from heal.models.base_modules.postprocess import PostProcessorBase
from .target import distance2bbox, get_points

try:
    from torchvision.ops.boxes import batched_nms
except ImportError:
    batched_nms = None

__all__ = ["FCOSDecoder", "multiclass_nms"]


@OBJECT_REGISTRY.register
class FCOSDecoder(PostProcessorBase):
    """FCOS post-process decoder: decode head outputs to bboxes and run NMS."""

    def __init__(
        self,
        num_classes: int,
        strides: Sequence[int],
        transforms: Optional[Sequence[dict]] = None,
        inverse_transform_key: Optional[Sequence[str]] = None,
        nms_use_centerness: bool = True,
        nms_sqrt: bool = True,
        test_cfg: Optional[dict] = None,
        truncate_bbox: bool = True,
        filter_score_mul_centerness: bool = False,
        label_offset: int = 0,
        upscale_bbox_pred: bool = False,
        bbox_relu: bool = False,
    ):
        """Initialize FCOS decoder.

        Args:
            num_classes: Number of foreground classes (without background).
            strides: Feature map stride for each FPN level.
            transforms: Optional post transforms with ``inverse_transform``.
            inverse_transform_key: Keys from ``meta_data`` for inverse
                transforms.
            nms_use_centerness: Use centerness as ``score_factors`` in NMS.
            nms_sqrt: Apply sqrt to ``scores * centerness`` before NMS.
            test_cfg: Inference config (``nms_pre``, ``score_thr``, ``nms``,
                ``max_per_img``).
            truncate_bbox: Clip decoded bbox to image/pad shape.
            filter_score_mul_centerness: Threshold on fused score after
                centerness fusion.
            label_offset: Label offset added to output class id.
            upscale_bbox_pred: Multiply bbox prediction by stride before
                decode.
            bbox_relu: Apply ReLU on bbox preds before decode.
        """
        super(FCOSDecoder, self).__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.transforms = transforms
        self.inverse_transform_key = inverse_transform_key
        self.nms_use_centerness = nms_use_centerness
        self.nms_sqrt = nms_sqrt
        self.test_cfg = test_cfg or {}
        self.truncate_bbox = truncate_bbox
        self.filter_score_mul_centerness = filter_score_mul_centerness
        self.label_offset = label_offset
        self.upscale_bbox_pred = upscale_bbox_pred
        self.bbox_relu = bbox_relu

    def forward(self, pred: Sequence[torch.Tensor], meta_data: Dict[str, Any]):
        """Decode multi-level FCOS predictions for a batch.

        Args:
            pred: Model outputs `(cls_scores, bbox_preds, centernesses)`.
            meta_data: Per-image metadata for shape recovery and inverse
                transforms.

        Returns:
            Dict with decoded detection results and optional image ids.
        """
        cls_scores, bbox_preds, centernesses = pred
        if self.bbox_relu:
            bbox_preds = [torch.nn.functional.relu(i) for i in bbox_preds]
        if self.upscale_bbox_pred:
            bbox_preds = list(bbox_preds)
            for i in range(len(bbox_preds)):
                bbox_preds[i] = bbox_preds[i] * self.strides[i]
        num_levels = len(cls_scores)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = get_points(
            featmap_sizes,
            self.strides,
            bbox_preds[0].dtype,
            bbox_preds[0].device,
        )
        results = {}
        det_results = []
        for img_id in range(bbox_preds[0].shape[0]):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = (
                [centernesses[i][img_id].detach() for i in range(num_levels)]
                if centernesses is not None
                else [None] * num_levels
            )
            h_index = meta_data["layout"][img_id].index("h")
            w_index = meta_data["layout"][img_id].index("w")
            if "pad_shape" in meta_data:
                max_shape = (
                    meta_data["pad_shape"][img_id][h_index],
                    meta_data["pad_shape"][img_id][w_index],
                )
            elif "img_shape" in meta_data:
                max_shape = (
                    meta_data["img_shape"][img_id][h_index],
                    meta_data["img_shape"][img_id][w_index],
                )
            else:
                max_shape = (
                    meta_data["img_height"][img_id],
                    meta_data["img_width"][img_id],
                )
            max_shape = max_shape if self.truncate_bbox else None
            inverse_info = {}
            if self.inverse_transform_key:
                for key in self.inverse_transform_key:
                    if key in meta_data:
                        inverse_info[key] = meta_data[key][img_id]
            det_bboxes = self._decode_single(
                cls_score_list,
                bbox_pred_list,
                centerness_pred_list,
                mlvl_points,
                max_shape,
                inverse_info,
            )
            det_results.append(det_bboxes)
        results["pred_bboxes"] = det_results
        results["img_name"] = meta_data.get("img_name")
        results["img_id"] = meta_data.get("img_id")
        return results

    def _decode_single(
        self,
        cls_score_list,
        bbox_pred_list,
        centerness_pred_list,
        mlvl_points,
        max_shape,
        inverse_info,
    ):
        """Decode one image's predictions and apply NMS.

        Args:
            cls_score_list: Per-level class logits for a single image.
            bbox_pred_list: Per-level bbox distance preds for one image.
            centerness_pred_list: Per-level centerness preds, or None.
            mlvl_points: Per-level anchor points for feature locations.
            max_shape: Optional ``(h, w)`` used to clip decoded boxes.
            inverse_info: Metadata passed to inverse transforms.

        Returns:
            Tensor ``(N, 6)`` as ``[x1, y1, x2, y2, score, label]``.
        """
        cfg = self.test_cfg
        mlvl_bboxes = []
        mlvl_scores = []
        has_centerness = centerness_pred_list[0] is not None
        if has_centerness:
            mlvl_centerness = []
        for cls_score, bbox_pred, centerness, points in zip(
            cls_score_list, bbox_pred_list, centerness_pred_list, mlvl_points
        ):
            scores = (
                cls_score.permute(1, 2, 0)
                .reshape(-1, self.num_classes)
                .sigmoid()
            )
            if has_centerness:
                centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get("nms_pre", -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if has_centerness:
                    max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                else:
                    max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                if has_centerness:
                    centerness = centerness[topk_inds]
            bboxes = distance2bbox(points, bbox_pred, max_shape=max_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            if has_centerness:
                mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if self.transforms:
            for transform in self.transforms[::-1]:
                if hasattr(transform, "inverse_transform"):
                    mlvl_bboxes = transform.inverse_transform(
                        inputs=mlvl_bboxes,
                        task_type="detection",
                        inverse_info=inverse_info,
                    )
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = (
            torch.cat(mlvl_centerness) if has_centerness else None
        )
        score_thr = cfg.get("score_thr", 0.05)
        if isinstance(score_thr, str):
            score_thr = float(score_thr)
        nms_cfg = cfg.get("nms", {})
        nms = nms_cfg.get("name", "nms")
        iou_threshold = nms_cfg.get("iou_threshold", 0.6)
        max_per_img = cfg.get("max_per_img", 100)
        det_bboxes = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            score_thr,
            nms,
            iou_threshold,
            max_per_img,
            score_factors=mlvl_centerness if self.nms_use_centerness else None,
            nms_sqrt=self.nms_sqrt,
            filter_score_mul_centerness=self.filter_score_mul_centerness,
            label_offset=self.label_offset,
        )
        return det_bboxes


def multiclass_nms(
    multi_bboxes,
    multi_scores,
    score_thr,
    nms,
    iou_threshold,
    max_per_img=-1,
    score_factors=None,
    nms_sqrt=False,
    filter_score_mul_centerness=False,
    label_offset=0,
):
    """Run class-wise NMS on decoded bboxes.

    Args:
        multi_bboxes: Decoded bboxes ``(num_boxes, 4)`` or
            ``(num_boxes, num_classes*4)``.
        multi_scores: Class scores ``(num_boxes, num_classes + 1)``.
        score_thr: Score threshold for candidate filtering.
        nms: NMS algorithm name. Currently only ``"nms"`` is supported.
        iou_threshold: IoU threshold for NMS.
        max_per_img: Max detections per image. ``-1`` means no limit.
        score_factors: Optional factors (e.g. centerness) for class scores.
        nms_sqrt: Apply sqrt after score-factor fusion.
        filter_score_mul_centerness: Threshold on fused scores.
        label_offset: Value added to output class labels.

    Returns:
        Tensor ``(N, 6)`` as ``[x1, y1, x2, y2, score, label]``.
    """
    if batched_nms is None:
        raise RuntimeError("torchvision is required for multiclass_nms")
    num_classes = multi_scores.size(1) - 1
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4
        )
    scores = multi_scores[:, :-1]
    if score_factors is not None:
        if not nms_sqrt:
            scores_new = scores * score_factors[:, None]
        else:
            scores_new = torch.sqrt(scores * score_factors[:, None])
    else:
        scores_new = scores
    if filter_score_mul_centerness and score_factors is not None:
        valid_mask = scores_new > score_thr
        scores = scores_new
    elif score_factors is not None:
        valid_mask = scores > score_thr
        scores = scores_new
    else:
        valid_mask = scores > score_thr
    bboxes = torch.masked_select(
        bboxes,
        torch.stack((valid_mask, valid_mask, valid_mask, valid_mask), -1),
    ).view(-1, 4)
    scores = torch.masked_select(scores, valid_mask)
    labels = valid_mask.nonzero()[:, 1] + label_offset

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        labels = multi_bboxes.new_zeros((0, 1), dtype=torch.float32)
        return torch.cat((bboxes, labels), dim=-1)

    if nms == "nms":
        keep = batched_nms(bboxes, scores, labels, iou_threshold)
    else:
        raise NotImplementedError(f"nms type {nms} not implemented")
    dets = torch.cat([bboxes[keep], scores[keep][:, None]], -1)
    if max_per_img > 0:
        dets = dets[:max_per_img]
        keep = keep[:max_per_img]
    return torch.cat((dets, labels[keep].view(-1, 1).float()), dim=-1)
