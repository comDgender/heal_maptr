# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import Any, Dict, List, Optional

import torch
from hat_engine.registry import OBJECT_REGISTRY


def _decode_box_params(raw: torch.Tensor) -> torch.Tensor:
    """Map network box encoding to (xyz, size, yaw, extra)."""
    yaw = torch.atan2(raw[:, 6], raw[:, 7])
    return torch.cat(
        [raw[:, 0:3], raw[:, 3:6].exp(), yaw[:, None], raw[:, 8:]],
        dim=-1,
    )


@OBJECT_REGISTRY.register
class SparseBEVDecoder(object):
    """
    Sparse BEV Decoder for object detection.

    Args:
        num_output: Number of top scoring predictions to output.
                    Defaults to 300.
        score_threshold: Threshold for filtering predictions based on score.
                         Defaults to None.
    """

    def __init__(
        self,
        num_output: int = 300,
        score_threshold: Optional[float] = None,
    ):
        super(SparseBEVDecoder, self).__init__()
        self.num_output = num_output
        self.score_threshold = score_threshold

    def _select_topk(self, cls_scores: torch.Tensor) -> tuple:
        probs = cls_scores.sigmoid()
        n_cls = probs.shape[-1]
        scores, flat_idx = probs.flatten(start_dim=1).topk(
            self.num_output, dim=1, sorted=True
        )
        labels = flat_idx % n_cls
        thr_mask = None
        if self.score_threshold is not None:
            thr_mask = scores >= self.score_threshold
        return scores, labels, flat_idx, n_cls, thr_mask

    def _apply_quality(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        flat_idx: torch.Tensor,
        thr_mask: Optional[torch.Tensor],
        quality: torch.Tensor,
        n_cls: int,
    ) -> tuple:
        ctr = torch.gather(quality[..., 0], 1, flat_idx // n_cls)
        scores = scores * ctr.sigmoid()
        scores, order = torch.sort(scores, dim=1, descending=True)
        labels = torch.gather(labels, 1, order)
        flat_idx = torch.gather(flat_idx, 1, order)
        if thr_mask is not None:
            thr_mask = torch.gather(thr_mask, 1, order)
        return scores, labels, flat_idx, thr_mask

    def __call__(
        self,
        cls_scores: torch.Tensor,
        box_preds: torch.Tensor,
        qulity: torch.Tensor = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform decoding of predictions into a dictionary of detections.

        Args:
            cls_scores: Predicted class scores,
                        shape (bs, num_pred, num_cls).
            box_preds: Predicted box coordinates,
                       shape (bs, num_pred, 4).
            qulity: Predicted quality scores,
                    shape (bs, num_pred). Defaults to None.

        Returns:
            Dictionary containing decoded detections with keys
            'scores', 'boxes', and optionally 'qulity'.
        """
        if isinstance(cls_scores, list):
            cls_scores = cls_scores[-1]
            box_preds = box_preds[-1]
            if qulity is not None:
                qulity = qulity[-1]

        scores, labels, flat_idx, n_cls, thr_mask = self._select_topk(
            cls_scores
        )
        if qulity is not None:
            scores, labels, flat_idx, thr_mask = self._apply_quality(
                scores, labels, flat_idx, thr_mask, qulity, n_cls
            )

        batch_out = []
        for bi in range(scores.shape[0]):
            lab = labels[bi]
            scr = scores[bi]
            boxes = box_preds[bi, flat_idx[bi] // n_cls]
            if self.score_threshold is not None:
                m = thr_mask[bi]
                lab, scr, boxes = lab[m], scr[m], boxes[m]
            batch_out.append(
                {
                    "bboxes": _decode_box_params(boxes).cpu(),
                    "scores": scr.cpu(),
                    "labels": lab.cpu(),
                }
            )
        return batch_out
