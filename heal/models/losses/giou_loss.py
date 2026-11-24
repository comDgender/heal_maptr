# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import torch
import torch.nn as nn
from hat_engine.models.losses.utils import weight_reduce_loss
from hat_engine.registry import OBJECT_REGISTRY
from torch.amp import autocast

__all__ = ["GIoULoss"]


@OBJECT_REGISTRY.register
class GIoULoss(nn.Module):
    """Generalized IoU loss for detection.

    Args:
        loss_name: Key used in the returned loss dictionary.
        loss_weight: Global multiplier applied to the loss.
        eps: Small value used to avoid division by zero.
        reduction: Reduction method passed to loss reduction.
    """

    def __init__(self, loss_name, loss_weight=1.0, eps=1e-6, reduction="mean"):
        super(GIoULoss, self).__init__()
        self.loss_name = loss_name
        self.loss_weight = loss_weight
        self.eps = eps
        self.reduction = reduction

    @staticmethod
    def _cal_giou_loss(pred, target, eps=1e-6):
        """Calculate unreduced generalized IoU loss.

        Args:
            pred: Predicted boxes in ``xyxy`` format.
            target: Target boxes in ``xyxy`` format.
            eps: Small value used to avoid division by zero.

        Returns:
            torch.Tensor: Element-wise GIoU loss.
        """
        lt = torch.max(pred[..., :2], target[..., :2])
        rb = torch.min(pred[..., 2:4], target[..., 2:4])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[..., 0] * wh[..., 1]
        ap = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
        ag = (target[..., 2] - target[..., 0]) * (
            target[..., 3] - target[..., 1]
        )
        union = ap + ag - overlap + eps
        ious = overlap / union
        enclose_x1y1 = torch.min(pred[..., :2], target[..., :2])
        enclose_x2y2 = torch.max(pred[..., 2:4], target[..., 2:4])
        enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1] + eps
        gious = ious - (enclose_area - union) / enclose_area
        return 1 - gious

    @autocast("cuda", enabled=False)
    def forward(self, pred, target, weight=None, avg_factor=None):
        """Compute weighted generalized IoU loss.

        Args:
            pred: Predicted boxes in ``xyxy`` format.
            target: Target boxes in ``xyxy`` format.
            weight: Optional element-wise loss weights.
            avg_factor: Optional average factor for mean reduction.

        Returns:
            dict[str, torch.Tensor]: Loss dictionary keyed by ``loss_name``.
        """
        result_dict = {}
        result_dict[self.loss_name] = None
        if pred is not None:
            pred = pred.float()
            if weight is not None and not torch.any(weight > 0):
                return {self.loss_name: (pred * weight).sum()}
            if weight is not None and weight.dim() > 1:
                assert weight.shape == pred.shape
                weight = weight.mean(-1)
            loss = self._cal_giou_loss(pred, target, self.eps)
            loss = weight_reduce_loss(loss, weight, self.reduction, avg_factor)
            result_dict[self.loss_name] = self.loss_weight * loss
        return result_dict
