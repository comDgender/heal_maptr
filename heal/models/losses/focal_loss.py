# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import itertools
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from hat_engine.models.losses.utils import weight_reduce_loss
from hat_engine.registry import OBJECT_REGISTRY
from torch.amp import autocast

__all__ = ["FocalLoss", "GaussianFocalLoss", "LaneFastFocalLoss"]


@OBJECT_REGISTRY.register
class FocalLoss(nn.Module):
    """Sigmoid focal loss for detection.

    Supports two target formats:
    - ``index``: integer labels ``(N,)`` or ``(B, N)``, used by FCOS.
    - ``one_hot``: binary labels with the same shape as ``pred``, used by
      PointPillars.

    When ``loss_name`` is set, returns ``{loss_name: loss}``; otherwise returns
    the loss tensor directly.
    """

    def __init__(
        self,
        loss_name: Optional[str] = None,
        num_classes: Optional[int] = None,
        alpha: float = 0.25,
        gamma: float = 2.0,
        loss_weight: float = 1.0,
        eps: float = 1e-12,
        reduction: str = "mean",
        from_logits: bool = True,
        target_format: Literal["auto", "index", "one_hot"] = "auto",
        ohem_fp_threshold: Optional[float] = 1.0,
        ohem_fp_loss_weight: Optional[float] = None,
    ):
        super(FocalLoss, self).__init__()
        self.loss_name = loss_name
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.from_logits = from_logits
        self.target_format = target_format
        self.ohem_fp_threshold = ohem_fp_threshold
        self.ohem_fp_loss_weight = ohem_fp_loss_weight

    def _uses_index_target(self, target: torch.Tensor) -> bool:
        if self.target_format == "index":
            return True
        if self.target_format == "one_hot":
            return False
        return target.dim() == 1 or not target.is_floating_point()

    def _to_one_hot(self, target: torch.Tensor) -> torch.Tensor:
        if self.num_classes is None:
            raise ValueError(
                "num_classes is required when target uses index format."
            )
        target = target.clone()
        target[target < 0] = self.num_classes - 1
        one_hot = F.one_hot(target, self.num_classes)
        return one_hot[..., : self.num_classes - 1]

    @autocast("cuda", enabled=False)
    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        points_per_strides=None,
        valid_classes_list=None,
    ):
        pred = pred.float()
        if self.from_logits:
            pred = pred.sigmoid()

        uses_index = self._uses_index_target(target)
        if uses_index:
            one_hot = self._to_one_hot(target)
        else:
            one_hot = target

        pt = torch.where(torch.eq(one_hot, 1), pred, 1 - pred)
        t = torch.ones_like(one_hot)
        at = torch.where(
            torch.eq(one_hot, 1), self.alpha * t, (1 - self.alpha) * t
        )
        loss = (
            -at
            * torch.pow((1 - pt), self.gamma)
            * torch.log(torch.clamp(pt, min=self.eps, max=1.0))
        )

        if (
            uses_index
            and valid_classes_list is not None
            and loss.shape[-1] > 1
        ):
            valid_loss_mask = torch.zeros_like(loss)
            start_indexs = [0] + list(
                itertools.accumulate(
                    [xx * len(valid_classes_list) for xx in points_per_strides]
                )
            )[:-1]
            for str_id, points_per_stride in enumerate(points_per_strides):
                for ii, valid_classes in enumerate(valid_classes_list):
                    start_idx = start_indexs[str_id] + ii * points_per_stride
                    end_idx = (
                        start_indexs[str_id] + (ii + 1) * points_per_stride
                    )
                    valid_loss_mask[start_idx:end_idx][:, valid_classes] = 1.0
            loss = loss * valid_loss_mask

        if not uses_index and self.ohem_fp_loss_weight is not None:
            ohem_fp_mask = (target == 0) & (pred > self.ohem_fp_threshold)
            loss = torch.where(
                ohem_fp_mask, loss * self.ohem_fp_loss_weight, loss
            )

        if weight is not None:
            if weight.shape != loss.shape:
                if weight.size(0) == loss.size(0):
                    weight = weight.view(-1, 1)
                else:
                    assert weight.numel() == loss.numel()
                    weight = weight.view(loss.size(0), -1)
            assert weight.ndim == loss.ndim

        loss = weight_reduce_loss(loss, weight, self.reduction, avg_factor)
        loss = loss * self.loss_weight

        if self.loss_name is None:
            return loss
        return {self.loss_name: loss}


@OBJECT_REGISTRY.register
class GaussianFocalLoss(nn.Module):
    """Guassian focal loss.

    Args:
        alpha: A weighting factor for positive sample.
        gamma: Used in focal loss to balance contribution
                of easy examples and hard examples.
        loss_weight: Weight factor for guassian focal loss.
    """

    def __init__(
        self, alpha: float = 2.0, gamma: float = 4.0, loss_weight: float = 1.0
    ):
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss_weight = loss_weight

    def gaussian_focal_loss(self, pred, gaussian_target, grad_tensor):
        eps = 1e-10
        pos_pos = gaussian_target.eq(1)
        neg_pos = gaussian_target.lt(1)
        neg_weights = (1 - gaussian_target).pow(self.gamma)
        pos_loss = (
            -(pred + eps).log()
            * (1 - pred).pow(self.alpha)
            * pos_pos
            * grad_tensor
        )
        neg_loss = (
            -(1 - pred + eps).log()
            * pred.pow(self.alpha)
            * neg_weights
            * neg_pos
            * grad_tensor
        )
        nums_pos = pos_pos.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        loss = neg_loss if nums_pos == 0 else (pos_loss + neg_loss) / nums_pos
        return loss

    def forward(self, logits, labels, grad_tensor=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction
                in gaussian distribution.
        """
        if grad_tensor is None:
            grad_tensor = torch.ones_like(logits)
        loss_reg = self.loss_weight * self.gaussian_focal_loss(
            logits, labels, grad_tensor
        )
        return loss_reg


@OBJECT_REGISTRY.register
class LaneFastFocalLoss(GaussianFocalLoss):  # noqa: D205,D400
    """Modified focal loss. Exactly the same as CornerNet,
      Runs faster and costs a little bit more memory,
      For Lane task, return effective loss when num_pos > 2.

    Args:
        alpha: A weighting factor for positive sample.
        gamma: Used in focal loss to balance contribution
                of easy examples and hard examples.
        loss_weight: Weight factor for guassian focal loss.
    """

    def __init__(
        self, alpha: float = 2.0, gamma: float = 4.0, loss_weight: float = 1.0
    ):
        super(LaneFastFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss_weight = loss_weight

    def gaussian_focal_loss(self, pred, gaussian_target, grad_tensor=None):

        pos_inds = gaussian_target.eq(1).float()
        neg_inds = gaussian_target.lt(1).float()
        neg_weights = torch.pow(1 - gaussian_target, self.gamma)

        pos_loss = (
            torch.log(pred)
            * torch.pow(1 - pred, self.alpha)
            * pos_inds
            * grad_tensor
        )
        neg_loss = (
            torch.log(1 - pred)
            * torch.pow(pred, self.alpha)
            * neg_weights
            * neg_inds
            * grad_tensor
        )
        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos > 2:
            return -(pos_loss + neg_loss) / num_pos
        else:
            return torch.tensor(0, dtype=torch.float32).to(pred.device)
