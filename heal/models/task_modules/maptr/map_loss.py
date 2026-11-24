# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

# Copyright (c) OpenMMLab. All rights reserved.
import functools
from typing import Optional, Union

import torch
import torch.nn.functional as F
from hat_engine.registry import OBJECT_REGISTRY
from torch import nn as nn
from torch.nn.functional import smooth_l1_loss
from torch.quantization import DeQuantStub

__all__ = ["PtsDirCosLoss", "PtsL1Loss", "PtsL1Cost", "OrderedPtsL1Cost"]


def reduce_loss(loss, reduction):
    """Collapse an element-wise loss tensor.

    ``reduction`` may be ``"none"``, ``"mean"``, or ``"sum"``.
    """
    mode = F._Reduction.get_enum(reduction)
    if mode == 0:  # none
        return loss
    if mode == 1:  # mean
        return loss.mean()
    if mode == 2:  # sum
        return loss.sum()
    return loss


def weight_reduce_loss(loss, weight=None, reduction="mean", avg_factor=None):
    """Scale ``loss`` by optional ``weight``, then reduce.

    When ``avg_factor`` is set, ``reduction="mean"`` divides the sum by that
    factor; ``"none"`` leaves the weighted tensor unchanged; ``"sum"`` is
    invalid together with ``avg_factor``.
    """
    if weight is not None:
        loss = loss * weight

    if avg_factor is None:
        return reduce_loss(loss, reduction)

    if reduction == "mean":
        return loss.sum() / avg_factor
    if reduction == "none":
        return loss
    raise ValueError('avg_factor can not be used with reduction="sum"')


def weighted_loss(loss_func):
    @functools.wraps(loss_func)
    def wrapper(
        pred, target, weight=None, reduction="mean", avg_factor=None, **kwargs
    ):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
        return loss

    return wrapper


def custom_weight_dir_reduce_loss(
    loss, weight=None, reduction="mean", avg_factor=None
):
    """Weight and reduce directional map-point losses.

    ``loss`` is shaped ``(num_sample, num_dir)``. Unlike
    :func:`weight_reduce_loss`, ``avg_factor`` is required.
    """
    if weight is not None:
        loss = loss * weight

    if avg_factor is None:
        raise ValueError("avg_factor should not be none for OrderedPtsL1Loss")

    if reduction == "mean":
        return loss.sum() / avg_factor
    if reduction == "none":
        return loss
    raise ValueError('avg_factor can not be used with reduction="sum"')


def custom_weighted_dir_loss(loss_func):
    """Decorate a per-element dir loss with weight / reduction.

    ``loss_func`` must return an unreduced tensor. The wrapped call accepts
    ``weight``, ``reduction``, and ``avg_factor`` and routes them through
    :func:`custom_weight_dir_reduce_loss`.
    """

    @functools.wraps(loss_func)
    def wrapper(
        pred, target, weight=None, reduction="mean", avg_factor=None, **kwargs
    ):
        per_elem = loss_func(pred, target, **kwargs)
        return custom_weight_dir_reduce_loss(
            per_elem, weight, reduction, avg_factor
        )

    return wrapper


@weighted_loss
def _smooth_l1_loss(pred, target, beta=1.0):
    """Smooth-L1 (Huber) on matching ``pred`` / ``target`` tensors.

    Quadratic below ``beta``, linear above. Empty ``target`` yields a zero
    scalar that still depends on ``pred`` for autograd.
    """
    if beta <= 0:
        raise AssertionError(f"beta must be > 0, got {beta}")
    if target.numel() == 0:
        return pred.sum() * 0
    if pred.shape != target.shape:
        raise AssertionError(
            f"pred/target shape mismatch: {tuple(pred.shape)} vs "
            f"{tuple(target.shape)}"
        )
    abs_err = (pred - target).abs()
    quadratic = abs_err.square().mul(0.5 / beta)
    linear = abs_err - 0.5 * beta
    return torch.where(abs_err < beta, quadratic, linear)


@weighted_loss
def pts_l1_loss(pred, target):
    """Element-wise L1 over map points ``[N, P, C]``."""
    if target.numel() == 0:
        return pred.sum() * 0
    if pred.shape != target.shape:
        raise AssertionError(
            f"pred/target shape mismatch: {tuple(pred.shape)} vs "
            f"{tuple(target.shape)}"
        )
    return (pred - target).abs()


@custom_weighted_dir_loss
def pts_dir_cos_loss(pred, target):
    """Dir cosine similiarity loss.

    Args:
        pred (torch.Tensor): shape [num_samples, num_dir, num_coords]
        target (torch.Tensor): shape [num_samples, num_dir, num_coords]

    """
    if target.numel() == 0:
        return pred.sum() * 0
    # import pdb;pdb.set_trace()
    num_samples, num_dir, num_coords = pred.shape
    loss_func = torch.nn.CosineEmbeddingLoss(reduction="none")
    tgt_param = target.new_ones((num_samples, num_dir))
    tgt_param = tgt_param.flatten(0)
    loss = loss_func(pred.flatten(0, 1), target.flatten(0, 1), tgt_param)
    loss = loss.view(num_samples, num_dir)
    return loss


@OBJECT_REGISTRY.register
class PtsDirCosLoss(nn.Module):
    """Cosine loss on polyline edge directions (see ``pts_dir_cos_loss``).

    Args:
        reduction: ``"none"`` / ``"mean"`` / ``"sum"``.
        loss_weight: Scalar multiplier on the reduced loss.
    """

    def __init__(self, reduction="mean", loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        reduction_override=None,
    ):
        """Apply weighted directional cosine loss."""
        allowed = (None, "none", "mean", "sum")
        if reduction_override not in allowed:
            raise AssertionError(
                f"reduction_override must be one of {allowed}, "
                f"got {reduction_override!r}"
            )
        reduce_mode = reduction_override or self.reduction
        return self.loss_weight * pts_dir_cos_loss(
            pred,
            target,
            weight,
            reduction=reduce_mode,
            avg_factor=avg_factor,
        )


@OBJECT_REGISTRY.register
class PtsL1Loss(nn.Module):
    """Map-point L1 loss; switches to Smooth-L1 when ``beta > 0``.

    Args:
        reduction: ``"none"`` / ``"mean"`` / ``"sum"``.
        loss_weight: Scalar multiplier on the reduced loss.
        beta: Smooth-L1 threshold; ``0`` keeps plain L1.
    """

    def __init__(self, reduction="mean", loss_weight=1.0, beta=0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.beta = beta

    def forward(
        self,
        pred,
        target,
        weight=None,
        avg_factor=None,
        reduction_override=None,
    ):
        """Apply weighted point L1 / Smooth-L1 loss."""
        allowed = (None, "none", "mean", "sum")
        if reduction_override not in allowed:
            raise AssertionError(
                f"reduction_override must be one of {allowed}, "
                f"got {reduction_override!r}"
            )
        reduce_mode = reduction_override or self.reduction
        if self.beta > 0:
            point_loss = _smooth_l1_loss(
                pred,
                target,
                weight,
                reduction=reduce_mode,
                avg_factor=avg_factor,
                beta=self.beta,
            )
        else:
            point_loss = pts_l1_loss(
                pred,
                target,
                weight,
                reduction=reduce_mode,
                avg_factor=avg_factor,
            )
        return self.loss_weight * point_loss


@OBJECT_REGISTRY.register
class PtsL1Cost(object):
    """OrderedPtsL1Cost.

    Args:
        weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        """Call function.

        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (x, y), which are all in range [0, 1]. Shape
                [num_query, num_pts, 2].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x,y).
                Shape [num_gt, num_ordered, num_pts, 2].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        num_gts, num_pts, num_coords = gt_bboxes.shape
        # import pdb;pdb.set_trace()
        bbox_pred = bbox_pred.view(bbox_pred.size(0), -1)
        gt_bboxes = gt_bboxes.view(num_gts, -1)
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight


@OBJECT_REGISTRY.register
class OrderedPtsL1Cost(object):
    """OrderedPtsL1Cost.

    Args:
        weight (int | float, optional): loss_weight
        beta (float): The smooth threshold. Defaults is 0.
    """

    def __init__(self, weight=1.0, beta=0):
        self.weight = weight
        self.beta = beta

    def __call__(self, bbox_pred, gt_bboxes):
        """Call function.

        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (x, y), which are all in range [0, 1]. Shape
                [num_query, num_pts, 2].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x,y).
                Shape [num_gt, num_ordered, num_pts, 2].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        num_gts, num_orders, num_pts, num_coords = gt_bboxes.shape
        # import pdb;pdb.set_trace()
        bbox_pred = bbox_pred.view(bbox_pred.size(0), -1)
        gt_bboxes = gt_bboxes.flatten(2).view(num_gts * num_orders, -1)
        if self.beta > 0:
            num_pred = len(bbox_pred)
            bbox_pred = bbox_pred.unsqueeze(1).repeat(1, len(gt_bboxes), 1)
            gt_bboxes = gt_bboxes.unsqueeze(0).repeat(num_pred, 1, 1)
            bbox_cost = smooth_l1_loss(
                bbox_pred, gt_bboxes, reduction="none", beta=self.beta
            ).sum(-1)
        else:
            bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight


@OBJECT_REGISTRY.register
class SimpleLoss(torch.nn.Module):
    def __init__(
        self,
        pos_weight,
        loss_weight,
        reduction: str = "mean",
    ):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.Tensor([pos_weight]), reduction="none"
        )
        self.loss_weight = loss_weight
        self.reduction = reduction

        self.dequant = DeQuantStub()

    def forward(
        self,
        ypred,
        ytgt,
        weight: Optional[torch.Tensor] = None,
        avg_factor: Optional[Union[float, torch.Tensor]] = None,
    ):
        if isinstance(ypred, (list, tuple)):
            assert isinstance(
                ytgt, (list, tuple)
            ), "ypred and ytgt should have the same type"
            total_loss = 0
            for pred, gt in zip(ypred, ytgt):
                loss = self.loss_fn(pred, gt)
                loss = self.dequant(loss)
                loss = weight_reduce_loss(
                    loss.float(), weight, self.reduction, avg_factor
                )
                total_loss += loss * self.loss_weight
            return total_loss

        loss = self.loss_fn(ypred, ytgt)
        loss = self.dequant(loss)
        loss = weight_reduce_loss(
            loss.float(), weight, self.reduction, avg_factor
        )
        return loss * self.loss_weight
