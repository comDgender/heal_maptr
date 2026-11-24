# -*- coding: utf-8 -*-
# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
# Source code reference to mmdetection, gluonnas

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from hat_engine.models.losses.utils import weight_reduce_loss
from hat_engine.utils.apply_func import _as_list
from torch.amp import autocast

__all__ = [
    "CrossEntropyLoss",
    "CEWithLabelSmooth",
    "SoftTargetCrossEntropy",
]


def binary_cross_entropy(
    pred, target, weight, reduction, avg_factor, class_weight=None, **kwargs
):  # noqa: D205,D400
    """Measure binary cross entropy between target and pred
    logits, then apply element-wise weight and reduce loss.

    Args:
        pred (Tensor): Logits Tensor of arbitrary shape.
        target (Tensor): Tensor of the same shape as pred.
        weight (Tensor): Element-wise weight loss weight.
        reduction (str): The method used to reduce the loss. Options are
            [`none`, `mean`, `sum`].
        avg_factor (float): Average factor that is used to average the loss.
        class_weight (Tensor, optional): Weight of each class, must be a vector
            with length equal to the number of classes.
    """
    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy_with_logits(
        pred, target.float(), pos_weight=class_weight, reduction="none"
    )
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss.float(), weight, reduction=reduction, avg_factor=avg_factor
    )

    return loss


def cross_entropy(
    pred,
    target,
    weight=None,
    reduction="mean",
    avg_factor=None,
    class_weight=None,
    ignore_index=-100,
):  # noqa: D205,D400
    """Calculate the cross entropy loss using F.cross_entropy,
    then apply element-wise weight and reduce loss.

    Args:
        pred (torch.Tensor): Same as F.cross_entropy, Logits Tensor.
        target (torch.Tensor): Same as F.cross_entropy.
        weight (torch.Tensor): Element-wise weight loss weight.
        reduction (str): The method used to reduce the loss. Options are
            [`none`, `mean`, `sum`].
        avg_factor (float): Average factor that is used to average the loss.
        class_weight : a manual rescaling weight given to each class. If given,
            has to be a Tensor of size `C`.
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient.
    """
    # element-wise losses
    loss = F.cross_entropy(
        pred,
        target.long(),
        weight=class_weight,
        reduction="none",
        ignore_index=ignore_index,
    )
    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
        loss = loss * weight
        if weight.sum() == 0:
            return loss.sum()
        loss = loss.sum() / weight.sum()
    else:
        loss = weight_reduce_loss(loss.float(), weight, reduction, avg_factor)

    return loss


class CrossEntropyLoss(nn.Module):
    """Calculate cross entropy loss of multi stride output.

    Args:
        use_sigmoid (bool): Whether logits tensor is converted to probability
            through sigmoid, Defaults to False.
            If `True`, use `F.binary_cross_entropy_with_logits`.
            If `False`, use `F.cross_entropy`.
        reduction (str): The method used to reduce the loss. Options are
            [`none`, `mean`, `sum`].
        class_weight (list[float]): Weight of each class. Defaults is None.
        loss_weight (float): Global weight of loss. Defaults is 1.
        ignore_index (int): Only works when using cross_entropy.
        loss_name (str): The key of loss in return dict. If None, return loss
            directly.

    Returns:
        cross entropy loss

    """

    def __init__(
        self,
        use_sigmoid: bool = False,
        reduction: str = "mean",
        class_weight: Optional[List[float]] = None,
        loss_weight: float = 1.0,
        ignore_index: int = -1,
        loss_name: Optional[str] = None,
        auto_class_weight: Optional[bool] = False,
        weight_min: Optional[float] = None,
        weight_noobj: Optional[float] = None,
        num_class: int = 0,
    ):
        super(CrossEntropyLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.ignore_index = ignore_index
        self.loss_name = loss_name
        self.auto_class_weight = auto_class_weight
        self.num_class = num_class

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        else:
            self.cls_criterion = cross_entropy
        if self.auto_class_weight:
            self.weight_min = weight_min
            self.weight_noobj = weight_noobj

    @autocast("cuda", enabled=False)
    def forward(self, pred, target, weight=None, avg_factor=None):
        loss_cls = None
        object_num = torch.reshape(target, (-1,)).shape[0]
        if pred is not None:
            preds = _as_list(pred)
            if self.auto_class_weight:
                class_weight = []
                for i in range(self.num_class):
                    num = len(
                        torch.where(torch.reshape(target, (-1,)) == i)[0]
                    )
                    if num == 0:
                        class_weight.append(self.weight_noobj)
                    else:
                        class_weight.append(
                            round(
                                np.max(
                                    [1 - num / object_num, self.weight_min]
                                ),
                                2,
                            )
                        )
                class_weight = preds[0].new_tensor(class_weight)
                class_weight = class_weight
            else:
                if self.class_weight is not None:
                    class_weight = preds[0].new_tensor(self.class_weight)
                    class_weight = class_weight
                else:
                    class_weight = None
            loss_cls = 0
            for each_pred in preds:
                loss_cls += self.loss_weight * self.cls_criterion(
                    each_pred,
                    target,
                    weight,
                    class_weight=class_weight,
                    reduction=self.reduction,
                    avg_factor=avg_factor,
                    ignore_index=self.ignore_index,
                )
        if self.loss_name is None:
            return loss_cls
        else:
            return {self.loss_name: loss_cls}


class CEWithLabelSmooth(torch.nn.CrossEntropyLoss):
    """
    The losses of cross-entropy with label smooth.

    Args:
        smooth_alpha (float): Alpha of label smooth.
        ignore_index (int): Specifies a target value that is ignored
            and does not contribute to the loss.
        loss_weight (float): Global weight of loss. Defaults is 1.0.
    """

    def __init__(
        self, smooth_alpha=0.1, ignore_index: int = -100, loss_weight=1.0
    ):
        super(CEWithLabelSmooth, self).__init__()
        self.smooth_alpha = smooth_alpha
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight

    def forward(self, input, target):
        n = input.size()[-1]
        log_pred = F.log_softmax(input, dim=-1)
        loss = -log_pred.sum(dim=-1).mean()
        nll = F.nll_loss(
            log_pred, target, ignore_index=self.ignore_index, reduction="mean"
        )
        sa = self.smooth_alpha
        return self.loss_weight * (sa * (loss / n) + (1 - sa) * nll)


class SoftTargetCrossEntropy(torch.nn.Module):
    """
    The losses of cross-entropy with soft target.

    Args:
        loss_name (str): The name of returned losses.
    """

    def __init__(self, loss_name=None):
        super(SoftTargetCrossEntropy, self).__init__()
        self.loss_name = loss_name

    def forward(self, input, target):
        if target.ndim == 1:
            loss_cls = F.cross_entropy(input, target)
        else:
            loss = torch.sum(-target * F.log_softmax(input, dim=-1), dim=-1)
            loss_cls = loss.mean()
        if self.loss_name is None:
            return loss_cls
        else:
            return {self.loss_name: loss_cls}
