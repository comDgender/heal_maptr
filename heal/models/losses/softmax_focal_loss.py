# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import Sequence, Union

import torch
from hat_engine.models.losses.utils import weight_reduce_loss
from hat_engine.registry import OBJECT_REGISTRY

__all__ = ["SoftmaxFocalLoss"]


@OBJECT_REGISTRY.register
@OBJECT_REGISTRY.alias("ClassificationFocalLoss")
class SoftmaxFocalLoss(torch.nn.Module):
    """Softmax focal loss for multi-scale segmentation outputs.

    Args:
        loss_name: Key used in the returned loss dictionary.
        num_classes: Number of segmentation classes.
        alpha: Focal loss class-balancing factor.
        gamma: Focal loss focusing factor.
        reduction: Reduction method passed to loss reduction.
        weight: Loss weight or per-output weights for multi-scale logits.
    """

    def __init__(
        self,
        loss_name: str,
        num_classes: int,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
        weight: Union[float, Sequence] = 1.0,
    ):
        super().__init__()
        self.loss_name = loss_name
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight

    def focal_loss(self, logits, label):
        """Compute focal loss for one logits/label pair.

        Args:
            logits: Predicted class logits.
            label: One-hot target tensor with the same spatial size.

        Returns:
            torch.Tensor: Reduced focal loss tensor.
        """
        probs = torch.softmax(logits, dim=1)
        probs = probs + 1e-9
        ce = (-torch.log(probs)) * label
        weight = torch.pow(1.0 - probs, self.gamma)
        loss = ce * weight * self.alpha
        loss = loss.sum(dim=1)
        return weight_reduce_loss(loss, reduction=self.reduction)

    def forward(self, logits, labels):
        """Compute focal loss and return it in a named dictionary.

        Args:
            logits: Single logits tensor or multi-scale logits sequence.
            labels: Single label tensor or label sequence aligned with logits.

        Returns:
            dict[str, torch.Tensor]: Loss dictionary keyed by ``loss_name``.
        """
        if isinstance(logits, Sequence):
            if not isinstance(self.weight, Sequence):
                weights = [self.weight] * len(logits)
            else:
                weights = self.weight
            return {
                self.loss_name: sum(
                    [
                        self.focal_loss(logit, label) * w
                        for logit, label, w in zip(logits, labels, weights)
                    ]
                )
            }
        return {self.loss_name: self.focal_loss(logits, labels) * self.weight}
