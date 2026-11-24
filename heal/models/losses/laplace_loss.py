# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import torch
import torch.nn as nn
from hat_engine.registry import OBJECT_REGISTRY

__all__ = ["LaplaceNLLLoss"]


@OBJECT_REGISTRY.register
class LaplaceNLLLoss(nn.Module):
    """
    Negative log-likelihood loss function based on the Laplace distribution.

    The ``scale.clamp_`` below operates on a cloned tensor under
    ``torch.no_grad`` and is therefore safe on the float / autograd path.
    This loss is only used in float training (``QCNetOE.forward`` skips it
    once ``deploy=True`` and during qat-build), so QTensor-aware clamp is
    not required.

    Args:
        eps: A small value to avoid division by zero. Default is 1e-6.
        reduction: Specifies the reduction to apply to the
            output: 'mean', 'sum', or 'none'. Default is 'mean'.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean") -> None:
        super(LaplaceNLLLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        loc, scale = pred.chunk(2, dim=-1)
        scale = scale.clone()
        with torch.no_grad():
            scale.clamp_(min=self.eps)
        nll = torch.log(2 * scale) + torch.abs(target - loc) / scale
        if self.reduction == "mean":
            return nll.mean()
        elif self.reduction == "sum":
            return nll.sum()
        elif self.reduction == "none":
            return nll
        else:
            raise ValueError(
                "{} is not a valid value for reduction".format(self.reduction)
            )
