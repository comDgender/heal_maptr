# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
# Minimal base for post processors.

from abc import abstractmethod
from typing import Any, Dict, Optional, Sequence, Union

import torch
import torch.nn as nn

__all__ = ["PostProcessorBase"]


class PostProcessorBase(nn.Module):
    """Interface class of post processor."""

    @abstractmethod
    def forward(
        self,
        pred: Union[
            torch.Tensor, Sequence[torch.Tensor], Dict[str, torch.Tensor]
        ],
        meta_data: Optional[Dict[str, Any]] = None,
    ):
        """Do post process for model predictions."""
        raise NotImplementedError
