# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import torch
from horizon_plugin_pytorch.nn.quantized import FloatFunctional as FF
from torch import nn


class FFN(nn.Module):
    """The basic structure of FFN.

    Args:
        dim: The inputs dim.
        scale: The scale for inputs dim in hidden layers.
        bias: Whether use bias,
        dropout: Probability of an element to be zeroed.
    """

    def __init__(
        self, dim: int, scale: int = 2, bias: bool = True, dropout: float = 0.0
    ):
        super().__init__()
        self.ffn1 = nn.Linear(dim, int(dim * scale), bias=bias)
        self.ffn1_act = nn.ReLU(inplace=True)
        self.ffn_dropout1 = nn.Dropout(dropout)
        self.ffn2 = nn.Linear(int(dim * scale), dim, bias=bias)
        self.ffn_dropout2 = nn.Dropout(dropout)
        self.add = FF()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Foward FFN."""
        x = self.ffn1(x)
        x = self.ffn1_act(x)
        x = self.ffn_dropout1(x)
        x = self.ffn2(x)
        x = self.ffn_dropout2(x)
        x = self.add.add(x, skip)
        return x
