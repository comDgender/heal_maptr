# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import numpy as np
import torch.nn as nn

__all__ = [
    "constant_init",
    "xavier_init",
    "normal_init",
    "kaiming_init",
    "bias_init_with_prob",
]


def _fill_bias(module, bias):
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    weight = getattr(module, "weight", None)
    if weight is not None:
        nn.init.constant_(weight, val)
    _fill_bias(module, bias)


def xavier_init(module, gain=1, bias=0, distribution="normal"):
    assert distribution in ("uniform", "normal")
    weight = getattr(module, "weight", None)
    if weight is not None:
        init_fn = (
            nn.init.xavier_uniform_
            if distribution == "uniform"
            else nn.init.xavier_normal_
        )
        init_fn(weight, gain=gain)
    _fill_bias(module, bias)


def normal_init(module, mean=0, std=1, bias=0):
    weight = getattr(module, "weight", None)
    if weight is not None:
        nn.init.normal_(weight, mean, std)
    _fill_bias(module, bias)


def kaiming_init(
    module,
    a=0,
    mode="fan_out",
    nonlinearity="relu",
    bias=0,
    distribution="normal",
):
    assert distribution in ("uniform", "normal")
    weight = getattr(module, "weight", None)
    if weight is not None:
        if distribution == "uniform":
            nn.init.kaiming_uniform_(
                weight, a=a, mode=mode, nonlinearity=nonlinearity
            )
        else:
            nn.init.kaiming_normal_(
                weight, a=a, mode=mode, nonlinearity=nonlinearity
            )
    _fill_bias(module, bias)


def bias_init_with_prob(prior_prob):
    """Initialize conv/fc bias value according to giving probability."""
    return float(-np.log((1 - prior_prob) / prior_prob))
