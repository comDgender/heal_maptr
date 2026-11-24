# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import math

import torch
import torch.nn as nn


def _zero_bias(bias) -> None:
    """Zero-initialize a bias tensor when it exists."""
    if bias is not None:
        nn.init.zeros_(bias)


def _glorot_bound(fan_in: float, fan_out: float) -> float:
    """Symmetric uniform bound used by Glorot/Xavier initialization."""
    return (6.0 / (fan_in + fan_out)) ** 0.5


def _init_recurrent(m: nn.Module, gates: int) -> None:
    """Shared init routine for LSTM/GRU style recurrent modules."""
    for name, param in m.named_parameters():
        if "weight_ih" in name:
            for chunk in param.chunk(gates, 0):
                nn.init.xavier_uniform_(chunk)
        elif "weight_hh" in name:
            for chunk in param.chunk(gates, 0):
                nn.init.orthogonal_(chunk)
        elif "weight_hr" in name:
            nn.init.xavier_uniform_(param)
        elif "bias_ih" in name:
            nn.init.zeros_(param)
        elif "bias_hh" in name:
            nn.init.zeros_(param)
            # LSTM forget-gate bias is set to one.
            if gates == 4:
                nn.init.ones_(param.chunk(gates, 0)[1])


def weight_init(m: nn.Module) -> None:
    norm_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.LayerNorm,
    )
    conv_types = (nn.Conv1d, nn.Conv2d, nn.Conv3d)

    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        _zero_bias(m.bias)
    elif isinstance(m, conv_types):
        bound = _glorot_bound(
            m.in_channels / m.groups, m.out_channels / m.groups
        )
        nn.init.uniform_(m.weight, -bound, bound)
        _zero_bias(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, norm_types):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            bound = _glorot_bound(m.embed_dim, m.embed_dim)
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            for proj in (m.q_proj_weight, m.k_proj_weight, m.v_proj_weight):
                nn.init.xavier_uniform_(proj)
        _zero_bias(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        _zero_bias(m.out_proj.bias)
        for tensor in (m.bias_k, m.bias_v):
            if tensor is not None:
                nn.init.normal_(tensor, mean=0.0, std=0.02)
    elif isinstance(m, (nn.LSTM, nn.LSTMCell)):
        _init_recurrent(m, gates=4)
    elif isinstance(m, (nn.GRU, nn.GRUCell)):
        _init_recurrent(m, gates=3)


def wrap_angle(
    angle: torch.Tensor, min_val: float = -math.pi, max_val: float = math.pi
) -> torch.Tensor:

    angle = torch.where(angle < min_val, angle + 2 * math.pi, angle)
    angle = torch.where(angle > max_val, angle - 2 * math.pi, angle)
    return angle
