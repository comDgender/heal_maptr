# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

"""Utilities for BEVFormer view transformer (coords, module cloning)."""

import copy
from typing import List, Tuple

import torch
from torch import nn


def get_clone_module(module: nn.Module, N: int) -> nn.ModuleList:
    """Deep-copy a module template ``N`` times (encoder/decoder layer stacks)."""  # noqa: E501
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_min_max_coords(
    real_range: List[float],
    grid_resolution: List[float],
) -> Tuple[float]:
    """Pillar center bounds excluding half-cell border (for grid_sample coords)."""  # noqa: E501
    min_x = real_range[0] + grid_resolution[0] / 2
    min_y = real_range[1] + grid_resolution[1] / 2
    max_x = real_range[2] - grid_resolution[0] / 2
    max_y = real_range[3] - grid_resolution[1] / 2
    return min_x, max_x, min_y, max_y


def gen_coords(bev_size: Tuple[int], pc_range: Tuple[float]) -> torch.Tensor:
    """Generate BEV pillar center coordinates (1, H, W, 2) in metric space."""
    real_w = pc_range[3] - pc_range[0]
    real_h = pc_range[4] - pc_range[1]

    W = bev_size[0]
    H = bev_size[1]

    grid_resolution = (real_w / W, real_h / H)
    real_range = (pc_range[0], pc_range[1], pc_range[3], pc_range[4])

    bev_min_x, bev_max_x, bev_min_y, bev_max_y = get_min_max_coords(
        real_range,
        grid_resolution,
    )

    x = (
        torch.linspace(bev_min_x, bev_max_x, W).reshape((1, W)).repeat(H, 1)
    ).double()
    y = (
        torch.linspace(bev_min_y, bev_max_y, H).reshape((H, 1)).repeat(1, W)
    ).double()
    coords = torch.stack([x, y], dim=-1).unsqueeze(0)
    return coords
