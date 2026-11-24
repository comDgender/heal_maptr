# Copyright (c) Beijing Horizon Information Technology Co., Ltd., all rights reserved.  # noqa: E501

from typing import List

import numpy as np
import torch


def _positive_quadratic_root(
    quad_a: float, quad_b: float, quad_c: float
) -> float:
    radicand = quad_b * quad_b - 4.0 * quad_a * quad_c
    return (quad_b + np.sqrt(radicand)) * 0.5


def gaussian_radius(det_size: List[int], min_overlap: float = 0.5) -> float:
    box_h, box_w = det_size
    edge_sum = box_h + box_w
    box_area = box_w * box_h
    iou_floor = min_overlap

    radius_candidates = [
        _positive_quadratic_root(
            1.0,
            edge_sum,
            box_area * (1.0 - iou_floor) / (1.0 + iou_floor),
        ),
        _positive_quadratic_root(
            4.0,
            2.0 * edge_sum,
            (1.0 - iou_floor) * box_area,
        ),
        _positive_quadratic_root(
            4.0 * iou_floor,
            -2.0 * iou_floor * edge_sum,
            (iou_floor - 1.0) * box_area,
        ),
    ]
    return float(min(radius_candidates))


def gaussian2D(shape: List[int], sigma: float = 1) -> np.ndarray:
    half_h = (shape[0] - 1.0) * 0.5
    half_w = (shape[1] - 1.0) * 0.5
    y_grid, x_grid = np.ogrid[-half_h : half_h + 1, -half_w : half_w + 1]
    inv_var = 1.0 / (2.0 * sigma * sigma)
    kernel = np.exp(-(x_grid * x_grid + y_grid * y_grid) * inv_var)
    tiny = np.finfo(kernel.dtype).eps * kernel.max()
    kernel[kernel < tiny] = 0
    return kernel


def _gaussian_paste_bounds(
    cx: int, cy: int, radius: int, map_h: int, map_w: int
) -> tuple:
    pad_left = min(cx, radius)
    pad_right = min(map_w - cx, radius + 1)
    pad_top = min(cy, radius)
    pad_bottom = min(map_h - cy, radius + 1)
    return pad_left, pad_right, pad_top, pad_bottom


def _kernel_roi_slices(
    radius: int, pad_left: int, pad_right: int, pad_top: int, pad_bottom: int
) -> tuple:
    kern_row_lo = radius - pad_top
    kern_row_hi = radius + pad_bottom
    kern_col_lo = radius - pad_left
    kern_col_hi = radius + pad_right
    return (slice(kern_row_lo, kern_row_hi), slice(kern_col_lo, kern_col_hi))


def draw_umich_gaussian(
    heatmap: np.ndarray,
    center: np.ndarray,
    radius: float,
    k: float = 1,
    return_gaussian: bool = False,
) -> np.ndarray:
    kern_size = int(2 * radius + 1)
    kernel = gaussian2D((kern_size, kern_size), sigma=kern_size / 6.0)

    cx, cy = int(center[0]), int(center[1])
    map_h, map_w = heatmap.shape[:2]
    pad_l, pad_r, pad_t, pad_b = _gaussian_paste_bounds(
        cx, cy, int(radius), map_h, map_w
    )
    row_sl = slice(cy - pad_t, cy + pad_b)
    col_sl = slice(cx - pad_l, cx + pad_r)
    kern_sl = _kernel_roi_slices(int(radius), pad_l, pad_r, pad_t, pad_b)

    target_patch = heatmap[row_sl, col_sl]
    source_patch = kernel[kern_sl]
    if min(source_patch.shape) > 0 and min(target_patch.shape) > 0:
        np.maximum(target_patch, source_patch * k, out=target_patch)

    if return_gaussian:
        coverage = np.zeros_like(kernel)
        coverage[kern_sl] = 1
        return heatmap, source_patch, coverage
    return heatmap


def draw_umich_gaussian_torch(
    heatmap: torch.Tensor, center: torch.Tensor, radius: float, k: float = 1
) -> torch.Tensor:
    kern_size = int(2 * radius + 1)
    kernel = gaussian2D((kern_size, kern_size), sigma=kern_size / 6.0)

    cx, cy = int(center[0]), int(center[1])
    map_h, map_w = heatmap.shape[:2]
    pad_l, pad_r, pad_t, pad_b = _gaussian_paste_bounds(
        cx, cy, int(radius), map_h, map_w
    )
    row_sl = slice(cy - pad_t, cy + pad_b)
    col_sl = slice(cx - pad_l, cx + pad_r)
    kern_sl = _kernel_roi_slices(int(radius), pad_l, pad_r, pad_t, pad_b)

    target_patch = heatmap[row_sl, col_sl]
    source_patch = torch.from_numpy(kernel[kern_sl]).to(
        device=heatmap.device, dtype=torch.float32
    )
    if min(source_patch.shape) > 0 and min(target_patch.shape) > 0:
        torch.max(target_patch, source_patch * k, out=target_patch)
    return heatmap
