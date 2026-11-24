# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

"""Rotated 2D IoU API; numba is loaded only when these functions are called."""

from typing import Optional

import numpy as np

__all__ = ["rotate_iou_v2"]


def rotate_iou_v2(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    criterion: Optional[int] = -1,
    device_id: Optional[int] = 0,
) -> np.ndarray:
    """Compute rotated ious between GT and pred bboxes (lazy-imports numba backend)."""  # noqa: E501
    from heal.core._rotate_box_utils_v2_numba import (
        rotate_iou_v2 as _rotate_iou_v2,
    )

    return _rotate_iou_v2(
        pred_boxes,
        gt_boxes,
        criterion=criterion,
        device_id=device_id,
    )
