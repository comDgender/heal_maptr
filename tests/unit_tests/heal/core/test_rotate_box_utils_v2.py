"""Tests for heal.core.rotate_box_utils_v2 lazy numba backend."""

import numpy as np


def test_rotate_iou_v2_identical_boxes():
    from heal.core.rotate_box_utils_v2 import rotate_iou_v2

    boxes = np.array([[0.0, 0.0, 2.0, 2.0, 0.0]], dtype=np.float32)
    iou = rotate_iou_v2(boxes, boxes)
    assert iou.shape == (1, 1)
    assert abs(float(iou[0, 0]) - 1.0) < 1e-3
