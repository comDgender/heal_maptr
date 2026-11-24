"""Tests for heal.data.transforms.detection."""

import numpy as np
import pytest


def test_normalize_forward():
    pytest.importorskip("cv2")
    from heal.data.transforms.detection import Normalize

    norm = Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    img = np.zeros((4, 4, 3), dtype=np.float32)
    data = {"img": img.copy(), "layout": "hwc"}
    out = norm(data)
    assert out["img"].shape == (4, 4, 3)
