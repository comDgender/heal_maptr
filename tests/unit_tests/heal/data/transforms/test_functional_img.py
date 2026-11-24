"""Tests for heal.data.transforms.functional_img."""

import numpy as np
import pytest


def test_imresize_hwc_nearest():
    pytest.importorskip("cv2")
    from heal.data.transforms.functional_img import imresize

    img = np.zeros((10, 20, 3), dtype=np.uint8)
    out = imresize(img, 5, 10, layout="hwc", interpolation="nearest")
    assert out.shape == (10, 5, 3)
