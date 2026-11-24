"""Tests for heal.data.transforms.multi_views."""

import numpy as np
from PIL import Image

from heal.data.transforms.multi_views import MultiViewsImgResize


def test_multi_views_img_resize_updates_images():
    resize = MultiViewsImgResize(size=(32, 64))
    data = {
        "img": [
            Image.new("RGB", (128, 64)),
            Image.new("RGB", (128, 64)),
        ],
        "lidar2img": [np.eye(4), np.eye(4)],
    }
    out = resize(data)
    assert len(out["img"]) == 2
    assert out["img"][0].size == (64, 32)
