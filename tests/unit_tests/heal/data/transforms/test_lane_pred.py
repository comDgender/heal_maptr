"""Tests for heal.data.transforms.lane_pred."""

import pytest


def test_lane_pred_importable():
    from heal.data.transforms.lane_pred import (
        HueSaturationValue,
        JPEGCompress,
        MeanBlur,
        MedianBlur,
        RandomBrightnessContrast,
        LaneRandomResizedCrop,
        RGBShift,
        ShiftScaleRotate,
    )

    assert HueSaturationValue is not None
    assert RGBShift is not None
    assert JPEGCompress is not None
    assert MeanBlur is not None
    assert MedianBlur is not None
    assert RandomBrightnessContrast is not None
    assert ShiftScaleRotate is not None
    assert LaneRandomResizedCrop is not None


def test_mean_blur_call():
    pytest.importorskip("cv2")
    import numpy as np

    from heal.data.transforms.lane_pred import MeanBlur

    transform = MeanBlur(ksize=3, p=1.0)
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[4, 4] = 255
    data = {"img": img}
    out = transform(data)
    assert out["img"].shape == (8, 8, 3)


def test_median_blur_rejects_even_ksize():
    pytest.importorskip("cv2")
    import pytest as _pytest

    from heal.data.transforms.lane_pred import MedianBlur

    with _pytest.raises(ValueError):
        MedianBlur(ksize=4)
