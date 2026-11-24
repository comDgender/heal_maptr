# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from torch import Tensor

from heal.data.transforms.affine import AffineMat2DGenerator

__all__ = [
    "imresize",
    "random_flip",
    "image_pad",
    "image_normalize",
    "imresize_pad_to_keep_ratio",
]

cv2_interp_codes = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}


def imresize(
    img: np.ndarray,
    w,
    h,
    layout,
    divisor=1,
    keep_ratio=False,
    return_scale=False,
    interpolation="bilinear",
    raw_scaler_enable=False,
    sample1c_enable=True,
    raw_pattern=None,
    split_transform: bool = False,
    split_trans_h=256,
    split_trans_w=256,
) -> Union[np.ndarray, Tuple[np.ndarray, type(None), float, float]]:
    """Resize image with OpenCV. Simplified: no raw_scaler/sample1c/split."""
    assert layout in ["hwc", "chw", "hw"]
    if layout == "hwc":
        img_h, img_w, img_c = img.shape
    elif layout == "chw":
        img_c, img_h, img_w = img.shape
        img = np.ascontiguousarray(img.transpose((1, 2, 0)))
    else:
        img_h, img_w = img.shape

    if keep_ratio:
        short = min(w, h)
        max_size = max(w, h)
        im_size_min, im_size_max = (
            (img_h, img_w) if img_w > img_h else (img_w, img_h)
        )
        scale = float(short) / float(im_size_min)
        if np.floor(scale * im_size_max / divisor) * divisor > max_size:
            scale = float(np.floor(max_size / divisor) * divisor) / im_size_max
        new_w = int(np.floor(img_w * scale / divisor) * divisor)
        new_h = int(np.floor(img_h * scale / divisor) * divisor)
    else:
        new_w = int(np.floor(w / divisor) * divisor)
        new_h = int(np.floor(h / divisor) * divisor)

    resized_img = cv2.resize(
        img, (new_w, new_h), interpolation=cv2_interp_codes[interpolation]
    )
    if layout == "chw":
        resized_img = np.ascontiguousarray(resized_img.transpose((2, 0, 1)))

    w_scale = float(new_w / img_w)
    h_scale = float(new_h / img_h)
    if return_scale:
        return resized_img, None, w_scale, h_scale
    return resized_img


def imresize_pad_to_keep_ratio(
    img: np.ndarray,
    target_hw,
    layout,
    keep_ratio=True,
):
    """Resize image with padding to keep ratio. Returns (img, 3x3 R matrix)."""
    assert layout in ["hwc", "chw", "hw"]
    if layout == "chw":
        img = np.ascontiguousarray(img.transpose((1, 2, 0)))

    R = AffineMat2DGenerator.resize(img.shape[:2], target_hw, keep_ratio)
    img = cv2.warpPerspective(img, R, dsize=target_hw[::-1])
    return img, R


def random_flip(
    img: Union[np.ndarray, torch.Tensor], layout, px=0, py=0, raw_pattern=None
) -> Tuple[Union[np.ndarray, torch.Tensor], Tuple[bool, bool], type(None)]:
    """Randomly flip image along horizontal and vertical."""
    assert layout in ["hwc", "chw", "hw"]
    assert isinstance(img, (torch.Tensor, np.ndarray))
    h_index = layout.index("h")
    w_index = layout.index("w")
    flip_x = np.random.choice([False, True], p=[1 - px, px])
    flip_y = np.random.choice([False, True], p=[1 - py, py])

    if isinstance(img, np.ndarray):
        if flip_x:
            img = np.flip(img, axis=w_index)
        if flip_y:
            img = np.flip(img, axis=h_index)
    else:
        if flip_x:
            img = img.flip(w_index)
        if flip_y:
            img = img.flip(h_index)

    return img, (flip_x, flip_y), None


def image_pad(
    img: Union[np.ndarray, torch.Tensor],
    layout,
    shape=None,
    divisor=1,
    pad_val=0,
) -> Union[np.ndarray, torch.Tensor]:
    """Pad image to a certain shape."""
    assert layout in ["hwc", "chw", "hw"]
    if isinstance(pad_val, Sequence):
        assert layout in ["hwc", "chw"]
        c_index = layout.index("c")
        assert len(pad_val) == img.shape[c_index]
        pad_val = torch.tensor(pad_val)
        if layout == "hwc":
            pad_val = pad_val.unsqueeze(0).unsqueeze(0)
        elif layout == "chw":
            pad_val = pad_val.unsqueeze(-1).unsqueeze(-1)
        if isinstance(img, np.ndarray):
            pad_val = pad_val.numpy()

    if shape is None:
        shape = img.shape
        if divisor == 1:
            return img
    assert len(shape) in [2, 3]
    if layout == "chw":
        if len(shape) == 2:
            h = max(img.shape[1], shape[0])
            w = max(img.shape[2], shape[1])
        else:
            h = max(img.shape[1], shape[1])
            w = max(img.shape[2], shape[2])
    else:
        h = max(img.shape[0], shape[0])
        w = max(img.shape[1], shape[1])
    pad_h = int(np.ceil(h / divisor)) * divisor
    pad_w = int(np.ceil(w / divisor)) * divisor

    if len(shape) == 3:
        if layout == "hwc":
            shape = (pad_h, pad_w, shape[-1])
        elif layout == "chw":
            shape = (shape[0], pad_h, pad_w)
    else:
        shape = (pad_h, pad_w)

    if len(shape) < len(img.shape):
        if layout == "hwc":
            shape = tuple(shape) + (img.shape[-1],)
        elif layout == "chw":
            shape = (img.shape[0],) + tuple(shape)
    assert len(shape) == len(img.shape)
    for i in range(len(shape)):
        assert shape[i] >= img.shape[i], "padded shape must >= src shape"

    if isinstance(img, Tensor):
        pad = torch.zeros(shape, dtype=img.dtype, device=img.device)
    elif isinstance(img, np.ndarray):
        pad = np.zeros(shape, dtype=img.dtype)
    else:
        raise TypeError
    pad[...] = pad_val

    if len(img.shape) == 2:
        pad[: img.shape[0], : img.shape[1]] = img
    elif layout == "hwc":
        pad[: img.shape[0], : img.shape[1], ...] = img
    else:
        pad[:, : img.shape[1], : img.shape[2]] = img
    return pad


def image_normalize(img: Union[np.ndarray, Tensor], mean, std, layout):
    """Normalize the image with mean and std."""
    assert layout in ["hwc", "chw"]
    c_index = layout.index("c")

    return_ndarray = False
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img.astype(np.float32))
        return_ndarray = True
    elif isinstance(img, Tensor):
        img = img.float()
    else:
        raise TypeError

    if isinstance(mean, Sequence):
        assert len(mean) == img.shape[c_index]
    else:
        mean = [mean] * img.shape[c_index]

    if isinstance(std, Sequence):
        assert len(std) == img.shape[c_index]
    else:
        std = [std] * img.shape[c_index]

    dtype = img.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=img.device)
    std = torch.as_tensor(std, dtype=dtype, device=img.device)
    if (std == 0).any():
        raise ValueError("std evaluated to zero.")
    if c_index == 0:
        mean = mean[:, None, None]
        std = std[:, None, None]
    else:
        mean = mean[None, None, :]
        std = std[None, None, :]
    img = img.sub(mean).div(std)

    if return_ndarray:
        img = img.numpy()
    return img
