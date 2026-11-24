# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import Any

import numpy as np
from hat_engine.registry import OBJECT_REGISTRY
from PIL import Image


def _zero_grid_stripes(
    mask: np.ndarray,
    size: int,
    stripe_size: int,
    stripe_len: int,
    start: int,
    axis: int,
) -> None:
    for i in range(size // stripe_size):
        begin = stripe_size * i + start
        end = min(begin + stripe_len, size)
        if axis == 0:
            mask[begin:end, :] *= 0
        else:
            mask[:, begin:end] *= 0


@OBJECT_REGISTRY.register
class GridMask(object):
    """Generate GridMask for grid masking augmentation.

    Args:
        use_h: Whether to generate grid for height dim.
        use_w: Whether to generate grid for width dim.
        rotate: Rotation of grid mesh.
        offset: Whether to randomly add offset.
        ratio: Black grid mask ratio.
        limit_d_ratio_min: Min black add white mask ratio.
        limit_d_ratio_max: Max black add white mask ratio.
        mode: 0 or 1, if use ~mask.
        prob: Probability of occurrence.
    """

    def __init__(
        self,
        use_h: bool,
        use_w: bool,
        rotate: float = 1.0,
        offset: bool = False,
        ratio: float = 0.5,
        limit_d_ratio_min: float = 0.0,
        limit_d_ratio_max: float = 1.0,
        mode: int = 0,
        prob: float = 1.0,
    ):
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.limit_d_ratio_min = limit_d_ratio_min
        self.limit_d_ratio_max = limit_d_ratio_max
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch: int, max_epoch: int) -> Any:
        self.prob = self.st_prob * epoch / max_epoch

    def __call__(self, data):
        if np.random.rand() > self.prob:
            return data

        if isinstance(data, dict):
            x = data["img"]
        else:
            x = data

        h, w, c = x.shape
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(
            max(2, h * self.limit_d_ratio_min), h * self.limit_d_ratio_max
        )
        self.l_ = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        if self.use_h:
            _zero_grid_stripes(
                mask, hh, d, self.l_, np.random.randint(d), axis=0
            )
        if self.use_w:
            _zero_grid_stripes(
                mask, ww, d, self.l_, np.random.randint(d), axis=1
            )

        mask = np.asarray(
            Image.fromarray(np.uint8(mask)).rotate(
                np.random.randint(self.rotate)
            )
        )
        top, left = (hh - h) // 2, (ww - w) // 2
        mask = mask[top : top + h, left : left + w]

        if self.mode == 1:
            mask = 1 - mask

        mask = np.expand_dims(mask.astype(np.float32), axis=2)
        mask = np.tile(mask, [1, 1, c])

        if self.offset:
            offset = np.float32(2 * (np.random.rand(h, w) - 0.5))
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask

        if isinstance(data, dict):
            data["img"] = x
            return data
        else:
            return x
