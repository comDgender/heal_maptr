# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import Tuple

import numpy as np

__all__ = ["AffineMat2DGenerator"]

numeric_types = (float, int, np.generic)


class AffineMat2DGenerator(object):
    """Generate affine transformation matrix (3x3)."""

    @staticmethod
    def resize(
        src_shape: Tuple, dst_shape: Tuple, pad_to_keep_ratio: bool = False
    ):
        """Return the resize matrix of affine.

        Args:
            src_shape: Source shape in hw/hwc format.
            dst_shape: Destinate shape in hw/hwc format.
            pad_to_keep_ratio: Whether to do padding to keep ratio.
        """
        r_h, r_w = src_shape[:2]
        d_h, d_w = dst_shape[:2]
        Rs = np.eye(3)
        if pad_to_keep_ratio:
            C = np.eye(3)
            C[0, 2] = -r_w / 2
            C[1, 2] = -r_h / 2

            if r_w / r_h < d_w / d_h:
                ratio = d_h / r_h
            else:
                ratio = d_w / r_w
            Rs[0, 0] *= ratio
            Rs[1, 1] *= ratio

            T = np.eye(3)
            T[0, 2] = 0.5 * d_w
            T[1, 2] = 0.5 * d_h
            return T @ Rs @ C

        Rs[0, 0] *= d_w / r_w
        Rs[1, 1] *= d_h / r_h
        return Rs
