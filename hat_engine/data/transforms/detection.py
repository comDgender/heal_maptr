# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import logging
import math
import random
from typing import Tuple

import cv2

from hat_engine.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)

__all__ = [
    "RandomResizedCrop",
]


@OBJECT_REGISTRY.register
class RandomResizedCrop(object):  # noqa: D205,D400
    """Torchvision's variant of crop a random part of the input,
    and rescale it to some size.

    Used for np.ndarray. This transform is same as
    albumentations.augmentations.transforms.RandomResizedCrop.

    Args:
        height: height after crop and resize.
        width: width after crop and resize.
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped.
        interpolation: flag that is used to specify the interpolation
            algorithm. Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR,
            cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p: probability of applying the transform. Default: 1.
    """

    def __init__(
        self,
        height: int,
        width: int,
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (0.75, 1.3333333333333333),
        interpolation: int = cv2.INTER_LINEAR,
        p: float = 1.0,
    ):
        self.height = height
        self.width = width
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.p = p

    def get_params_dependent_on_targets(self, data):
        img = data["img"]
        area = img.shape[0] * img.shape[1]

        for _ in range(10):
            target_area = random.uniform(*self.scale) * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= img.shape[1] and 0 < h <= img.shape[0]:
                i = random.randint(0, img.shape[0] - h)
                j = random.randint(0, img.shape[1] - w)
                return {
                    "crop_height": h,
                    "crop_width": w,
                    "h_start": i * 1.0 / (img.shape[0] - h + 1e-10),
                    "w_start": j * 1.0 / (img.shape[1] - w + 1e-10),
                }

        # Fallback to central crop
        in_ratio = img.shape[1] / img.shape[0]
        if in_ratio < min(self.ratio):
            w = img.shape[1]
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = img.shape[0]
            w = int(round(h * max(self.ratio)))
        else:  # whole image
            w = img.shape[1]
            h = img.shape[0]
        i = (img.shape[0] - h) // 2
        j = (img.shape[1] - w) // 2
        return {
            "crop_height": h,
            "crop_width": w,
            "h_start": i * 1.0 / (img.shape[0] - h + 1e-10),
            "w_start": j * 1.0 / (img.shape[1] - w + 1e-10),
        }

    def _get_random_crop_coords(
        self, height, width, crop_height, crop_width, h_start, w_start
    ):
        y1 = int((height - crop_height) * h_start)
        y2 = y1 + crop_height
        x1 = int((width - crop_width) * w_start)
        x2 = x1 + crop_width
        return x1, y1, x2, y2

    def _crop_img(self, img, x1, y1, x2, y2):
        img = img[y1:y2, x1:x2]
        return img

    def _crop_scale_lines(self, gt_lines, x1, y1, scale_x, scale_y):
        for gt_line in gt_lines:
            gt_line[:, 0] = (gt_line[:, 0] - x1) * scale_x
            gt_line[:, 1] = (gt_line[:, 1] - y1) * scale_y
        return gt_lines

    def __call__(self, data):
        if random.random() < self.p:
            img = data["img"]
            img_height = img.shape[0]
            img_width = img.shape[1]
            params = self.get_params_dependent_on_targets(data)

            crop_height = params["crop_height"]
            crop_width = params["crop_width"]
            h_start = params["h_start"]
            w_start = params["w_start"]

            x1, y1, x2, y2 = self._get_random_crop_coords(
                img_height,
                img_width,
                crop_height,
                crop_width,
                h_start,
                w_start,
            )
            croped_img = self._crop_img(img, x1, y1, x2, y2)
            resized_img = cv2.resize(
                croped_img,
                dsize=(self.width, self.height),
                interpolation=self.interpolation,
            )
            data["img"] = resized_img

            if "gt_lines" in data:
                gt_lines = data["gt_lines"]
                scale_x = self.width / crop_width
                scale_y = self.height / crop_height

                data["gt_lines"] = self._crop_scale_lines(
                    gt_lines,
                    x1,
                    y1,
                    scale_x,
                    scale_y,
                )
        return data
