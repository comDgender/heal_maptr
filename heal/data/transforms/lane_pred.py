# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
# Lane-prediction specific image augmentations. These classes are not in
# heal.data.transforms.detection (which only covers YOLOX-style detection
# transforms), so they live here. The shared Resize/RandomFlip/ToTensor/
# FixedCrop used by lane_pred come from heal.data.transforms.detection.

import logging
import math
import random
from typing import Optional, Tuple

import cv2
import numpy as np
from hat_engine.data.utils import encode_decode_img
from hat_engine.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)

__all__ = [
    "RGBShift",
    "HueSaturationValue",
    "JPEGCompress",
    "MeanBlur",
    "MedianBlur",
    "RandomBrightnessContrast",
    "ShiftScaleRotate",
    "LaneRandomResizedCrop",
]


@OBJECT_REGISTRY.register
class HueSaturationValue(object):  # noqa: D205,D400
    """Randomly change hue, saturation and value of the input image.

    Used for unit8 np.ndarray, RGB image input.

    Args:
        hue_range: range for changing hue. Default: (-20, 20).
        sat_range: range for changing saturation. Default: (-30, 30).
        val_range: range for changing value. Default: (-20, 20).
        p: probability of applying the transform. Default: 0.5.
    """

    def __init__(
        self,
        hue_range: Tuple[float, float] = (-20, 20),
        sat_range: Tuple[float, float] = (-30, 30),
        val_range: Tuple[float, float] = (-20, 20),
        p: float = 0.5,
    ):
        self.hue_range = hue_range
        self.sat_range = sat_range
        self.val_range = val_range
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            img = data["img"]
            r = [
                random.uniform(self.hue_range[0], self.hue_range[1]),
                random.uniform(self.sat_range[0], self.sat_range[1]),
                random.uniform(self.val_range[0], self.val_range[1]),
            ]

            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
            dtype = img.dtype

            x = np.arange(0, 256, dtype=np.int16)

            lut_hue = ((x + r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x + r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x + r[2], 0, 255).astype(dtype)
            img_hsv = cv2.merge(
                (
                    cv2.LUT(hue, lut_hue),
                    cv2.LUT(sat, lut_sat),
                    cv2.LUT(val, lut_val),
                )
            ).astype(dtype)
            img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
            data["img"] = img
        return data


@OBJECT_REGISTRY.register
class RGBShift(object):  # noqa: D205,D400
    """Randomly shift values for each channel of the input image.

    Used for np.ndarray.

    Args:
        r_shift_limit: range for changing values for the red channel.
        g_shift_limit: range for changing values for the green channel.
        b_shift_limit: range for changing values for the blue channel.
        p: probability of applying the transform. Default: 0.5.
    """

    def __init__(
        self,
        r_shift_limit: Tuple[float, float] = (-20, 20),
        g_shift_limit: Tuple[float, float] = (-20, 20),
        b_shift_limit: Tuple[float, float] = (-20, 20),
        p: float = 0.5,
    ):
        self.r_shift_limit = r_shift_limit
        self.g_shift_limit = g_shift_limit
        self.b_shift_limit = b_shift_limit
        self.p = p

    def get_params(self):
        r_shift = random.uniform(self.r_shift_limit[0], self.r_shift_limit[1])
        g_shift = random.uniform(self.g_shift_limit[0], self.g_shift_limit[1])
        b_shift = random.uniform(self.b_shift_limit[0], self.b_shift_limit[1])

        return r_shift, g_shift, b_shift

    def __call__(self, data):
        if random.random() < self.p:
            r_shift, g_shift, b_shift = self.get_params()

            img = data["img"]
            color_space = data.get("color_space", None)

            if color_space is None:
                color_space = "bgr"
                logger.warning(
                    "current color_space is unknown, treat as bgr "
                    "by default"
                )
            assert color_space.lower() in [
                "rgb",
                "bgr",
            ], "color_space must one of rgb and bgr using RGBShift"
            rgb_input = True if color_space.lower() == "rgb" else False

            layout = data.get("layout", None)
            if layout is None:
                layout = "hwc"
                logger.warning(
                    "current layout is unknown, treat as hwc " "by default"
                )
            assert layout.lower() in [
                "hwc",
                "chw",
            ], "layout must one of hwc and chw using RGBShift"

            hwc_input = True if layout.lower() == "hwc" else False
            if not hwc_input:
                img = np.ascontiguousarray(img.transpose((1, 2, 0)))
            shift_rgb_img = self.shift_rgb(
                img=img,
                r_shift=r_shift,
                g_shift=g_shift,
                b_shift=b_shift,
                rgb_input=rgb_input,
            )
            if not hwc_input:
                shift_rgb_img = np.ascontiguousarray(
                    shift_rgb_img.transpose((2, 0, 1))
                )
            data["img"] = shift_rgb_img
        return data

    def shift_rgb(self, img, r_shift, g_shift, b_shift, rgb_input):
        if not rgb_input:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        if img.dtype == np.uint8:
            shift_img = self._shift_rgb_uint8(img, r_shift, g_shift, b_shift)
        else:
            shift_img = self._shift_rgb_non_uint8(
                img, r_shift, g_shift, b_shift
            )
        if not rgb_input:
            cv2.cvtColor(shift_img, cv2.COLOR_RGB2BGR, shift_img)
        return shift_img

    def _shift_image_uint8(self, img, value):
        max_value = 255
        lut = np.arange(0, max_value + 1).astype("float32")
        lut += value
        lut = np.clip(lut, 0, max_value).astype(img.dtype)
        return cv2.LUT(img, lut)

    def _shift_rgb_uint8(self, img, r_shift, g_shift, b_shift):
        h, w, c = img.shape
        if r_shift == g_shift == b_shift:
            img = img.reshape([h, w * c])
            result_img = self._shift_image_uint8(img, r_shift)
        else:
            result_img = np.empty_like(img)
            shifts = [r_shift, g_shift, b_shift]
            for i, shift in enumerate(shifts):
                result_img[..., i] = self._shift_image_uint8(
                    img[..., i], shift
                )

        return result_img.reshape([h, w, c])

    def _shift_rgb_non_uint8(self, img, r_shift, g_shift, b_shift):
        if r_shift == g_shift == b_shift:
            return img + r_shift

        result_img = np.empty_like(img)
        shifts = [r_shift, g_shift, b_shift]
        for i, shift in enumerate(shifts):
            result_img[..., i] = img[..., i] + shift
        return result_img


@OBJECT_REGISTRY.register
class JPEGCompress(object):
    """Do JPEG compression to downgrade image quality.

    Affected keys: img.

    Args:
        p : prob
        max_quality : (0, 100] JPEG compression highest quality
        min_quality : (0, 100] JPEG compression lowest quality
    """

    def __init__(
        self,
        p: float = 0.08,
        max_quality: int = 95,
        min_quality: int = 30,
    ):
        self.p = p
        self.max_quality = max_quality
        self.min_quality = min_quality

    def _jpeg_compress_aug(self, data):
        do_compress = np.random.choice([False, True], p=[1 - self.p, self.p])

        if do_compress:
            img = data["img"]
            scale = random.random()
            quality = (
                scale * (self.max_quality - self.min_quality)
                + self.min_quality
            )
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
            if np.random.uniform() < 0.5:
                decimg = encode_decode_img(img, ".jpg", encode_param)
            else:
                img_bgr = img[:, :, [2, 1, 0]]
                decimg = encode_decode_img(img_bgr, ".jpg", encode_param)
                decimg = decimg[:, :, [2, 1, 0]]
            data["img"] = decimg

    def __call__(self, data):
        self._jpeg_compress_aug(data)
        return data


@OBJECT_REGISTRY.register
class MeanBlur(object):  # noqa: D205,D400
    """Apply mean blur to the input image using a fix-sized kernel.

    Args:
        ksize: maximum kernel size for blurring the input image. Default: 3.
        p: probability of applying the transform. Default: 0.5.
    """

    def __init__(self, ksize: int = 3, p: float = 0.5):
        self.ksize = ksize
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            blur_img = cv2.blur(data["img"], ksize=(self.ksize, self.ksize))
            data["img"] = blur_img
        return data


@OBJECT_REGISTRY.register
class MedianBlur(object):  # noqa: D205,D400
    """Apply median blur to the input image using a fix-sized kernel.

    Args:
        ksize: maximum kernel size for blurring the input image. Default: 3.
        p: probability of applying the transform. Default: 0.5.
    """

    def __init__(self, ksize: int = 3, p: float = 0.5):
        if ksize % 2 != 1:
            raise ValueError("MedianBlur supports only odd blur limits.")
        self.ksize = ksize
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            median_blur_img = cv2.medianBlur(data["img"], ksize=self.ksize)
            data["img"] = median_blur_img
        return data


@OBJECT_REGISTRY.register
class RandomBrightnessContrast(object):
    """Randomly change brightness and contrast of the input image.

    Used for unit8 np.ndarray.

    Args:
        brightness_limit: factor range for changing brightness. Default: (-0.2, 0.2).  # noqa
        contrast_limit: factor range for changing contrast. Default: (-0.2, 0.2).  # noqa
        brightness_by_max: If True adjust contrast by image dtype maximum,
            else adjust contrast by image mean.
        p: probability of applying the transform. Default: 0.5.
    """

    def __init__(
        self,
        brightness_limit: Tuple[float, float] = (-0.2, 0.2),
        contrast_limit: Tuple[float, float] = (-0.2, 0.2),
        brightness_by_max: bool = True,
        p=0.5,
    ):
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.brightness_by_max = brightness_by_max
        self.p = p

    def _brightness_contrast_adjust(
        self, img, alpha=1, beta=0, beta_by_max=False
    ):
        dtype = np.dtype("uint8")
        max_value = 255
        lut = np.arange(0, max_value + 1).astype("float32")
        if alpha != 1:
            lut *= alpha
        if beta != 0:
            if beta_by_max:
                lut += beta * max_value
            else:
                lut += beta * np.mean(img)

        lut = np.clip(lut, 0, max_value).astype(dtype)
        img = cv2.LUT(img, lut)
        return img

    def __call__(self, data):
        if random.random() < self.p:
            img = data["img"]
            alpha = 1.0 + random.uniform(
                self.contrast_limit[0],
                self.contrast_limit[1],
            )
            beta = random.uniform(
                self.brightness_limit[0],
                self.brightness_limit[1],
            )

            result_img = self._brightness_contrast_adjust(
                img, alpha, beta, self.brightness_by_max
            )
            data["img"] = result_img
        return data


@OBJECT_REGISTRY.register
class ShiftScaleRotate(object):
    """Randomly apply affine transforms: translate, scale and rotate.

    Used for np.ndarray hwc img. Also transforms gt_lines with the same
    affine matrix.

    Args:
        shift_limit: shift factor range for both height and width. Default: (-0.0625, 0.0625).  # noqa
        scale_limit: scaling factor range. Default: (-0.1, 0.1).
        rotate_limit: rotation range. Default: (-45, 45).
        interpolation: cv2 interpolation flag. Default: cv2.INTER_LINEAR.
        border_mode: cv2 border extrapolation flag. Default: cv2.BORDER_REFLECT_101.  # noqa
        value: padding value if border_mode is cv2.BORDER_CONSTANT.
        p: probability of applying the transform. Default: 0.5.
    """

    def __init__(
        self,
        shift_limit: Tuple[float, float] = (-0.0625, 0.0625),
        scale_limit: Tuple[float, float] = (-0.1, 0.1),
        rotate_limit: Tuple[float, float] = (-45.0, 45.0),
        interpolation: int = cv2.INTER_LINEAR,
        border_mode: int = cv2.BORDER_REFLECT_101,
        value: Optional[int] = None,
        p: float = 0.5,
    ):
        self.shift_limit = shift_limit
        self.scale_limit = scale_limit
        self.rotate_limit = rotate_limit
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.p = p

    def _shift_scale_rotate_lines(self, data, matrix):
        gt_lines = data["gt_lines"]
        for i in range(len(gt_lines)):
            gt_lines[i] = cv2.transform(
                np.expand_dims(gt_lines[i], axis=0), matrix
            ).squeeze()
        data["gt_lines"] = gt_lines

    def _get_rotation_matrix(self, height, width, angle, scale, dx, dy):
        center = (width / 2, height / 2)
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        matrix[0, 2] += dx * width
        matrix[1, 2] += dy * height
        return matrix

    def __call__(self, data):
        if random.random() < self.p:
            height, width = data["img"].shape[:2]
            angle = random.uniform(self.rotate_limit[0], self.rotate_limit[1])
            scale = random.uniform(self.scale_limit[0], self.scale_limit[1])
            dx = random.uniform(self.shift_limit[0], self.shift_limit[1])
            dy = random.uniform(self.shift_limit[0], self.shift_limit[1])

            mat = self._get_rotation_matrix(
                height, width, angle, scale, dx, dy
            )

            data["img"] = cv2.warpAffine(
                src=data["img"],
                M=mat,
                dsize=(width, height),
                flags=self.interpolation,
                borderMode=self.border_mode,
                borderValue=self.value,
            )

            if "gt_lines" in data:
                self._shift_scale_rotate_lines(data, matrix=mat)
        return data


@OBJECT_REGISTRY.register
class LaneRandomResizedCrop(object):  # noqa: D205,D400
    """Crop a random part of the input and rescale it to some size.

    Used for np.ndarray. Also crops and rescales gt_lines.

    Args:
        height: height after crop and resize.
        width: width after crop and resize.
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped.
        interpolation: cv2 interpolation flag. Default: cv2.INTER_LINEAR.
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
            # Sample aspect in log-space so the ratio range is uniform.
            ar_lo, ar_hi = math.log(self.ratio[0]), math.log(self.ratio[1])
            ar = math.exp(random.uniform(ar_lo, ar_hi))
            crop_w = int(round(math.sqrt(target_area * ar)))
            crop_h = int(round(math.sqrt(target_area / ar)))
            img_h, img_w = img.shape[0], img.shape[1]
            if 0 < crop_w <= img_w and 0 < crop_h <= img_h:
                top = random.randint(0, img_h - crop_h)
                left = random.randint(0, img_w - crop_w)
                return {
                    "crop_height": crop_h,
                    "crop_width": crop_w,
                    "h_start": top / (img_h - crop_h + 1e-10),
                    "w_start": left / (img_w - crop_w + 1e-10),
                }

        # Fallback: clamp crop to image while respecting ratio bounds.
        img_h, img_w = img.shape[0], img.shape[1]
        img_ar = img_w / img_h
        ratio_lo, ratio_hi = min(self.ratio), max(self.ratio)
        if img_ar < ratio_lo:
            crop_w = img_w
            crop_h = int(round(crop_w / ratio_lo))
        elif img_ar > ratio_hi:
            crop_h = img_h
            crop_w = int(round(crop_h * ratio_hi))
        else:
            crop_w, crop_h = img_w, img_h
        top = (img_h - crop_h) // 2
        left = (img_w - crop_w) // 2
        return {
            "crop_height": crop_h,
            "crop_width": crop_w,
            "h_start": top / (img_h - crop_h + 1e-10),
            "w_start": left / (img_w - crop_w + 1e-10),
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
