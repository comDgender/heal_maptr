# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

# (Resize,
# Pad, RandomFlip, Normalize, ToTensor, AugmentHSV, Mosaic).

import copy
import logging
import math
import random
from typing import Dict, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from hat_engine.registry import OBJECT_REGISTRY

from heal.core.box_utils import bbox_overlaps
from heal.data.transforms.functional_img import (
    image_normalize,
    image_pad,
    imresize,
    imresize_pad_to_keep_ratio,
    random_flip,
)

logger = logging.getLogger(__name__)

__all__ = [
    "Resize",
    "RandomFlip",
    "Pad",
    "Normalize",
    "ToTensor",
    "AugmentHSV",
    "Mosaic",
    "Resize3D",
    "RandomCrop",
    "FixedCrop",
]


@OBJECT_REGISTRY.register
class Resize(object):
    """Resize image & bbox & gt_lines.

    Affected keys: img, ori_img, img_shape, resized_shape, pad_shape,
    scale_factor, keep_ratio, gt_bboxes, gt_lines.
    """

    def __init__(
        self,
        img_scale: Union[Sequence[int], Sequence[Sequence[int]]] = None,
        max_scale: Union[Sequence[int], Sequence[Sequence[int]]] = None,
        multiscale_mode: str = "range",
        ratio_range: Tuple[float, float] = None,
        keep_ratio: bool = True,
        pad_to_keep_ratio: bool = False,
        interpolation: str = "bilinear",
        raw_scaler_enable: bool = False,
        sample1c_enable: bool = False,
        divisor: int = 1,
        rm_neg_coords: bool = True,
        split_transform: bool = False,
        split_trans_w: int = 256,
        split_trans_h: int = 256,
    ):
        # Normalize to None or a list of per-candidate (h, w)-like sequences.
        if img_scale is None:
            self.img_scale = None
        elif not isinstance(img_scale, (list, tuple)):
            self.img_scale = [img_scale]
        elif isinstance(img_scale[0], (tuple, list)):
            self.img_scale = list(img_scale)
        else:
            self.img_scale = [img_scale]

        if self.img_scale is not None:
            for scale in self.img_scale:
                if not isinstance(scale, (tuple, list)):
                    raise AssertionError(
                        "img_scale must be a sequence or a list of sequences"
                    )

        self.max_scale = max_scale
        if ratio_range is not None:
            if len(self.img_scale) != 1:
                raise AssertionError(
                    "ratio_range requires exactly one img_scale entry"
                )
        elif multiscale_mode not in ("value", "range", "max_size"):
            raise AssertionError(
                f"unsupported multiscale_mode: {multiscale_mode}"
            )
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.pad_to_keep_ratio = pad_to_keep_ratio

        allowed_interp = (
            "nearest",
            "bilinear",
            "bicubic",
            "area",
            "lanczos",
        )
        if interpolation not in allowed_interp:
            raise AssertionError(
                f"interpolation must be one of {allowed_interp}"
            )
        self.interpolation = interpolation
        self.raw_scaler_enable = raw_scaler_enable
        self.sample1c_enable = sample1c_enable
        self.divisor = divisor
        self.rm_neg_coords = rm_neg_coords
        self.split_transform = split_transform
        self.split_trans_w = split_trans_w
        self.split_trans_h = split_trans_h

    @staticmethod
    def random_select(img_scales):
        """Pick one scale from a discrete candidate list uniformly."""
        idx = int(np.random.randint(0, len(img_scales)))
        return img_scales[idx], idx

    @staticmethod
    def random_sample(img_scales):
        """Sample long/short edges between the two endpoint scales."""
        if len(img_scales) != 2:
            raise AssertionError(
                f"random_sample needs 2 scales, got {len(img_scales)}"
            )
        long_lo, long_hi = sorted(max(edge) for edge in img_scales)
        short_lo, short_hi = sorted(min(edge) for edge in img_scales)
        sampled_long = int(np.random.randint(long_lo, long_hi + 1))
        sampled_short = int(np.random.randint(short_lo, short_hi + 1))
        return (sampled_long, sampled_short), None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Scale a base (h, w) by a uniform random ratio in ``ratio_range``."""
        if not isinstance(img_scale, (tuple, list)) or len(img_scale) != 2:
            raise AssertionError(
                "img_scale must be a length-2 sequence (height, width)"
            )
        ratio_lo, ratio_hi = ratio_range
        ratio = float(np.random.uniform(ratio_lo, ratio_hi))
        scaled = (int(img_scale[0] * ratio), int(img_scale[1] * ratio))
        return scaled, 0

    @staticmethod
    def max_size(max_scale, origin_shape):
        if max(origin_shape) > max(max_scale):
            resize_scale = max_scale
        else:
            resize_scale = origin_shape
        return resize_scale, 0

    def _random_scale(self, data):
        if "layout" not in data:
            data["layout"] = "hwc"
        if "img_shape" not in data:
            data["img_shape"] = np.array(data["img"].shape)

        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range
            )
        elif self.multiscale_mode == "max_size":
            scale, scale_idx = self.max_size(
                self.max_scale, data["img_shape"][:2]
            )
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == "range":
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == "value":
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError
        data["scale"] = scale
        data["scale_idx"] = scale_idx

    def _resize_img(self, data):
        h, w = data["scale"][0], data["scale"][1]
        img = data["img"]
        layout = data["layout"]
        if self.pad_to_keep_ratio and self.keep_ratio:
            resized_img, R = imresize_pad_to_keep_ratio(
                img, (h, w), layout, keep_ratio=True
            )
            data["scale_factor"] = R
        else:
            resized_img, _, w_scale, h_scale = imresize(
                img,
                w,
                h,
                layout,
                keep_ratio=self.keep_ratio,
                return_scale=True,
                interpolation=self.interpolation,
                raw_scaler_enable=self.raw_scaler_enable,
                raw_pattern=data.get("cur_pattern"),
                sample1c_enable=self.sample1c_enable,
                divisor=self.divisor,
                split_transform=self.split_transform,
                split_trans_w=self.split_trans_w,
                split_trans_h=self.split_trans_h,
            )
            data["scale_factor"] = np.array(
                [w_scale, h_scale, w_scale, h_scale], dtype=np.float32
            )

        if "ori_img" in data:
            ori_img = data["ori_img"]
            if self.pad_to_keep_ratio:
                resized_ori_img, _ = imresize_pad_to_keep_ratio(
                    ori_img, (h, w), layout, keep_ratio=self.keep_ratio
                )
            else:
                resized_ori_img, _, _, _ = imresize(
                    ori_img,
                    w,
                    h,
                    layout,
                    keep_ratio=self.keep_ratio,
                    return_scale=True,
                    interpolation=self.interpolation,
                    divisor=self.divisor,
                )
            data["resized_ori_img"] = resized_ori_img

        data["img"] = resized_img
        data["img_shape"] = resized_img.shape
        data["resized_shape"] = resized_img.shape
        data["pad_shape"] = resized_img.shape
        data["keep_ratio"] = self.keep_ratio

    def _resize_bbox(self, data):
        if "gt_bboxes" not in data or not data["gt_bboxes"].any():
            return
        if data["layout"] == "hwc":
            h, w = data["img_shape"][:2]
        else:
            h, w = data["img_shape"][1:]
        bboxes = data["gt_bboxes"]
        if self.pad_to_keep_ratio and self.keep_ratio:
            n = bboxes.shape[0]
            R = data["scale_factor"]
            xy = np.ones((n * 4, 3))
            xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
            xy = xy @ R.T
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = (
                np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)))
                .reshape(4, n)
                .T
            )
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, w - 1)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, h - 1)
            bboxes = xy.astype(np.float32)
        else:
            bboxes = bboxes * data["scale_factor"]
            scale_offset = data.get("scale_offset", (0, 0))
            bboxes[:, 0::2] += scale_offset[1]
            bboxes[:, 1::2] += scale_offset[0]
            if self.rm_neg_coords:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, w - 1)
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, h - 1)
        data["gt_bboxes"] = bboxes

    def _resize_lines(self, data):
        scale_factor = data["scale_factor"]
        w_scale, h_scale = scale_factor[0], scale_factor[1]
        gt_lines = data["gt_lines"]
        for gt_line in gt_lines:
            gt_line[:, 0] *= w_scale
            gt_line[:, 1] *= h_scale
        data["gt_lines"] = gt_lines

    def inverse_transform(self, inputs, task_type: str, inverse_info: Dict):
        """Inverse of resize to map prediction回原图.

        Args:
            inputs: 预测结果，检测为 (N, >=4) 的 boxes，分割为 mask。
            task_type: `detection` 或 `segmentation`。
            inverse_info: 包含 `scale_factor` 等信息的字典。
        """
        if task_type == "detection":
            scale_factor = inverse_info["scale_factor"]
            if not isinstance(scale_factor, torch.Tensor):
                scale_factor = inputs.new_tensor(scale_factor)
            # HAT 中假定 boxes 在前 4 维
            inputs[:, :4] = inputs[:, :4] / scale_factor
            return inputs
        elif task_type == "segmentation":
            scale_factor = inverse_info["scale_factor"][:2]
            if isinstance(scale_factor, torch.Tensor):
                scale_factor = scale_factor.detach().cpu().numpy()
            elif isinstance(scale_factor, (tuple, list)):
                scale_factor = np.array(scale_factor)
            else:
                assert isinstance(scale_factor, np.ndarray)
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.detach().cpu().numpy()
            before_resize_shape = inputs.shape / scale_factor
            out_height, out_width = before_resize_shape
            out_img = cv2.resize(
                inputs,
                (int(out_width), int(out_height)),
                interpolation=cv2.INTER_NEAREST,
            )
            return out_img
        else:
            # 其他 task_type 直接返回
            return inputs

    def __call__(self, data):
        self._random_scale(data)
        self._resize_img(data)
        if "gt_bboxes" in data:
            self._resize_bbox(data)
        if "gt_lines" in data:
            self._resize_lines(data)
        return data


@OBJECT_REGISTRY.register
class RandomFlip(object):
    """Flip image & bbox & gt_lines.

    Affected keys: img, gt_bboxes, gt_lines.
    """

    def __init__(self, px: Optional[float] = 0.5, py: Optional[float] = 0):
        assert 0 <= px <= 1 and 0 <= py <= 1
        self.px = px
        self.py = py

    def _flip_img(self, data):
        raw_pattern = data.get("cur_pattern")
        data["img"], (flip_x, flip_y), raw_pattern = random_flip(
            data["img"], data["layout"], self.px, self.py, raw_pattern
        )
        if raw_pattern is not None:
            data["cur_pattern"] = raw_pattern
        return flip_x, flip_y

    def _flip_bbox(self, data):
        if "img_shape" not in data:
            img_shape = data["img"].shape
        else:
            img_shape = data.get("pad_shape", data["img_shape"])
        h, w = img_shape[:2]
        bboxes = data["gt_bboxes"]
        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        if self.flip_x:
            flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
            flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
        if self.flip_y:
            flipped[..., 1::4] = h - bboxes[..., 3::4] - 1
            flipped[..., 3::4] = h - bboxes[..., 1::4] - 1
        data["gt_bboxes"] = flipped

    def _flip_point_sets(self, data, key):
        if "img_shape" not in data:
            height, width = (data["img_height"], data["img_width"])
        else:
            height, width, _ = data["img_shape"]

        gt_point_sets = data[key]
        for gt_point_set in gt_point_sets:
            if self.flip_x:
                gt_point_set[:, 0] = (width - 1) - gt_point_set[:, 0]
            if self.flip_y:
                gt_point_set[:, 1] = (height - 1) - gt_point_set[:, 1]
        data[key] = gt_point_sets

    def __call__(self, data):
        if "layout" not in data:
            data["layout"] = "hwc"
        flip_x, flip_y = self._flip_img(data)
        self.flip_x = flip_x
        self.flip_y = flip_y
        if "gt_bboxes" in data:
            self._flip_bbox(data)
        if "gt_lines" in data:
            self._flip_point_sets(data, "gt_lines")
        return data


@OBJECT_REGISTRY.register
class RandomCrop(object):
    def __init__(
        self,
        size: Optional[Tuple[int, int]] = None,
        min_area: float = -1,
        min_iou: float = -1,
        center_crop_prob: float = 0.0,
        center_shake: Optional[Tuple[float, float]] = None,
        truncate_gt: bool = True,
        h_ratio_range: Optional[Tuple[float, float]] = None,
        w_ratio_range: Optional[Tuple[float, float]] = None,
        wh_ratio_range: Optional[Tuple[float, float]] = None,
        discriminate_ignore_classes: bool = False,
        keep_raw_pattern: bool = False,
        without_background: bool = False,
        crop_around_gt: bool = False,
        repeat_times: int = 1,
        inclusion_rate: float = 0.0,
        filter_area: bool = True,
        rm_neg_coords: bool = True,
        discard_truncate: bool = False,
    ):  # noqa: D205,D400
        """
        .. note::
            Affected keys: 'img', 'img_shape', 'pad_shape', 'layout',
            'gt_bboxes', 'gt_classes', 'gt_seg', 'gt_flow'.

        Args:
            size: Expected size after cropping, (h, w). If not given, it will
                be changed to shape of each image.
            min_area: If min_area > 0, boxes whose areas are less than
                min_area will be ignored.
            min_iou: If min_iou > 0, boxes whose iou between before and
                after truncation < min_iou will be ignored.
            center_crop_prob: The center_crop_prob is the center crop
                probability.
            center_shake: The list is the center shake's top, bottom, left,
                right range.
            truncate_gt: if True, truncate the gt_boxes when the gt_boxes
                exceed the crop area. Default True.
            h_ratio_range: lower & upper bound to randomly change the height
                of target size.
            w_ratio_range: lower & upper bound to randomly change the width
                of target size.
            wh_ratio_range: lower & upper bound to randomly change the width
                and height of target size simultaneously.
            discriminate_ignore_classes: if True, ignored area by min_iou
                retain the original class info. And class id should greater
                than 0. Default False. Support class id starts from 0.
            keep_raw_pattern: if True, limit the crop coordinates to keep raw
                pattern when crop raw image.
            without_background: if True, the class label starts from 0, else 0
                represents background.
            crop_around_gt: if Ture, will do the image crop around the gt
                using the function _crop_img_around_gt() instead of
                _crop_img(). This helps generate more effective training
                examples. Default False.
            repeat_times: When one crop has no gt bbox or few gt bboxes, e.g.
                the number of gt_bboxes in this crop is less than
                total_num_of_gt_bboxes x inclusion_rate in it,
                _crop_img_around_gt() will do repeat_times random crop so that
                there are required inclusion_rate gt_bboxes in this crop.
                Default 1.
            inclusion_rate: The minimum proportion of gt bboxes you want one
                crop to include. Default 0.0.
            filter_area: Whether to filter boxes with small area.
            rm_neg_coords: Whether to remove negative coordinates.
            discard_truncate: Whether to remove truncate coordinates.
        """
        self.size = size
        self.min_area = min_area
        self.min_iou = min_iou
        assert 0 <= center_crop_prob <= 1
        self.center_crop_prob = center_crop_prob
        self.center_shake = center_shake
        self.truncate_gt = truncate_gt
        assert size is None or (
            isinstance(size, tuple) and len(size) == 2
        ), "size must be tuple and of size 2"
        assert h_ratio_range is None or (
            isinstance(h_ratio_range, tuple) and len(h_ratio_range) == 2
        ), "h_ratio_range must be tuple and of size 2"
        assert w_ratio_range is None or (
            isinstance(w_ratio_range, tuple) and len(w_ratio_range) == 2
        ), "w_ratio_range must be tuple and of size 2"
        assert wh_ratio_range is None or (
            isinstance(wh_ratio_range, tuple) and len(wh_ratio_range) == 2
        ), "w_ratio_range must be tuple and of size 2"
        self.w_ratio_range = w_ratio_range
        self.h_ratio_range = h_ratio_range
        self.wh_ratio_range = wh_ratio_range
        self.discriminate_ignore_classes = discriminate_ignore_classes
        self.keep_raw_pattern = keep_raw_pattern
        self.without_background = without_background
        self.crop_around_gt = crop_around_gt
        assert repeat_times >= 1
        self.repeat_times = repeat_times
        assert 0 <= inclusion_rate <= 1
        self.inclusion_rate = inclusion_rate
        self.filter_area = filter_area
        self.rm_neg_coords = rm_neg_coords
        self.discard_truncate = discard_truncate

    def _union_of_bboxes(self, data: Dict):
        """Calculate union of bounding boxes.

        Args:
            data (dict): A dict that includes all information of an image.

        Returns:
            tuple: A minimum bounding box `(x_min, y_min, x_max, y_max)` that
                includes all gt bboxes.
        """
        if data["layout"] == "hwc":
            h, w = data["img_shape"][:2]
        else:
            h, w = data["img_shape"][1:]
        x1, y1 = w, h
        x2, y2 = 0, 0
        bboxes = data["gt_bboxes"]
        classes = data["gt_classes"]
        # if there is any gt bboxes
        if (classes >= 0).any():
            for bbox, cls in zip(bboxes, classes):
                if cls < 0:
                    continue
                x_min, y_min, x_max, y_max = bbox[:4]
                x1, y1 = int(np.min([x1, x_min])), int(np.min([y1, y_min]))
                x2, y2 = int(np.max([x2, x_max])), int(np.max([y2, y_max]))
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)
            return np.array([x1, y1, x2, y2], dtype=int)
        return np.zeros((0, 4), dtype=int)

    def _is_good_enough(self, data: Dict, crop_bbox: np.ndarray):
        """Test if one crop has enough gt bboxes in it.

        Args:
            data (dict): A dict that includes all information of an image.
            crop_bbox (np.array): The cropped box.

        Returns:
            bool: If the cropped box has enough gt bboxes in it.
        """

        bboxes = data["gt_bboxes"].copy()
        classes = data["gt_classes"].copy()
        valid_indexes = classes >= 0
        if valid_indexes.any():
            bboxes = bboxes[valid_indexes, :].reshape(-1, 4)
            crop_bbox = crop_bbox.reshape(1, -1)
            # Note the order of bboxes and crop_bbox in this function, which
            # can't be swapped
            iofs = bbox_overlaps(bboxes, crop_bbox, mode="iof")
            rate = iofs.sum() / bboxes.shape[0]
            if rate >= self.inclusion_rate:
                return True
            else:
                return False
        else:
            return True

    @staticmethod
    def _get_rand_ratio(ratio_range):
        if ratio_range is None:
            return 1
        rrange = ratio_range[1] - ratio_range[0]
        assert rrange >= 0
        rratio = np.random.rand() * rrange + ratio_range[0]
        return rratio

    def _get_crop_region(self, data, iter_i=0):
        img_shape = data["img_shape"]
        layout = data["layout"]
        assert layout in ["hwc", "chw"], (
            "layout of img must be `chw` or " "`hwc`"
        )
        do_center_crop = np.random.rand() < self.center_crop_prob

        if layout == "chw":
            img_shape = (img_shape[1], img_shape[2], img_shape[0])
        img_h, img_w = img_shape[:2]
        crop_size = (
            list(self.size) if self.size is not None else [img_h, img_w]
        )
        rratio = self._get_rand_ratio(self.wh_ratio_range)
        crop_size = [s * rratio for s in crop_size]
        crop_size[0] *= self._get_rand_ratio(self.h_ratio_range)
        crop_size[1] *= self._get_rand_ratio(self.w_ratio_range)

        margin_h = max(img_shape[0] - crop_size[0], 1)
        margin_w = max(img_shape[1] - crop_size[1], 1)
        lower_lim_x1, upper_lim_x1 = 0, margin_w
        lower_lim_y1, upper_lim_y1 = 0, margin_h

        if self.crop_around_gt and not do_center_crop:
            # get the union box of all gt boxes
            union_box = self._union_of_bboxes(data)
            if union_box.any():
                union_x1, union_y1, union_x2, union_y2 = union_box
            else:
                union_x1, union_y1, union_x2, union_y2 = (
                    0,
                    0,
                    img_w - 1,
                    img_h - 1,
                )
            # get the range of the crop's up-left corner
            lower_lim_x1 = max(0, int(min(union_x1, union_x2 - crop_size[1])))
            lower_lim_y1 = max(0, int(min(union_y1, union_y2 - crop_size[0])))
            upper_lim_x1 = min(
                img_w, int(max(union_x1, union_x2 - crop_size[1]))
            )
            upper_lim_y1 = min(
                img_h, int(max(union_y1, union_y2 - crop_size[0]))
            )

            # avoid the cropped box exceeding the image boundary
            upper_lim_x1 = min(upper_lim_x1, int(max(0, img_w - crop_size[1])))
            upper_lim_y1 = min(upper_lim_y1, int(max(0, img_h - crop_size[0])))
            lower_lim_x1, upper_lim_x1 = min(lower_lim_x1, upper_lim_x1), max(
                lower_lim_x1, upper_lim_x1
            )
            lower_lim_y1, upper_lim_y1 = min(lower_lim_y1, upper_lim_y1), max(
                lower_lim_y1, upper_lim_y1
            )

            # randomly generate upper left corner coordinates and extend the
            # limits by 15 pixels
            if lower_lim_x1 < union_x1:
                lower_lim_x1 = lower_lim_x1 + 15
                upper_lim_x1 = max(0, upper_lim_x1 - 15)
                # in case lower_lim_x1 > upper_lim_x1 after the extension above
                if lower_lim_x1 > upper_lim_x1:
                    lower_lim_x1, upper_lim_x1 = upper_lim_x1, lower_lim_x1
            else:
                lower_lim_x1 = max(0, lower_lim_x1 - 15)
                upper_lim_x1 = upper_lim_x1 + 15
            if lower_lim_y1 < union_y1:
                lower_lim_y1 = lower_lim_y1 + 15
                upper_lim_y1 = max(0, upper_lim_y1 - 15)
                # in case lower_lim_y1 > upper_lim_y1 after the extension above
                if lower_lim_y1 > upper_lim_y1:
                    lower_lim_y1, upper_lim_y1 = upper_lim_y1, lower_lim_y1
            else:
                lower_lim_y1 = max(0, lower_lim_y1 - 15)
                upper_lim_y1 = upper_lim_y1 + 15
            if iter_i == self.repeat_times - 2:
                # randomly crop upper left part or lower right part if a good
                # crop can't be achieved in repeat_times-1 times
                if np.random.choice([0, 1]) == 0:
                    upper_lim_x1 = min(lower_lim_x1 + 1, img_w)
                    upper_lim_y1 = min(lower_lim_y1 + 1, img_h)
                else:
                    lower_lim_x1 = max(0, upper_lim_x1 - 1)
                    lower_lim_y1 = max(0, upper_lim_y1 - 1)

        elif do_center_crop:
            if self.center_shake is not None:
                center_shake = copy.deepcopy(list(self.center_shake))
                if "scale_factor" in data.keys():
                    scale_factor_h = data["scale_factor"][1]
                    scale_factor_w = data["scale_factor"][0]
                    center_shake[0] = int(center_shake[0] * scale_factor_h)
                    center_shake[1] = int(center_shake[1] * scale_factor_h)
                    center_shake[2] = int(center_shake[2] * scale_factor_w)
                    center_shake[3] = int(center_shake[3] * scale_factor_w)
                center_h = np.floor(margin_h / 2)
                center_w = np.floor(margin_w / 2)
                lower_lim_y1 = max(center_h - center_shake[0], 0)
                upper_lim_y1 = min(center_h + center_shake[1], margin_h)
                lower_lim_x1 = max(center_w - center_shake[2], 0)
                upper_lim_x1 = min(center_w + center_shake[3], margin_w)
            else:
                lower_lim_y1 = upper_lim_y1 = np.floor(margin_h / 2)
                lower_lim_x1 = upper_lim_x1 = np.floor(margin_w / 2)

        offset_h = np.random.randint(lower_lim_y1, upper_lim_y1 + 1)
        offset_w = np.random.randint(lower_lim_x1, upper_lim_x1 + 1)

        if self.keep_raw_pattern:
            cur_pattern = data["cur_pattern"]
            tar_pattern = data["raw_pattern"]
            if tar_pattern == cur_pattern:
                x_offset, y_offset = 0, 0
            elif (
                cur_pattern[0] == tar_pattern[2]
                and cur_pattern[1] == tar_pattern[3]
            ):
                # rggb->gbrg, bggr->grbg
                x_offset, y_offset = 1, 0
            elif (
                cur_pattern[0] == tar_pattern[1]
                and cur_pattern[2] == tar_pattern[3]
            ):
                # rggb->grbg, bggr->gbrg
                x_offset, y_offset = 0, 1
            else:  # This is not happening in ["RGGB", "BGGR", "GRBG", "GBRG"]
                raise RuntimeError(
                    "Unexpected pair of input and target bayer pattern!"
                )
            if y_offset < margin_h + 1:
                offset_h = random.randrange(y_offset, margin_h + 1, 2)
            else:
                offset_h = y_offset
            if x_offset < margin_w + 1:
                offset_w = random.randrange(x_offset, margin_w + 1, 2)
            else:
                offset_w = x_offset
        crop_y1, crop_y2 = offset_h, min(offset_h + crop_size[0], img_h)
        crop_x1, crop_x2 = offset_w, min(offset_w + crop_size[1], img_w)
        if layout == "hwc":
            data["img_shape"] = (
                int(crop_y2) - int(crop_y1),
                int(crop_x2) - int(crop_x1),
                img_shape[2],
            )
        else:
            data["img_shape"] = (
                img_shape[2],
                int(crop_y2) - int(crop_y1),
                int(crop_x2) - int(crop_x1),
            )
        return (
            offset_h,
            offset_w,
            np.array([crop_y1, crop_y2, crop_x1, crop_x2]),
        )

    def _crop_img(self, data, offset_h, offset_w, crop_bbox):
        if "limit_box" in data:
            limit_box = data["limit_box"]
            crop_bbox[2] = min(crop_bbox[2], limit_box[0])
            crop_bbox[0] = min(crop_bbox[0], limit_box[1])
            crop_bbox[3] = max(crop_bbox[3], limit_box[2])
            crop_bbox[1] = max(crop_bbox[1], limit_box[3])
        # crop the image
        if data["layout"] == "hwc":
            crop_img = data["img"][
                int(crop_bbox[0]) : int(crop_bbox[1]),
                int(crop_bbox[2]) : int(crop_bbox[3]),
                :,
            ]
        else:
            crop_img = data["img"][
                :,
                int(crop_bbox[0]) : int(crop_bbox[1]),
                int(crop_bbox[2]) : int(crop_bbox[3]),
            ]
        img_shape = crop_img.shape
        data["img"] = crop_img
        data["img_shape"] = img_shape
        data["pad_shape"] = img_shape
        data["crop_offset"] = [offset_w, offset_h, offset_w, offset_h]
        data["crop_bbox"] = crop_bbox

    def _crop_bbox(self, data, offset_h, offset_w):
        # crop bboxes and clip to the image boundary
        img_shape = data["img_shape"]
        if data["layout"] == "chw":
            img_shape = (img_shape[1], img_shape[2], img_shape[0])
        bbox_offset = np.array(
            [offset_w, offset_h, offset_w, offset_h], dtype=np.float32
        )
        if data["gt_bboxes"].any():
            bboxes = data["gt_bboxes"] - bbox_offset
            boxes_real = bboxes.copy()
            classes = data["gt_classes"].copy()
            classes_real = classes.copy()
            gt_ids_flag = "gt_ids" in data
            if gt_ids_flag:
                gt_ids = data["gt_ids"].copy()
            else:
                gt_ids = data["gt_classes"].copy()
            if self.rm_neg_coords:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
            # default setttings, ignore the tiny boxes
            area = np.maximum(0, bboxes[:, 2] - bboxes[:, 0]) * np.maximum(
                0, bboxes[:, 3] - bboxes[:, 1]
            )
            if self.filter_area:
                area_ignore_index = (area < 1e-3) & (classes >= 0)
                if self.discriminate_ignore_classes:
                    classes[area_ignore_index] = -abs(
                        classes[area_ignore_index]
                        + int(self.without_background)
                    )
                    gt_ids[area_ignore_index] = -abs(
                        gt_ids[area_ignore_index]
                        + int(self.without_background)
                    )
                else:
                    classes[area_ignore_index] = -1
                    gt_ids[area_ignore_index] = -1
            # user settings, ignore boxes whose area is less than min_area
            if self.min_area > 0:
                area_ignore_index = (area < self.min_area) & (classes >= 0)
                if self.discriminate_ignore_classes:
                    classes[area_ignore_index] = -abs(
                        classes[area_ignore_index]
                        + int(self.without_background)
                    )
                    gt_ids[area_ignore_index] = -abs(
                        gt_ids[area_ignore_index]
                        + int(self.without_background)
                    )
                else:
                    classes[area_ignore_index] = -1
                    gt_ids[area_ignore_index] = -1
            # ignore the regions where the iou between before and after
            # truncation < min_iou
            if self.min_iou > 0:
                if self.discard_truncate:
                    discard_ids = [True] * boxes_real.shape[0]
                for i in range(boxes_real.shape[0]):
                    iou = bbox_overlaps(
                        bboxes[i].reshape((1, -1)),
                        boxes_real[i].reshape((1, -1)),
                    )[0][0]
                    if iou < self.min_iou and classes[i] >= 0:
                        if self.discard_truncate:
                            discard_ids[i] = False
                        elif self.discriminate_ignore_classes:
                            classes[i] = -1 * (
                                abs(classes[i]) + int(self.without_background)
                            )
                            gt_ids[i] = -1 * (
                                abs(gt_ids[i]) + int(self.without_background)
                            )
                        else:
                            classes[i] = -1
                            gt_ids[i] = -1
                if self.discard_truncate:
                    classes = classes[discard_ids]
                    gt_ids = gt_ids[discard_ids]
                    bboxes = bboxes[discard_ids]
                    boxes_real = boxes_real[discard_ids]
                classes_real = classes.copy()
                gt_ids_real = gt_ids.copy()

            # filter out the gt bboxes that are completely cropped
            if self.rm_neg_coords:
                valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                    bboxes[:, 3] > bboxes[:, 1]
                )
            else:
                cropped_boxes = bboxes.reshape(-1, 2, 2)
                max_size = np.array([img_shape[1], img_shape[0]])
                cropped_boxes = np.minimum(
                    cropped_boxes.reshape(-1, 2, 2), max_size
                )
                cropped_boxes = np.clip(cropped_boxes, 0, 1000000)
                valid_inds = np.all(
                    cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], axis=1
                )
            bboxes = bboxes[valid_inds]
            classes = classes[valid_inds]
            gt_ids = gt_ids[valid_inds]
            # if no gt bbox remains after cropping, set bboxes shape (0, 4)
            # keep dtype consistency with origin bboxes and classes
            if not np.any(valid_inds):
                boxes_real = bboxes = np.zeros(
                    (0, 4), dtype=data["gt_bboxes"].dtype
                )
                classes_real = classes = np.zeros(
                    (0,), dtype=data["gt_classes"].dtype
                )
                if gt_ids_flag:
                    gt_ids_real = gt_ids = np.zeros(
                        (0,), dtype=data["gt_ids"].dtype
                    )
                else:
                    gt_ids_real = gt_ids = np.zeros(
                        (0,), dtype=data["gt_classes"].dtype
                    )
                if "gt_tanalphas" in data:
                    data["gt_tanalphas"] = np.zeros((0,), dtype=np.float32)
            data["gt_bboxes"] = bboxes if self.truncate_gt else boxes_real
            data["gt_classes"] = classes if self.truncate_gt else classes_real
            if gt_ids_flag:
                data["gt_ids"] = gt_ids if self.truncate_gt else gt_ids_real

    def _crop_seg(self, data, crop_bbox):
        gt_seg = data["gt_seg"]
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        crop_gt_seg = gt_seg[
            int(crop_y1) : int(crop_y2), int(crop_x1) : int(crop_x2)
        ]
        data["gt_seg"] = crop_gt_seg

    def _crop_flow(self, data, crop_bbox):
        gt_flow = data["gt_flow"]
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        crop_gt_flow = gt_flow[
            int(crop_y1) : int(crop_y2), int(crop_x1) : int(crop_x2)
        ]
        data["gt_flow"] = crop_gt_flow

    def _crop_disp(self, data, crop_bbox):
        gt_disp = data["gt_disp"]
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        crop_gt_disp = gt_disp[
            int(crop_y1) : int(crop_y2), int(crop_x1) : int(crop_x2)
        ]
        data["gt_disp"] = crop_gt_disp

    def _crop_lines(self, data, offset_h, offset_w):
        gt_lines = data["gt_lines"]
        for gt_line in gt_lines:
            gt_line[:, 0] = gt_line[:, 0] - offset_w
            gt_line[:, 1] = gt_line[:, 1] - offset_h

        data["gt_lines"] = gt_lines

    def inverse_transform(
        self, inputs: torch.Tensor, task_type: str, inverse_info: Dict
    ):
        """Inverse option of transform to map the prediction to the original image.

        Args:
            inputs (array): Prediction
            task_type (str): `detection` or `segmentation`.
            inverse_info (dict): The transform keyword is the key,
                and the corresponding value is the value.

        """
        if task_type == "detection":
            crop_offset = inverse_info["crop_offset"]
            if not isinstance(crop_offset, torch.Tensor):
                crop_offset = inputs.new_tensor(crop_offset)
            crop_offset = crop_offset.reshape((1, 4))
            inputs[:, :4] = inputs[:, :4] + crop_offset
            return inputs
        elif task_type == "segmentation":
            before_crop_shape = inverse_info["before_crop_shape"][1:]
            crop_offset_x, crop_offset_y = inverse_info["crop_offset"][:2]
            crop_h, crop_w = inputs.shape
            out_img = np.full(before_crop_shape, 255)
            out_img[
                crop_offset_y : crop_offset_y + crop_h,
                crop_offset_x : crop_offset_x + crop_w,
            ] = inputs.cpu().numpy()
            return out_img
        else:
            raise Exception(
                "error task_type, your task_type[{}],"
                " we need segmentation or detection".format(task_type)
            )

    def __call__(self, data):
        data["crop_offset"] = [0, 0, 0, 0]
        data["pad_shape"] = data["img"].shape
        data_without_img = copy.deepcopy(data)
        image = data_without_img.pop("img")
        if self.crop_around_gt:
            assert "gt_bboxes" in data
        for i in range(self.repeat_times):
            data_this = copy.deepcopy(data_without_img)
            offset_h, offset_w, crop_bbox = self._get_crop_region(data_this, i)
            if "gt_bboxes" in data and self._is_good_enough(data, crop_bbox):
                break
        data = data_this
        data["img"] = image
        self._crop_img(data, offset_h, offset_w, crop_bbox)
        if "gt_bboxes" in data:
            self._crop_bbox(data_this, offset_h, offset_w)
        if "gt_seg" in data:
            self._crop_seg(data, crop_bbox)
        elif "gt_flow" in data:
            self._crop_flow(data, crop_bbox)
        elif "gt_disp" in data:
            self._crop_disp(data, crop_bbox)
        elif "gt_lines" in data:
            self._crop_lines(data, offset_h, offset_w)
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + ": "
        repr_str += f"size={self.size}, "
        repr_str += f"min_area={self.min_area}, "
        repr_str += f"min_iou={self.min_iou}"
        repr_str += f"keep_raw_pattern={self.keep_raw_pattern}"
        return repr_str


@OBJECT_REGISTRY.register
class FixedCrop(RandomCrop):
    """Crop image with fixed position and size.

    .. note::
        Affected keys: 'img', 'img_shape', 'pad_shape', 'layout',
        'before_crop_shape', 'crop_offset', 'gt_bboxes', 'gt_classes'.

    """

    def __init__(
        self,
        size: Tuple[int] = None,
        min_area: int = -1,
        min_iou: int = -1,
        dynamic_roi_params: Dict = None,
        discriminate_ignore_classes: Optional[bool] = False,
        allow_smaller: bool = False,
    ):  # noqa: D205,D400
        """

        Args:
            size (Tuple): Expected size after cropping, (w, h) or
                (x1, y1, w, h).
            min_area (Optional[int]): If min_area > 0, boxes whose areas are
                less than min_area will be ignored.
            min_iou (Optional[float]): If min_iou > 0, boxes whose iou between
                before and after truncation < min_iou will be ignored.
            dynamic_roi_param (Dict): Dynamic ROI parameters contains keys
                {'w', 'h', 'fp_x', 'fp_y'}. NOTE:
                ``get_dynamic_roi_from_camera`` is not migrated; passing a
                non-None value will raise.
            discriminate_ignore_classes (Optional[bool]): if True, ignored area
                by min_iou retain the original class info. And class id should
                greater than 0.
                Default False. Support class id starts from 0.
            allow_smaller: Weather to allow the image shape smaller
                than roi after resize.
        """
        if not dynamic_roi_params:
            assert size is not None
            if len(size) == 2:
                size = (0, 0, size[0], size[1])
            else:
                assert len(size) == 4
        self.size = size
        self.min_area = min_area
        self.min_iou = min_iou
        self.dynamic_roi_params = dynamic_roi_params
        self.discriminate_ignore_classes = discriminate_ignore_classes
        self.allow_smaller = allow_smaller

    def _crop_img(self, data):
        img = data["img"]
        if self.dynamic_roi_params:
            raise NotImplementedError(
                "get_dynamic_roi_from_camera is not migrated to HEAL. "
                "Pass dynamic_roi_params=None to use the fixed-size crop "
                "branch."
            )
        else:
            x1, y1, w, h = self.size
            x2, y2 = x1 + w, y1 + h
            assert w > 0 and h > 0

        if not self.allow_smaller:
            assert x2 <= img.shape[1] and y2 <= img.shape[0]
        offset_w, offset_h = x1, y1

        # crop the image
        crop_img = img[y1:y2, x1:x2, :]
        data["img"] = crop_img
        data["img_shape"] = crop_img.shape
        data["before_crop_shape"] = img.shape
        data["crop_offset"] = [x1, y1, x1, y1]
        return offset_h, offset_w

    def _crop_bbox(self, data, offset_h, offset_w):
        # crop bboxes and clip to the image boundary
        img_shape = data["img_shape"]
        bbox_offset = np.array(
            [offset_w, offset_h, offset_w, offset_h], dtype=np.float32
        )
        if data["gt_bboxes"].any():
            bboxes = data["gt_bboxes"] - bbox_offset
            boxes_real = bboxes.copy()
            classes = data["gt_classes"].copy()
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
            # default setttings, ignore the tiny boxes
            area = np.maximum(0, bboxes[:, 2] - bboxes[:, 0]) * np.maximum(
                0, bboxes[:, 3] - bboxes[:, 1]
            )
            area_ignore_index = (area < 1e-3) & (classes >= 0)
            if self.discriminate_ignore_classes:
                classes[area_ignore_index] = -abs(classes[area_ignore_index])
            else:
                classes[area_ignore_index] = -1
            # user settings, ignore boxes whose area is less than min_area
            if self.min_area > 0:
                area_ignore_index = (area < self.min_area) & (classes >= 0)
                if self.discriminate_ignore_classes:
                    classes[area_ignore_index] = -abs(
                        classes[area_ignore_index]
                    )
                else:
                    classes[area_ignore_index] = -1
            # ignore the regions where the iou between before and after
            # truncation < min_iou
            if self.min_iou > 0:
                for i in range(boxes_real.shape[0]):
                    iou = bbox_overlaps(
                        bboxes[i].reshape((1, -1)),
                        boxes_real[i].reshape((1, -1)),
                    )[0][0]
                    if iou < self.min_iou and classes[i] >= 0:
                        if self.discriminate_ignore_classes:
                            classes[i] = -1 * abs(classes[i])
                        else:
                            classes[i] = -1

            # filter out the gt bboxes that are completely cropped
            valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (
                bboxes[:, 3] > bboxes[:, 1]
            )
            # if no gt bbox remains after cropping, set bboxes shape (0, 4)
            if not np.any(valid_inds):
                bboxes = np.zeros((0, 4), dtype=np.float32)
                classes = np.zeros((0,), dtype=np.int64)
            data["gt_bboxes"] = bboxes
            data["gt_classes"] = classes

    def _crop_lines(self, data, offset_h, offset_w):
        gt_lines = data["gt_lines"]
        for gt_line in gt_lines:
            gt_line[:, 0] = gt_line[:, 0] - offset_w
            gt_line[:, 1] = gt_line[:, 1] - offset_h

        data["gt_lines"] = gt_lines

    def inverse_transform(
        self, inputs: torch.Tensor, task_type: str, inverse_info: Dict
    ):
        """Inverse option of transform to map the prediction to the original image.

        Args:
            inputs (array): Prediction
            task_type (str): `detection` or `segmentation`.
            inverse_info (dict): The transform keyword is the key,
                and the corresponding value is the value.

        """
        if task_type == "detection":
            crop_offset = inverse_info["crop_offset"]
            if not isinstance(crop_offset, torch.Tensor):
                crop_offset = inputs.new_tensor(crop_offset)
            crop_offset = crop_offset.reshape((1, 4))
            inputs[:, :4] = inputs[:, :4] + crop_offset
            return inputs
        elif task_type == "segmentation":
            crop_offset_x, crop_offset_y = inverse_info["crop_offset"][:2]
            crop_h, crop_w = inputs.shape
            before_crop_shape = (
                inverse_info["before_crop_shape"][:2]
                if inverse_info["before_crop_shape"][0] > 3
                else inverse_info["before_crop_shape"][1:]
            )
            if before_crop_shape:
                out_img = np.full(before_crop_shape, 255)
                out_img[
                    crop_offset_y : crop_offset_y + crop_h,
                    crop_offset_x : crop_offset_x + crop_w,
                ] = inputs.cpu().numpy()
            else:
                out_img = inputs.cpu().numpy()
            return out_img
        elif task_type == "cone_invasion":
            return inputs
        else:
            raise Exception(
                "error task_type, your task_type[{}],"
                " we need segmentation or detection".format(task_type)
            )

    def __call__(self, data):
        offset_h, offset_w = self._crop_img(data)
        if "gt_bboxes" in data:
            self._crop_bbox(data, offset_h, offset_w)
        if "gt_lines" in data:
            self._crop_lines(data, offset_h, offset_w)
        return data


@OBJECT_REGISTRY.register
class RandomSizeCrop(RandomCrop):
    """Random crop with size in [min_size, max_size]."""

    def __init__(
        self,
        min_size: int,
        max_size: int,
        **kwargs,
    ):
        super(RandomSizeCrop, self).__init__(
            size=(min_size, max_size),
            **kwargs,
        )
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, data):
        if data["layout"] == "hwc":
            img_h, img_w = data["img_shape"][:2]
        else:
            img_h, img_w = data["img_shape"][1:]
        assert img_h > self.min_size and img_w > self.min_size, (
            f"img shape should be larger than {self.min_size},"
            "but got {img_h}x{img_w}"
        )
        w = random.randint(self.min_size, min(img_w, self.max_size))
        h = random.randint(self.min_size, min(img_h, self.max_size))
        self.size = (h, w)
        offset_h, offset_w, crop_bbox = self._get_crop_region(data)
        self._crop_img(data, offset_h, offset_w, crop_bbox)
        if "gt_bboxes" in data:
            self._crop_bbox(data, offset_h, offset_w)
        elif "gt_seg" in data:
            self._crop_seg(data, crop_bbox)
        elif "gt_flow" in data:
            self._crop_flow(data, crop_bbox)
        return data


@OBJECT_REGISTRY.register
class Pad(object):
    """Pad image to size. Affected keys: img, pad_shape."""

    def __init__(
        self,
        size: Tuple = None,
        divisor: int = 1,
        pad_val: int = 0,
        seg_pad_val: int = 255,
    ):
        self.size = size
        self.divisor = divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

    def _pad_img(self, data):
        layout = data.get("layout", "hwc")
        if layout == "chw":
            data["before_pad_shape"] = np.array(data["img"].shape[1:])
        else:
            data["before_pad_shape"] = np.array(data["img"].shape[:2])
        padded_img = image_pad(
            data["img"], layout, self.size, self.divisor, self.pad_val
        )
        data["img"] = padded_img
        data["pad_shape"] = padded_img.shape

    def __call__(self, data):
        if "layout" not in data:
            data["layout"] = "hwc"
        self._pad_img(data)
        if "gt_seg" in data:
            if data["layout"] == "chw":
                size = data["pad_shape"][1:]
            else:
                size = data["pad_shape"][:2]
            data["gt_seg"] = image_pad(
                data["gt_seg"], "hw", size, 1, self.seg_pad_val
            )
        return data


@OBJECT_REGISTRY.register
class Normalize(object):
    """Normalize image. Affected keys: img."""

    def __init__(
        self,
        mean: Union[float, Sequence[float]],
        std: Union[float, Sequence[float]],
        raw_norm: bool = False,
        split_transform: bool = False,
    ):
        self.mean = mean
        self.std = std
        self.raw_norm = raw_norm
        self.split_transform = split_transform

    def __call__(self, data):
        layout = data.get("layout", "chw")
        if self.raw_norm:
            bit_nums_lower = data["bit_nums_lower"][0]
            in_img = data["img"] / (2**bit_nums_lower)
        else:
            in_img = data["img"]
        data["img"] = image_normalize(in_img, self.mean, self.std, layout)
        if self.split_transform and "me_in_img" in data:
            if self.raw_norm:
                me_in_img = data["me_in_img"] / (
                    2 ** data["bit_nums_lower"][0]
                )
            else:
                me_in_img = data["me_in_img"]
            data["me_in_img"] = image_normalize(
                me_in_img, self.mean, self.std, layout
            )
            data["me_in_img"] = np.transpose(data["me_in_img"], (2, 0, 1))
        return data


@OBJECT_REGISTRY.register
class ToTensor(object):
    """Convert to torch.Tensor; optionally convert img to YUV444."""

    def __init__(
        self,
        to_yuv: bool = False,
        use_yuv_v2: bool = True,
        split_transform: bool = False,
    ):
        self.to_yuv = to_yuv
        self.use_yuv_v2 = use_yuv_v2
        self.split_transform = split_transform

    @staticmethod
    def _convert_layout(img, layout: str):
        assert layout in ["hwc", "chw"]
        if layout == "chw":
            return img, layout
        if isinstance(img, torch.Tensor):
            img = img.permute((2, 0, 1))
        elif isinstance(img, np.ndarray):
            img = np.ascontiguousarray(img.transpose((2, 0, 1)))
        else:
            raise TypeError
        return img, "chw"

    @staticmethod
    def _to_tensor(data):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data.copy())
        elif isinstance(data, (list, tuple)) and not isinstance(data, str):
            return torch.tensor(data)
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        else:
            raise TypeError(
                f"type {type(data)} cannot be converted to tensor."
            )

    @staticmethod
    def _convert_hw_to_even(data):
        """Ensure H,W even for BgrToYuv444."""
        if data["layout"] == "hwc":
            h, w, c = data["img"].shape
        else:
            c, h, w = data["img"].shape
        crop_h = h if (h % 2) == 0 else h - 1
        crop_w = w if (w % 2) == 0 else w - 1
        if data["layout"] == "hwc":
            data["img"] = data["img"][:crop_h, :crop_w, :]
        else:
            data["img"] = data["img"][:, :crop_h, :crop_w]
        for key in ("gt_seg", "gt_seg_weights", "gt_flow", "gt_depth"):
            if key in data:
                data[key] = data[key][:crop_h, :crop_w]
        data["img_shape"] = np.array(data["img"].shape)
        data["pad_shape"] = np.array(data["img"].shape)
        return data

    def __call__(self, data):
        layout = data.get("layout", "hwc")
        data["img"], data["layout"] = self._convert_layout(data["img"], layout)
        data["img_shape"] = np.array(data["img"].shape)
        data["pad_shape"] = np.array(data["img"].shape)
        data["img"] = self._to_tensor(data["img"])
        if self.split_transform and "me_in_img" in data:
            data["me_in_img"] = self._to_tensor(data["me_in_img"])
        for key in (
            "gt_bboxes",
            "ig_bboxes",
            "gt_tanalphas",
            "gt_classes",
            "gt_labels",
            "gt_seg",
            "gt_seg_weights",
            "gt_flow",
            "gt_depth",
            "gt_ids",
        ):
            if key in data:
                if key == "gt_flow":
                    data[key], _ = self._convert_layout(data[key], layout)
                data[key] = self._to_tensor(data[key])

        if self.to_yuv:
            data = self._convert_hw_to_even(data)
            from hat_engine.data.transforms.common import (
                BgrToYuv444,
                BgrToYuv444V2,
            )

            color_space = data.get("color_space", "bgr")
            rgb_input = color_space.lower() == "rgb"
            if self.use_yuv_v2:
                yuv_converter = BgrToYuv444V2(rgb_input=rgb_input)
            else:
                yuv_converter = BgrToYuv444(rgb_input=rgb_input)
            data["img"] = yuv_converter(data["img"])
            data["color_space"] = "yuv"
        return data


@OBJECT_REGISTRY.register
class AugmentHSV(object):
    """Random HSV augmentation. Affected keys: img."""

    def __init__(
        self,
        hgain: float = 0.5,
        sgain: float = 0.5,
        vgain: float = 0.5,
        p: float = 1.0,
    ):
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
        self.p = p

    def __call__(self, data):
        if np.random.choice([False, True], p=[1 - self.p, self.p]):
            img = data["img"]
            if data.get("layout") == "chw":
                img = np.transpose(img, (1, 2, 0))
            r = (
                np.random.uniform(-1, 1, 3)
                * [self.hgain, self.sgain, self.vgain]
                + 1
            )
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
            dtype = img.dtype
            x = np.arange(0, 256, dtype=np.int16)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
            img_hsv = cv2.merge(
                (
                    cv2.LUT(hue, lut_hue),
                    cv2.LUT(sat, lut_sat),
                    cv2.LUT(val, lut_val),
                )
            ).astype(dtype)
            img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
            if data.get("layout") == "chw":
                img = np.transpose(img, (2, 0, 1))
            data["img"] = img
        return data


@OBJECT_REGISTRY.register
class Mosaic(object):
    """Mosaic augmentation for detection.

    ``__call__(data_list)`` returns a single data dict.
    """

    def __init__(
        self,
        image_size: int = 512,
        degrees: int = 10,
        translate: float = 0.1,
        scale: float = 0.1,
        shear: int = 10,
        perspective: float = 0.0,
        mixup: bool = True,
    ):
        self.image_size = image_size
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = [-image_size // 2, -image_size // 2]
        self.mixup = mixup

    def _fix_labels(self, labels, offsets):
        new_labels = labels.copy()
        new_labels[:, 0] = labels[:, 0] + offsets[1]
        new_labels[:, 1] = labels[:, 1] + offsets[0]
        new_labels[:, 2] = labels[:, 2] + offsets[1]
        new_labels[:, 3] = labels[:, 3] + offsets[0]
        return new_labels

    def _mixup(self, data1, data2):
        img1 = data1["img"]
        bboxes1 = data1["gt_bboxes"]
        labels1 = data1["gt_classes"]
        img2 = data2["img"]
        bboxes2 = data2["gt_bboxes"]
        labels2 = data2["gt_classes"]
        img2, bboxes2, _ = self._random_scale(img2, bboxes2)
        img2, bboxes2, labels2 = self._crop(img2, bboxes2, labels2)
        img2 = self._pad(img2)
        r = 0.5
        img1 = (img1 * r + img2 * (1 - r)).astype(np.uint8)
        data1["img"] = img1
        data1["gt_bboxes"] = np.concatenate((bboxes1, bboxes2), 0)
        data1["gt_classes"] = np.concatenate((labels1, labels2), 0)
        return data1

    def _random_scale(self, img, bboxes):
        min_ratio, max_ratio = 1 - self.scale, 1 + self.scale
        scale = random.uniform(min_ratio, max_ratio)
        max_side = self.image_size * scale
        h, w, _ = img.shape
        largest_side = max(h, w)
        scale = max_side / largest_side
        img = cv2.resize(
            img,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_LINEAR,
        )
        bboxes = bboxes * scale
        return img, bboxes, scale

    def _crop(self, img, bboxes, labels):
        margin_h = max(img.shape[0] - self.image_size, 0)
        margin_w = max(img.shape[1] - self.image_size, 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.image_size
        crop_x1, crop_x2 = offset_w, offset_w + self.image_size
        crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2, :]
        bbox_offset = np.array(
            [offset_w, offset_h, offset_w, offset_h], dtype=np.float32
        )
        bboxes = bboxes.copy()
        bboxes[:, :4] -= bbox_offset
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]].clip(0, crop_img.shape[1] - 1)
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]].clip(0, crop_img.shape[0] - 1)
        valid_ind = (bboxes[:, 2] > bboxes[:, 0]) & (
            bboxes[:, 3] > bboxes[:, 1]
        )
        return crop_img, bboxes[valid_ind], labels[valid_ind]

    def _pad(self, img):
        h, w, c = img.shape
        new_image = np.zeros(
            (self.image_size, self.image_size, c), dtype=np.uint8
        )
        new_image[:h, :w, :] = img.astype(np.uint8)
        return new_image

    def _box_candidates(self, box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))
        return (
            (w2 > wh_thr)
            & (h2 > wh_thr)
            & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr)
            & (ar < ar_thr)
        )

    def _random_rerspective(self, data):
        image = data["img"]
        bboxes = data["gt_bboxes"]
        labels = data["gt_classes"]
        height = image.shape[0] + self.border[0] * 2
        width = image.shape[1] + self.border[1] * 2
        C = np.eye(3)
        C[0, 2] = -image.shape[1] / 2
        C[1, 2] = -image.shape[0] / 2
        P = np.eye(3)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)
        R = np.eye(3)
        a = random.uniform(-self.degrees, self.degrees)
        s = random.uniform(1.0, 1 + self.scale)
        direction = random.uniform(0, 1)
        s = s if direction > 0.5 else 1 / s
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
        S = np.eye(3)
        S[0, 1] = math.tan(
            random.uniform(-self.shear, self.shear) * math.pi / 180
        )
        S[1, 0] = math.tan(
            random.uniform(-self.shear, self.shear) * math.pi / 180
        )
        T = np.eye(3)
        T[0, 2] = (
            random.uniform(0.5 - self.translate, 0.5 + self.translate) * width
        )
        T[1, 2] = (
            random.uniform(0.5 - self.translate, 0.5 + self.translate) * height
        )
        M = T @ S @ R @ P @ C
        if (
            (self.border[0] != 0)
            or (self.border[1] != 0)
            or (M != np.eye(3)).any()
        ):
            if self.perspective:
                image = cv2.warpPerspective(
                    image,
                    M,
                    dsize=(width, height),
                    borderValue=(114, 114, 114),
                )
            else:
                image = cv2.warpAffine(
                    image,
                    M[:2],
                    dsize=(width, height),
                    borderValue=(114, 114, 114),
                )
        n = len(bboxes)
        if n:
            xy = np.ones((n * 4, 3))
            xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)
            xy = xy @ M.T
            if self.perspective:
                xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)
            else:
                xy = xy[:, :2].reshape(n, 8)
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = (
                np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)))
                .reshape(4, n)
                .T
            )
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width - 1)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height - 1)
            i = self._box_candidates(box1=bboxes[:, :4].T * s, box2=xy.T)
            bboxes = bboxes[i].copy()
            bboxes[:, :4] = xy[i]
            data["gt_bboxes"] = bboxes
            data["gt_classes"] = labels[i]
        data["img"] = image
        data["scale_factor"] = np.asarray([s])

    @property
    def images_needed(self):
        return 4 if not self.mixup else 5

    def __call__(self, data_list):
        _, _, c = data_list[0]["img"].shape
        s = self.image_size
        img4 = np.full((s * 2, s * 2, c), 114, dtype=np.uint8)
        data1 = copy.deepcopy(data_list[0])
        image = data1["img"]
        oh, ow, _ = image.shape
        bboxes = data1["gt_bboxes"].copy()
        labels = data1["gt_classes"].copy()
        h, w = min(oh, s), min(ow, s)
        img4[s - h : s, s - w : s, :] = image[oh - h : oh, ow - w : ow, :]
        new_bboxes = self._fix_labels(bboxes, ((s - oh), (s - ow)))
        new_labels = labels

        data2 = copy.deepcopy(data_list[1])
        image = data2["img"]
        oh, ow, _ = image.shape
        bboxes = data2["gt_bboxes"].copy()
        labels = data2["gt_classes"].copy()
        h, w = min(oh, s), min(ow, s)
        img4[s - h : s, s : s + w, :] = image[oh - h : oh, :w, :]
        bboxes = self._fix_labels(bboxes, ((s - oh), s))
        new_bboxes = np.concatenate((new_bboxes, bboxes), axis=0)
        new_labels = np.concatenate((new_labels, labels), axis=0)

        data3 = copy.deepcopy(data_list[2])
        image = data3["img"]
        oh, ow, _ = image.shape
        bboxes = data3["gt_bboxes"].copy()
        labels = data3["gt_classes"].copy()
        h, w = min(oh, s), min(ow, s)
        img4[s : s + h, s - w : s, :] = image[:h, ow - w : ow, :]
        bboxes = self._fix_labels(bboxes, (s, s - ow))
        new_bboxes = np.concatenate((new_bboxes, bboxes), axis=0)
        new_labels = np.concatenate((new_labels, labels), axis=0)

        data4 = copy.deepcopy(data_list[3])
        image = data4["img"]
        oh, ow, _ = image.shape
        bboxes = data4["gt_bboxes"].copy()
        labels = data4["gt_classes"].copy()
        h, w = min(oh, s), min(ow, s)
        img4[s : s + h, s : s + w, :] = image[:h, :w, :]
        bboxes = self._fix_labels(bboxes, (s, s))
        new_bboxes = np.concatenate((new_bboxes, bboxes), axis=0)
        new_labels = np.concatenate((new_labels, labels), axis=0)

        new_bboxes[:, 0:4] = np.clip(new_bboxes[:, 0:4], 0, (2 * s - 1))
        data1["img"] = img4
        data1["gt_bboxes"] = new_bboxes
        data1["gt_classes"] = new_labels
        self._random_rerspective(data1)
        if self.mixup:
            data5 = copy.deepcopy(data_list[4])
            data1 = self._mixup(data1, data5)
        data1["img_shape"] = data1["img"].shape
        data1["pad_shape"] = data1["img"].shape
        if "layout" not in data1:
            data1["layout"] = "hwc"
        return data1


@OBJECT_REGISTRY.register
class Resize3D(Resize):
    """Resize images and 3D-specific fields (centers2d, cam2img).

    Extends :class:`Resize` to also scale ``centers2d`` and ``cam2img``
    after the standard 2D resize.

    Args:
        cam2img_keep_ratio: If True, scale both fx and fy by
            ``scale_factor[0]``; otherwise use the per-axis scale factors.
    """

    def __init__(self, cam2img_keep_ratio=True, **kwargs):
        super().__init__(**kwargs)
        self.cam2img_keep_ratio = cam2img_keep_ratio

    def _resize_3d(self, results):
        if "centers2d" in results and results["centers2d"].size(0) > 0:
            results["centers2d"] *= results["scale_factor"][:2]

        if "cam2img" in results:
            results["cam2img"][0] *= results["scale_factor"][0].repeat(
                len(results["cam2img"][0])
            )
            if self.cam2img_keep_ratio:
                results["cam2img"][1] *= results["scale_factor"][0].repeat(
                    len(results["cam2img"][1])
                )
            else:
                results["cam2img"][1] *= results["scale_factor"][1].repeat(
                    len(results["cam2img"][1])
                )

    def __call__(self, results):
        results = super().__call__(results)
        self._resize_3d(results)
        return results
