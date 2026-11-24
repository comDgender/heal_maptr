# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
#
# Sequence versions of detection transforms. Each Seq* transform applies
# the same random params (flip / scale / crop / color jitter) to every
# frame in a frame_data_list so that frame-to-frame consistency is
# preserved for tracking / video tasks.
#
# Only the Seq* classes used by motr_efficientnetb3 config are
# migrated: SeqRandomFlip, SeqResize, SeqToTensor, SeqBgrToYuv444,
# SeqNormalize, SeqRandomSizeCrop. Other Seq classes in the source file
# (SeqPad, SeqAugmentHSV, SeqToFasterRCNNData, SeqAlbuImageOnlyTransform)
# are not used by any HEAL config and are skipped per
# model-migration_v2 reference-import.md §2.

import copy
import random

import cv2  # noqa: F401  (kept for SeqAugmentHSV-style future use)
import numpy as np
from hat_engine.data.transforms.common import BgrToYuv444
from hat_engine.registry import OBJECT_REGISTRY

from heal.data.transforms.detection import (
    Normalize,
    RandomFlip,
    RandomSizeCrop,
    Resize,
    ToTensor,
)

__all__ = [
    "SeqRandomFlip",
    "SeqResize",
    "SeqBgrToYuv444",
    "SeqToTensor",
    "SeqNormalize",
    "SeqRandomSizeCrop",
]


@OBJECT_REGISTRY.register
class SeqRandomFlip(RandomFlip):
    """Flip image & bbox & mask & seg & flow for sequence."""

    def _gen_prob(self, px, py):
        self.flip_x = np.random.choice([False, True], p=[1 - px, px])
        self.flip_y = np.random.choice([False, True], p=[1 - py, py])

    def _flip_img(self, data):
        # Apply the same (flip_x, flip_y) determined by _gen_prob to every
        # frame in the sequence. We do NOT call random_flip here because
        # that function re-randomizes per call, breaking sequence
        # consistency.
        img = data["img"]
        layout = data["layout"]
        w_index = layout.index("w")
        h_index = layout.index("h")
        if isinstance(img, np.ndarray):
            if self.flip_x:
                img = np.flip(img, axis=w_index)
            if self.flip_y:
                img = np.flip(img, axis=h_index)
        else:
            if self.flip_x:
                img = img.flip(w_index)
            if self.flip_y:
                img = img.flip(h_index)
        data["img"] = img

    def _pre_single_data(self, data):
        self._flip_img(data)
        if "gt_bboxes" in data:
            self._flip_bbox(data)
        return data

    def __call__(self, data_seq):
        self._gen_prob(self.px, self.py)
        frame_data_list = [
            self._pre_single_data(data) for data in data_seq["frame_data_list"]
        ]
        return {"frame_data_list": frame_data_list}


@OBJECT_REGISTRY.register
class SeqResize(Resize):
    def _gen_scale(self, data):
        self._random_scale(data)
        return data["scale"], data["scale_idx"]

    def _pre_single_data(self, data, scale, scale_idx):
        data["scale"] = scale
        data["scale_idx"] = scale_idx
        self._resize_img(data)
        if "gt_bboxes" in data:
            self._resize_bbox(data)
        return data

    def __call__(self, data_seq):
        scale, scale_idx = self._gen_scale(data_seq["frame_data_list"][0])
        frame_data_list = [
            self._pre_single_data(data, scale, scale_idx)
            for data in data_seq["frame_data_list"]
        ]
        return {"frame_data_list": frame_data_list}


@OBJECT_REGISTRY.register
class SeqBgrToYuv444(BgrToYuv444):
    """BgrToYuv444 for sequence."""

    def _pre_single_data(self, data):
        return super().__call__(data)

    def __call__(self, data_seq):
        frame_data_list = [
            self._pre_single_data(data) for data in data_seq["frame_data_list"]
        ]
        seq_data = {
            "frame_data_list": frame_data_list,
            "frame_length": len(frame_data_list),
        }
        return seq_data


@OBJECT_REGISTRY.register
class SeqToTensor(ToTensor):
    """ToTensor for sequence."""

    def _pre_single_data(self, data):
        return super().__call__(data)

    def __call__(self, data_seq):
        frame_data_list = [
            self._pre_single_data(data) for data in data_seq["frame_data_list"]
        ]
        seq_data = {
            "frame_data_list": frame_data_list,
            "frame_length": len(frame_data_list),
        }
        return seq_data


@OBJECT_REGISTRY.register
class SeqNormalize(Normalize):
    """Normalize for sequence."""

    def _pre_single_data(self, data):
        return super().__call__(data)

    def __call__(self, data_seq):
        frame_data_list = [
            self._pre_single_data(data) for data in data_seq["frame_data_list"]
        ]
        seq_data = {
            "frame_data_list": frame_data_list,
            "frame_length": len(frame_data_list),
        }
        return seq_data


@OBJECT_REGISTRY.register
class SeqRandomSizeCrop(RandomSizeCrop):
    """RandomSizeCrop for sequence."""

    def _get_crop(self, frame_data_list):

        img_we = None
        img_he = None

        for data in frame_data_list:
            if data["layout"] == "hwc":
                img_h, img_w = data["img_shape"][:2]
            else:
                img_h, img_w = data["img_shape"][1:]
            assert img_h > self.min_size and img_w > self.min_size, (
                f"img shape should be larger than {self.min_size},"
                "but got {img_h}x{img_w}"
            )

            if img_he is None:
                img_he = img_h
            else:
                img_he = min(img_he, img_h)

            if img_we is None:
                img_we = img_w
            else:
                img_we = min(img_we, img_w)

        w = random.randint(self.min_size, min(img_we, self.max_size))
        h = random.randint(self.min_size, min(img_he, self.max_size))

        self.size = (h, w)
        offset_h, offset_w, crop_bbox = self._get_crop_region(data)

        return offset_h, offset_w, crop_bbox

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
        rratio = 1
        crop_size = [s * rratio for s in crop_size]
        crop_size[0] *= 1
        crop_size[1] *= 1

        margin_h = max(img_shape[0] - crop_size[0], 1)
        margin_w = max(img_shape[1] - crop_size[1], 1)
        lower_lim_x1, upper_lim_x1 = 0, margin_w
        lower_lim_y1, upper_lim_y1 = 0, margin_h

        if do_center_crop:
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

    def _pre_single_data(self, data, offset_h, offset_w, crop_bbox):
        self._crop_img(data, offset_h, offset_w, crop_bbox)
        if "gt_bboxes" in data:
            self._crop_bbox(data, offset_h, offset_w)
        elif "gt_seg" in data:
            self._crop_seg(data, crop_bbox)
        elif "gt_flow" in data:
            self._crop_flow(data, crop_bbox)
        return data

    def __call__(self, data_seq):

        offset_h, offset_w, crop_bbox = self._get_crop(
            data_seq["frame_data_list"]
        )

        frame_data_list = [
            self._pre_single_data(data, offset_h, offset_w, crop_bbox)
            for data in data_seq["frame_data_list"]
        ]
        seq_data = {
            "frame_data_list": frame_data_list,
            "frame_length": len(frame_data_list),
        }
        return seq_data
