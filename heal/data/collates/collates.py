# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
"""Collate helpers for 2D detection, segmentation, and LiDAR 3D tasks."""

import collections
import logging
import re
from typing import Any, Dict, List, Mapping, Union

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

from heal.data.metas import img_metas

try:
    from torch._six import string_classes
except ImportError:
    string_classes = (str, bytes)

np_str_obj_array_pattern = re.compile(r"[SaUO]")

__all__ = [
    "default_collate_v2",
    "collate_2d",
    "collate_2d_pad",
    "collate_lidar3d",
    "collate_mot_seq",
]


def collate_lidar3d(batch_list: List[Any]) -> Union[torch.Tensor, Dict]:
    example_merged = collections.defaultdict(list)
    for example in batch_list:
        if isinstance(example, list):
            for sub_example in example:
                for k, v in sub_example.items():
                    example_merged[k].append(v)
        else:
            for k, v in example.items():
                example_merged[k].append(v)

    ret = {}

    for key, elems in example_merged.items():
        if key in [
            "voxels",
            "num_points",
            "num_gt",
            "num_voxels",
        ]:
            ret[key] = torch.tensor(np.concatenate(elems, axis=0))
        elif key in ["annotations", "metadata"]:
            ret[key] = elems
        elif key == "calib":
            ret[key] = {}
            for elem in elems:
                for k1, v1 in elem.items():
                    if k1 not in ret[key]:
                        ret[key][k1] = [v1]
                    else:
                        ret[key][k1].append(v1)
            for k1, v1 in ret[key].items():
                ret[key][k1] = torch.tensor(np.stack(v1, axis=0))
        elif key in ["coordinates"]:
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)), mode="constant", constant_values=i
                )
                coors.append(coor_pad)
            ret[key] = torch.tensor(
                np.concatenate(coors, axis=0), dtype=torch.int64
            )

        elif key in ["points", "gt_boxes", "gt_classess"]:
            points_lst = [torch.tensor(points) for points in elems]
            ret[key] = points_lst
        elif key in ["gt_seg_labels", "gt_seg_mask"]:
            ret[key] = torch.tensor(np.stack(elems, axis=0))
        else:
            ret[key] = np.stack(elems, axis=0)

    return ret


def _collate_tensor_batch(batch):
    elem = batch[0]
    out = None
    if torch.utils.data.get_worker_info() is not None:
        numel = sum(x.numel() for x in batch)
        storage = elem.storage()._new_shared(numel)
        out = elem.new(storage).resize_(len(batch), *list(elem.size()))
    return torch.stack(batch, 0, out=out)


def _is_numpy_elem(elem):
    elem_type = type(elem)
    return elem_type.__module__ == "numpy" and elem_type.__name__ not in (
        "str_",
        "string_",
    )


def _collate_numpy_batch(batch, elem):
    elem_type = type(elem)
    if elem_type.__name__ in ("ndarray", "memmap"):
        if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
            raise TypeError(
                "default_collate: batch must contain tensors, numpy arrays"
                ", numbers, dicts or lists; found {}".format(elem.dtype)
            )
        return default_collate_v2([torch.as_tensor(b) for b in batch])
    if elem.shape == ():
        return torch.as_tensor(batch)
    return None


def _collate_namedtuple_batch(batch, elem_type):
    return elem_type(*(default_collate_v2(samples) for samples in zip(*batch)))


def default_collate_v2(batch):
    """Extend torch.utils.data.default_collate.

    Handles non-tensor types by converting to lists.

    Args:
        batch: A list of samples from dataset workers.

    Returns:
        Collated batch object. Tensors are stacked, mappings/sequences are
        recursively collated, and unsupported types are kept as lists.
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        return _collate_tensor_batch(batch)
    if _is_numpy_elem(elem):
        collated = _collate_numpy_batch(batch, elem)
        if collated is not None:
            return collated
    elif isinstance(elem, (float, int)):
        dtype = torch.float64 if isinstance(elem, float) else None
        return torch.tensor(batch, dtype=dtype)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {
            key: default_collate_v2([d[key] for d in batch]) for key in elem
        }
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):
        return _collate_namedtuple_batch(batch, elem_type)
    elif isinstance(elem, collections.abc.Sequence):
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError(
                "each element in list of batch should be of equal size"
            )
        transposed = zip(*batch)
        return [default_collate_v2(samples) for samples in transposed]
    return batch


def collate_2d(
    batch: List[Any], verbose: bool = False
) -> Union[torch.Tensor, Dict]:
    """Merge samples into a mini-batch for 2D tasks with inconsistent shapes.

    Keys in img_metas are kept as lists; other keys use default_collate.

    Args:
        batch: A list of samples, typically dicts from dataset `__getitem__`.
        verbose: Whether to log warning for keys not listed in `img_metas`.

    Returns:
        Collated tensor/dict. For dict samples, selected keys are kept as
        per-sample lists and other keys are stacked with default collate.
    """
    elem = batch[0]
    list_key = [
        "gt_classes",
        "gt_bboxes",
        "ig_bboxes",
        "gt_tanalphas",
        "gt_difficult",
        "gt_labels",
        "ori_img",
        "resized_ori_img",
        "img_id",
        "layout",
        "img_shape",
        "resized_shape",
        "color_space",
        "classes",
        "bboxes",
        "crop_offset",
        "before_crop_shape",
        "crop_roi",
        "structure",
        "dgain",
        "rg_gain",
        "bg_gain",
        "bit_nums_upper",
        "bit_nums_lower",
        "channels",
        "cur_pattern",
        "raw_pattern",
        "gt_lines",
        "ori_gt_lines",
        "before_pad_shape",
        "attr_labels",
        "gt_labels_3d",
        "gt_bboxes_3d",
        "centers2d",
        "depths",
        "filename",
        "ori_shape",
        "cam2img",
        "ego2global_translation",
        "ego2global_rotation",
        "sensor2ego_translation",
        "sensor2ego_rotation",
        "token",
        "scale",
        "padded_img",
        "crop_bbox",
        "transform_meta",
        "camera_info",
        "gt_invasion_status",
        "gt_beside_valid",
        "gt_invasion_scale",
    ]
    if not isinstance(elem, dict):
        return default_collate(batch)
    elif isinstance(elem, Mapping):
        return_data = {}
        unexpected_keys = []
        for key in elem:
            if key in list_key:
                collate_data = [d[key] for d in batch]
            else:
                collate_data = default_collate([d[key] for d in batch])
            if key not in img_metas:
                unexpected_keys.append(key)
            return_data[key] = collate_data
        if len(unexpected_keys) > 0 and verbose:
            logging.warning(
                "%s appear in keys of dataset. Please check whether it is an "
                "image task and meets expectations.",
                unexpected_keys,
            )
        return return_data
    return default_collate(batch)


def collate_mot_seq(
    batch: List[Dict],
) -> Union[torch.Tensor, Dict]:
    """Collate for mot seq data.

    Args:
        batch: list of data.
    """
    return_data = {}
    frames_data = []
    list_key = [
        "gt_classes",
        "gt_bboxes",
        "gt_difficult",
        "gt_labels",
        "ori_img",
        "resized_ori_img",
        "img_id",
        "layout",
        "img_shape",
        "resized_shape",
        "color_space",
        "classes",
        "bboxes",
        "crop_offset",
        "before_crop_shape",
        "crop_roi",
        "structure",
        "gt_ids",
        "img",
        "seq_name",
    ]
    for one_seq_data in batch:
        elem = one_seq_data["frame_data_list"][0]
        seq_collate_data = {}
        for key in elem:
            if key in list_key:
                if key == "img":
                    collate_data = []
                    for d in one_seq_data["frame_data_list"]:
                        if isinstance(d[key], np.ndarray):
                            collate_data.append(np.expand_dims(d[key], axis=0))
                        elif isinstance(d[key], torch.Tensor):
                            collate_data.append(d[key].unsqueeze(0))
                        else:
                            raise ValueError(
                                f"Unsupport image datatype: {type(d[key])}"
                            )
                else:
                    collate_data = [
                        d[key] for d in one_seq_data["frame_data_list"]
                    ]
            else:
                collate_data = default_collate(
                    [d[key] for d in one_seq_data["frame_data_list"]]
                )
            seq_collate_data.update({key: collate_data})
        frames_data.append(seq_collate_data)
    return_data.update({"frame_data_list": frames_data})

    return return_data


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def pad_batch_img(imgs: List[torch.Tensor]):
    # imgs[0] shape should be c * h * w
    max_size = _max_by_axis([list(img.shape) for img in imgs])
    batch_shape = [len(imgs)] + max_size
    dtype = imgs[0].dtype
    device = imgs[0].device
    pad_imgs = torch.zeros(batch_shape, dtype=dtype, device=device)
    for img, pad_img in zip(imgs, pad_imgs):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    return pad_imgs


def collate_2d_pad(batch: List[Any]) -> Union[torch.Tensor, Dict]:
    """Merge a list of samples to form a mini-batch of Tensor(s).

    Used in 2d task, for collating data with inconsistent shapes.
    Images with different shapes will pad to max shapes by axis.

    Args:
        batch: list of data.
    """

    elem = batch[0]
    # these key-value will skip default_collate
    list_key = [
        "gt_classes",
        "gt_bboxes",
        "ig_bboxes",
        "gt_difficult",
        "gt_labels",
        "ori_img",
        "resized_ori_img",
        "img_id",
        "layout",
        "img_shape",
        "resized_shape",
        "color_space",
        "classes",
        "bboxes",
        "crop_offset",
        "before_crop_shape",
        "crop_roi",
        "structure",
        "bit_nums_upper",
        "bit_nums_lower",
        "channels",
        "cur_pattern",
        "raw_pattern",
        "gt_lines",
        "ori_gt_lines",
        "before_pad_shape",
    ]
    if not isinstance(elem, dict):
        return pad_batch_img(batch)
    elif isinstance(elem, Mapping):
        return_data = {}
        for key in elem:
            if key in list_key:
                collate_data = [d[key] for d in batch if key in d]
            elif key == "img":
                collate_data = pad_batch_img([d[key] for d in batch])
                pad_img_shape = [collate_data[0].shape[-2:] for _ in batch]
                return_data.update({"batch_input_shape": pad_img_shape})
            else:
                collate_data = default_collate(
                    [d[key] for d in batch if key in d]
                )

            return_data.update({key: collate_data})
        return return_data
