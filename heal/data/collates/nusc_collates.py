# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import collections
from typing import Any, List

import numpy as np
import torch

from .collates import default_collate

__all__ = [
    "collate_bevfusion",
    "collate_camera_multitask",
    "collate_nuscenes",
    "collate_nuscenes_sequence",
    "collate_nuscenes_multitask",
    "collate_sparse_lidar_fusion",
]


def collate_nuscenes(batch: List[Any]):
    """Merge a list of samples to form a mini-batch of Tensor(s).

    Args:
        batch: list of data.
    """
    elem = batch[0]
    # these key-value will skip default_collate
    cam_keys = ["img", "ego2img", "lidar2img", "cam2ego", "camera_intrinsic"]
    list_keys = [
        "img_name",
        "cam_name",
        "bev_bboxes",
        "bev_cat_ids",
        "bev_bboxes_labels",
        "layout",
        "color_space",
        "meta",
        "sample_token",
        "ego2global",
        "ego_bboxes_labels",
        "lidar_bboxes_labels",
        "center2ds",
        "corner2ds",
        "mono_3d_bboxes",
        "mono_3d_labels",
        "depths",
        "attr_labels",
        "instance_ids",
        "gt_labels_map",
        "gt_instances",
        "gt_pv_seg_mask",
        "osm_vectors",
        "metadata",
        "mode",
    ]
    if not isinstance(elem, dict):
        return default_collate(batch)
    elif isinstance(elem, collections.abc.Mapping):
        return_data = {}
        for key in elem:
            if key in list_keys:
                collate_data = [d[key] for d in batch]
            elif key in cam_keys:
                collate_data = []
                for d in batch:
                    for item in d[key]:
                        if isinstance(item, np.ndarray):
                            item = torch.tensor(item)
                        collate_data.append(item)

                collate_data = torch.stack(collate_data, dim=0)
            elif key == "points":
                collate_data = []
                for d in batch:
                    points = torch.tensor(d[key])
                    collate_data.append(points)
            elif key == "lidar":
                collate_data = {}
                for sub_key in elem[key]:
                    if sub_key == "points":
                        collate_data[sub_key] = [
                            torch.tensor(d[key][sub_key]) for d in batch
                        ]
                    else:
                        collate_data[sub_key] = [
                            d[key][sub_key] for d in batch
                        ]
            else:
                collate_data = default_collate([d[key] for d in batch])
            return_data.update({key: collate_data})

        return return_data


def collate_nuscenes_sequence(batch: List[Any]):
    """Merge a list of sequence samples to form a mini-batch of Tensor(s).

    Args:
        batch: list of data, each element is a temporal sequence of frames.
    """
    cam_keys = [
        "img",
        "cam2ego",
        "camera_intrinsic",
        "gt_depth",
        "gt_depth_mask",
        "depth_map",
        "depth_map_mask",
    ]
    list_keys = [
        "img_name",
        "cam_name",
        "bev_bboxes",
        "bev_cat_ids",
        "bev_bboxes_labels",
        "layout",
        "color_space",
        "meta",
        "sample_token",
        "ego2global",
        "ego_bboxes_labels",
        "lidar_bboxes_labels",
        "lidar",
        "prev_bev_exists",
        "lidar2img",
        "can_bus",
        "img_shape",
        "ego2img",
        "cams_monos",
        "cams_monos_idx",
        "pad_shape",
        "scene",
        "img_metas",
        "gt_labels_map",
        "gt_instances",
        "gt_seg_mask",
        "gt_pv_seg_mask",
        "lidar2global",
        "osm_vectors",
        "gt_occ_info",
        "gt_moving_objects_info",
        "gt_static_objects_info",
        "gt_discrete_map_info",
        "lidar2ego_translation",
        "lidar2ego_rotation",
        "ego2global_translation",
        "ego2global_rotation",
    ]
    stack_keys = ["osm_mask"]

    elem = batch[0][0]
    seq_data = {}
    seq_data["seq_meta"] = []
    seq_length = len(batch[0])
    for _ in range(seq_length):
        seq_data["seq_meta"].append({})
    for key in elem:
        if key in list_keys:
            for idx in range(seq_length):
                if key == "can_bus":
                    collate_data = [
                        torch.tensor(d[idx][key], dtype=torch.float32)
                        for d in batch
                    ]
                    collate_data = torch.stack(collate_data, dim=0)
                elif key in [
                    "gt_moving_objects_info",
                    "gt_static_objects_info",
                    "gt_discrete_map_info",
                ]:
                    collate_data = {}
                    for k2 in batch[0][idx][key]:
                        collate_data[k2] = [d[idx][key][k2] for d in batch]
                elif key == "gt_occ_info":
                    collate_data = {}
                    for k2 in batch[0][idx][key]:
                        collate_data[k2] = torch.stack(
                            [
                                torch.tensor(
                                    d[idx][key][k2], dtype=torch.uint8
                                )
                                for d in batch
                            ]
                        )
                else:
                    collate_data = [d[idx][key] for d in batch]
                seq_data["seq_meta"][idx][key] = collate_data
        elif key in cam_keys:
            collate_data = []
            for idx in range(seq_length):
                for d in batch:
                    for item in d[idx][key]:
                        if isinstance(item, np.ndarray):
                            item = torch.tensor(item)
                        collate_data.append(item)
            collate_data = torch.stack(collate_data, dim=0)
            seq_data[key] = collate_data
        elif key == "points":
            collate_data = []
            for idx in range(seq_length):
                for d in batch:
                    points = torch.tensor(d[idx][key])
                    collate_data.append(points)
            seq_data[key] = collate_data
        elif key in stack_keys:
            if elem[key] is not None:
                collate_data = []
                for d in batch:
                    item = d[0][key]
                    collate_data.append(item)
                collate_data = torch.stack(collate_data, dim=0)
                seq_data[key] = collate_data
            else:
                seq_data[key] = None

    return seq_data


def collate_camera_multitask(batch: List[Any]):
    """Collate single-frame multitask samples via sequence collate."""
    return collate_nuscenes_sequence([[sample] for sample in batch])


def collate_sparse_lidar_fusion(batch: List[Any]):
    """Collate camera+lidar fusion samples for SparseBEV lidar fusion."""
    for sample in batch:
        if "points" not in sample and "lidar" in sample:
            points = sample["lidar"].get("points")
            if points is not None:
                sample["points"] = points
    return collate_nuscenes(batch)


def collate_bevfusion(batch: List[Any]):
    """Collate BevFusion multisensor multitask samples as single-frame sequences."""  # noqa: E501
    for sample in batch:
        if "points" not in sample and "lidar" in sample:
            points = sample["lidar"].get("points")
            if points is not None:
                sample["points"] = points
    return collate_nuscenes_sequence([[sample] for sample in batch])


def collate_nuscenes_multitask(batch: List[Any]):
    """Merge a list of samples to form a mini-batch of Tensor(s).

    Args:
        batch: list of data.
    """
    elem = batch[0]
    # these key-value will skip default_collate
    cam_keys = ["img", "ego2img", "lidar2img", "cam2ego", "camera_intrinsic"]
    list_keys = [
        "img_name",
        "cam_name",
        "bev_bboxes",
        "bev_cat_ids",
        "bev_bboxes_labels",
        "layout",
        "color_space",
        "meta",
        "sample_token",
        "token",
        "ego2global",
        "ego_bboxes_labels",
        "lidar_bboxes_labels",
        "points",
        "center2ds",
        "corner2ds",
        "mono_3d_bboxes",
        "mono_3d_labels",
        "depths",
        "attr_labels",
        "instance_ids",
        "gt_labels_map",
        "gt_instances",
        "gt_pv_seg_mask",
        "osm_vectors",
        "gt_seg_mask",
        "coords",
        "gt_moving_objects_info",
        "gt_static_objects_info",
        "gt_discrete_map_info",
        # E2E multi-frame keys from NavsimE2EDataset
        "history_frames",
        "ego_poses",
        "num_history_frames",
    ]
    if not isinstance(elem, dict):
        return default_collate(batch)
    elif isinstance(elem, collections.abc.Mapping):
        return_data = {}
        for key in elem:
            if key in list_keys:
                if key in [
                    "gt_moving_objects_info",
                    "gt_static_objects_info",
                    "gt_discrete_map_info",
                ]:
                    collate_data = {}
                    for k2 in batch[0][key]:
                        collate_data[k2] = [d[key][k2] for d in batch]
                else:
                    collate_data = [d[key] for d in batch]
            elif key in cam_keys:
                collate_data = []
                for d in batch:
                    for item in d[key]:
                        if isinstance(item, np.ndarray):
                            item = torch.tensor(item)
                        collate_data.append(item)

                collate_data = torch.stack(collate_data, dim=0)
            elif key == "points":
                collate_data = []
                for d in batch:
                    points = torch.tensor(d[key])
                    collate_data.append(points)
            else:
                collate_data = default_collate([d[key] for d in batch])
            return_data.update({key: collate_data})
        seq_batch = []
        for cur_seq_batch in batch:
            tmp_batch = {}
            for k, v in cur_seq_batch.items():
                if k != "img":
                    tmp_batch[k] = v
            # seq_batch.append([cur_seq_batch])
            seq_batch.append([tmp_batch])
        del batch
        seq_data = collate_nuscenes_sequence(seq_batch)
        return_data["seq_meta"] = seq_data["seq_meta"]
        return return_data
