# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import copy
import os

import numpy as np
import torch
from common import (
    aux_seg_cfg,
    batch_size_per_gpu,
    bev_size,
    data_rootdir,
    dataloader_workers,
    discrete_map_class_names,
    fixed_ptsnum_per_gt_line,
    input_size,
    map_classes,
    map_size,
    meta_rootdir,
    occ_train_lmdb,
    occ_val_lmdb,
    point_cloud_range,
)
from shapely.geometry import LineString

from heal.data.collates.nusc_collates import collate_nuscenes_multitask
from heal.data.datasets.nuscenes.nuscenes_map_geometry import (
    LiDARInstanceLines,
)

# Image/geo augs only. MapTR / discrete-map / split run in
# NuscenesMultitaskDataset.prepare_* after BevBBoxRotation (HAT order).
train_transforms = [
    dict(type="MultiViewsImgResize", size=input_size),
    dict(type="MultiViewsImgFlip"),
    dict(type="MultiViewsImgRotate", rot=(-5.4, 5.4)),
    dict(type="BevBBoxRotation", rotation_3d_range=(-0.3925, 0.3925)),
    dict(type="MultiViewsPhotoMetricDistortion"),
    dict(
        type="MultiViewsImgTransformWrapper",
        transforms=[
            dict(
                type="TorchVisionAdapter",
                interface="ColorJitter",
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1,
            ),
            dict(type="PILToNumpy"),
            dict(
                type="GridMask",
                use_h=True,
                use_w=True,
                rotate=1,
                offset=False,
                ratio=0.5,
                mode=1,
                prob=0.7,
            ),
            dict(type="ToTensor", to_yuv=False),
            dict(type="Pad", divisor=32),
            dict(type="BgrToYuv444", rgb_input=True),
            dict(type="Normalize", mean=128, std=128),
        ],
    ),
]


val_transforms = [
    dict(type="MultiViewsImgResize", size=input_size),
    dict(
        type="MultiViewsImgTransformWrapper",
        transforms=[
            dict(type="PILToTensor"),
            dict(type="Pad", divisor=32),
            dict(type="BgrToYuv444", rgb_input=True),
            dict(type="Normalize", mean=128.0, std=128.0),
        ],
    ),
]


# Shared params for the base camera dataset (dynamic/static det split +
# discrete_map + maptr GT come from here).
_map_multitask_common = dict(
    map_path=meta_rootdir,
    bev_size=bev_size,
    map_size=map_size,
    bev_range=point_cloud_range,
    discrete_map_classes=discrete_map_class_names,
    discrete_map_bev_size=bev_size,
    with_yaw=True,
    fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
    padding_value=-10000,
    map_classes=map_classes,
    with_bev_bboxes=False,
    with_ego_bboxes=False,
    with_bev_mask=False,
    with_lidar_bboxes=True,
    use_lidar_gt=True,
    split_dynamic_det=True,
    filter_empty_gt=False,
    # Align HAT dense_multitask: shorten DistStream scene groups for track.
    num_split=2,
    sd_map_path=None,
    aux_seg=aux_seg_cfg,
)


def _multitask_dataset(base_lmdb_path, occ_lmdb_path, transforms, test_mode):
    """Build a MergeDataset: base camera lmdb + separate occ lmdb.

    Geometric augs live on the camera child so ``prepare_train_data`` sees
    rotated ``lidar2global`` before building map GT / split (HAT order).
    """
    return dict(
        type="MergeDataset",
        join_key="sample_token",
        with_lidar_occ=True,
        transforms=None,
        datasets=[
            dict(
                type="NuscenesMultitaskDataset",
                data_path=base_lmdb_path,
                test_mode=test_mode,
                transforms=transforms,
                **_map_multitask_common,
            ),
            dict(
                type="NuscenesOccDataset",
                data_path=occ_lmdb_path,
                transforms=None,
            ),
        ],
    )


train_dataset = _multitask_dataset(
    os.path.join(data_rootdir, "train_lmdb"),
    occ_train_lmdb,
    train_transforms,
    test_mode=False,
)
val_dataset = _multitask_dataset(
    os.path.join(data_rootdir, "val_lmdb"),
    occ_val_lmdb,
    val_transforms,
    test_mode=True,
)


data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=train_dataset,
    batch_sampler=dict(
        type="DistStreamBatchSampler",
        batch_size=batch_size_per_gpu,
        dataset=train_dataset,
        keep_consistent_seq_aug=True,
    ),
    num_workers=dataloader_workers,
    pin_memory=True,
    collate_fn=collate_nuscenes_multitask,
)


val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=dataloader_workers,
    pin_memory=True,
    drop_last=False,
    collate_fn=collate_nuscenes_multitask,
)


calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader["dataset"]["datasets"][0][
    "transforms"
] = val_transforms
calibration_data_loader["batch_sampler"]["batch_size"] = 8


def loss_collector(outputs: dict):
    losses = []
    for _, loss in outputs.items():
        losses.append(loss)
    return losses


batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    enable_amp=False,
    loss_collector=loss_collector,
)
val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
    loss_collector=None,
)

calibration_batch_processor = copy.deepcopy(val_batch_processor)

# HBIR: same YUV444 + Normalize as float/calibration val.
int_infer_data_loader = copy.deepcopy(val_data_loader)
int_infer_data_loader["batch_size"] = 1
int_infer_data_loader["shuffle"] = False

# HBM / pyramid+NV12: keep RGB uint8; BevMultitaskIrInfer.img2nv12 converts
# on-device. Strip BgrToYuv444 + Normalize from nested MergeDataset transforms.
hbm_data_loader = copy.deepcopy(val_data_loader)
hbm_data_loader["batch_size"] = 1
hbm_data_loader["shuffle"] = False
hbm_data_loader["num_workers"] = 0


def _strip_yuv_normalize(node):
    if isinstance(node, dict):
        if node.get("type") == "MultiViewsImgTransformWrapper":
            node["transforms"] = [
                _sub
                for _sub in node.get("transforms", [])
                if _sub.get("type") not in ("BgrToYuv444", "Normalize")
            ]
        for _v in node.values():
            _strip_yuv_normalize(_v)
    elif isinstance(node, list):
        for _v in node:
            _strip_yuv_normalize(_v)


_strip_yuv_normalize(hbm_data_loader["dataset"])
hbm_batch_processor = copy.deepcopy(val_batch_processor)

example_inputs = dict(
    img=torch.randn(6, 3, input_size[0], input_size[1]),
    seq_meta=[
        {
            "lidar2img": [[np.zeros((4, 4))] * 6],
            "gt_moving_objects_info": {
                "lidar_bboxes_labels": [
                    torch.ones((2, 10), dtype=torch.float32)
                ],
                "instance_ids": [[1, 2]],
            },
            "gt_static_objects_info": {
                "lidar_bboxes_labels": [
                    torch.ones((2, 10), dtype=torch.float32)
                ],
                "instance_ids": [[1, 2]],
            },
            "gt_discrete_map_info": {
                "lidar_bboxes_labels": [
                    torch.zeros((2, 6), dtype=torch.float32)
                ],
            },
            "gt_occ_info": {
                "voxel_semantics": torch.ones((200, 200, 16)),
                "mask_camera": torch.ones((200, 200, 16)),
            },
            "gt_labels_map": torch.ones(1),
            "gt_seg_mask": torch.ones((1, 100, 50)),
            "gt_instances": LiDARInstanceLines(
                [LineString(np.random.randn(3, 2))],
                sample_dist=1,
                num_samples=250,
                padding=False,
                fixed_num=20,
                padding_value=-10000,
                patch_size=(80.0, 40.0),
            ),
        }
    ],
    timestamp=[torch.Tensor([1.5e9])],
    lidar2ego=torch.zeros((1, 4, 4)),
    lidar2global=torch.zeros((1, 4, 4)),
)

calibration_data_loader["num_workers"] = 0
