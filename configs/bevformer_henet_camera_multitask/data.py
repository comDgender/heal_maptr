"""Data loaders for BevFormer HENet camera-only det+occ pretrain."""

import os

import torch
from common import (
    batch_size_per_gpu,
    data_rootdir,
    dataloader_workers,
    map_size,
    meta_rootdir,
    occ_train_lmdb,
    occ_val_lmdb,
    point_cloud_range,
    use_lidar2img,
)

from heal.data.collates.nusc_collates import collate_camera_multitask

train_transforms = [
    dict(type="MultiViewsImgResize", size=(512, 960)),
    dict(type="MultiViewsImgFlip"),
    dict(type="MultiViewsImgRotate", rot=(-5.4, 5.4)),
    dict(type="BevBBoxRotation", rotation_3d_range=(-0.3925, 0.3925)),
    dict(type="MultiViewsPhotoMetricDistortion"),
    dict(
        type="MultiViewsImgTransformWrapper",
        transforms=[
            dict(type="PILToTensor"),
            dict(type="BgrToYuv444", rgb_input=True),
            dict(type="Normalize", mean=128, std=128),
        ],
    ),
]

val_transforms = [
    dict(type="MultiViewsImgResize", size=(512, 960)),
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


def _camera_multitask_dataset(
    base_lmdb_path, occ_lmdb_path, transforms, test_mode=False
):
    return dict(
        type="MergeDataset",
        join_key="sample_token",
        with_lidar_occ=True,
        transforms=transforms,
        datasets=[
            dict(
                type="NuscenesBevDataset",
                data_path=base_lmdb_path,
                map_size=map_size,
                map_path=meta_rootdir,
                with_lidar_bboxes=use_lidar2img,
                with_bev_bboxes=False,
                with_ego_bboxes=True,
                bev_range=point_cloud_range,
                test_mode=test_mode,
                transforms=None,
            ),
            dict(
                type="NuscenesOccDataset",
                data_path=occ_lmdb_path,
                transforms=None,
            ),
        ],
    )


train_dataset = _camera_multitask_dataset(
    os.path.join(data_rootdir, "train_lmdb"),
    occ_train_lmdb,
    train_transforms,
    test_mode=False,
)
val_dataset = _camera_multitask_dataset(
    os.path.join(data_rootdir, "val_lmdb"),
    occ_val_lmdb,
    val_transforms,
    test_mode=True,
)

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=train_dataset,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=dataloader_workers,
    pin_memory=False,
    collate_fn=collate_camera_multitask,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=val_dataset,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=1,
    shuffle=False,
    num_workers=dataloader_workers,
    pin_memory=True,
    collate_fn=collate_camera_multitask,
)


def loss_collector(outputs: dict):
    losses = []
    for _, loss in outputs.items():
        losses.append(loss)
    return losses


batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    enable_amp=True,
    loss_collector=loss_collector,
)

val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
    loss_collector=None,
)
