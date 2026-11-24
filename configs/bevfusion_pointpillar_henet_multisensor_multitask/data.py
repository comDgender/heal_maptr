"""Data loaders and batch processors for BevFusion multisensor multitask."""

import copy
import os

import torch
from common import (
    data_rootdir,
    dataloader_workers,
    lidar_load_dim,
    lidar_train_lmdb,
    lidar_use_dim,
    lidar_val_lmdb,
    map_size,
    meta_rootdir,
    num_lidar_sweeps,
    occ_train_lmdb,
    occ_val_lmdb,
    point_cloud_range,
    use_lidar2img,
)

from heal.data.collates.nusc_collates import collate_bevfusion

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

_lidar_dataset_common = dict(
    type="NuscenesLidarDataset",
    num_sweeps=num_lidar_sweeps,
    load_dim=lidar_load_dim,
    use_dim=lidar_use_dim,
    filter_empty_gt=False,
    transforms=None,
)


def _bevfusion_dataset(
    base_lmdb_path,
    lidar_lmdb_path,
    occ_lmdb_path,
    transforms,
    test_mode=False,
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
                **_lidar_dataset_common,
                data_path=lidar_lmdb_path,
                test_mode=test_mode,
            ),
            dict(
                type="NuscenesOccDataset",
                data_path=occ_lmdb_path,
                transforms=None,
            ),
        ],
    )


train_dataset = _bevfusion_dataset(
    os.path.join(data_rootdir, "train_lmdb"),
    lidar_train_lmdb,
    occ_train_lmdb,
    train_transforms,
    test_mode=False,
)
val_dataset = _bevfusion_dataset(
    os.path.join(data_rootdir, "val_lmdb"),
    lidar_val_lmdb,
    occ_val_lmdb,
    val_transforms,
    test_mode=True,
)


data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=train_dataset,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=1,
    shuffle=False,
    num_workers=dataloader_workers,
    pin_memory=False,
    collate_fn=collate_bevfusion,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=val_dataset,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=1,
    shuffle=False,
    num_workers=dataloader_workers,
    pin_memory=True,
    collate_fn=collate_bevfusion,
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

calibration_data_loader = copy.deepcopy(val_data_loader)
calibration_data_loader.pop("sampler")
calibration_example_data_loader = copy.deepcopy(calibration_data_loader)
calibration_example_data_loader["num_workers"] = 0
calibration_batch_processor = copy.deepcopy(val_batch_processor)

int_infer_data_loader = copy.deepcopy(val_data_loader)
int_infer_data_loader["batch_size"] = 1
int_infer_data_loader["shuffle"] = False

# On-board BPU eval: batch_size=1, deterministic order. The HBM infer model
# (BevFusionHbirInfer, use_nv12=True) consumes raw images and converts them to
# NV12 on host for the pyramid input_source, so drop BgrToYuv444 + Normalize
# that val transforms apply; keep only PILToTensor.
hbm_data_loader = copy.deepcopy(val_data_loader)
hbm_data_loader["batch_size"] = 1
hbm_data_loader["shuffle"] = False
for _transform in hbm_data_loader["dataset"]["transforms"]:
    if _transform.get("type") == "MultiViewsImgTransformWrapper":
        _transform["transforms"] = [
            _sub
            for _sub in _transform["transforms"]
            if _sub.get("type") not in ("BgrToYuv444", "Normalize")
        ]
hbm_batch_processor = copy.deepcopy(val_batch_processor)

infer_transforms = val_transforms

single_infer_dataset = copy.deepcopy(val_dataset)
single_infer_dataset["transforms"] = None
