"""Data loaders and batch processors for SparseBEV HENet-tiny NuScenes."""

import copy
import os

import torch
from common import (
    batch_size_per_gpu,
    data_rootdir,
    dataloader_workers,
    lidar_load_dim,
    lidar_train_lmdb,
    lidar_use_dim,
    lidar_val_lmdb,
    num_lidar_sweeps,
    resize_shape,
    val_data_shape,
)

from heal.data.collates.nusc_collates import collate_nuscenes

train_transforms = [
    dict(type="MultiViewsImgResize", scales=(0.40, 0.47)),
    dict(type="MultiViewsImgCrop", size=(256, 704), random=False),
    dict(type="MultiViewsImgFlip"),
    dict(type="MultiViewsImgRotate", rot=(-5.4, 5.4)),
    dict(type="BevBBoxRotation", rotation_3d_range=(-0.3925, 0.3925)),
    dict(type="MultiViewsPhotoMetricDistortion"),
    dict(
        type="MultiViewsGridMask",
        use_h=True,
        use_w=True,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=1,
        prob=0.7,
    ),
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
    dict(type="MultiViewsImgResize", size=resize_shape[1:]),
    dict(type="MultiViewsImgCrop", size=val_data_shape[1:]),
    dict(
        type="MultiViewsImgTransformWrapper",
        transforms=[
            dict(type="PILToTensor"),
            dict(type="BgrToYuv444", rgb_input=True),
            dict(type="Normalize", mean=128, std=128),
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


def _sparse_bev_merge_dataset(
    base_lmdb_path, lidar_lmdb_path, transforms, test_mode
):
    return dict(
        type="MergeDataset",
        join_key="sample_token",
        transforms=transforms,
        datasets=[
            dict(
                type="NuscenesBevDataset",
                data_path=base_lmdb_path,
                transforms=None,
                with_bev_bboxes=False,
                with_ego_bboxes=False,
                num_split=2,
            ),
            dict(
                **_lidar_dataset_common,
                data_path=lidar_lmdb_path,
                test_mode=test_mode,
            ),
        ],
    )


train_dataset = _sparse_bev_merge_dataset(
    os.path.join(data_rootdir, "train_lmdb"),
    lidar_train_lmdb,
    train_transforms,
    test_mode=False,
)
val_dataset = _sparse_bev_merge_dataset(
    os.path.join(data_rootdir, "val_lmdb"),
    lidar_val_lmdb,
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
        skip_prob=0.0,
        sequence_flip_prob=0.0,
    ),
    num_workers=dataloader_workers,
    pin_memory=True,
    collate_fn=collate_nuscenes,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=val_dataset,
    num_workers=dataloader_workers,
    pin_memory=True,
    collate_fn=collate_nuscenes,
    batch_size=1,
    shuffle=False,
)


def loss_collector(outputs: dict):
    losses = []
    for _, loss in outputs.items():
        losses.append(loss)
    return losses


batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    loss_collector=loss_collector,
    grad_scaler=torch.cuda.amp.GradScaler(init_scale=32.0),
    enable_amp=True,
)

val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
)

calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader["dataset"]["transforms"] = val_transforms
calibration_batch_processor = copy.deepcopy(val_batch_processor)

int_infer_data_loader = copy.deepcopy(val_data_loader)
int_infer_data_loader["batch_size"] = 1

# On-board BPU eval: batch_size=1, deterministic order. The HBM infer model
# consumes raw images directly (it does its own YUV/normalization internally
# via nv12_runtime), so drop the BgrToYuv444 + Normalize steps that val
# transforms apply; keep only PILToTensor.
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
