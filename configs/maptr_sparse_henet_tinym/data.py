"""Data loaders and batch processors for sparse MapTR."""

import copy
import os

import torch
from common import (
    aux_seg_cfg,
    batch_size_per_gpu,
    bev_h_,
    bev_w_,
    data_rootdir,
    dataloader_workers,
    enable_amp,
    enable_amp_dtype,
    fixed_ptsnum_per_gt_line,
    lidar_load_dim,
    lidar_train_lmdb,
    lidar_use_dim,
    lidar_val_lmdb,
    map_classes,
    map_train_lmdb,
    map_val_lmdb,
    meta_rootdir,
    num_lidar_sweeps,
    point_cloud_range,
    resize_shape,
    use_lidar_gt,
    val_data_shape,
)

from heal.data.collates.nusc_collates import collate_nuscenes

train_transforms = [
    dict(type="MultiViewsImgResize", scales=(0.40, 0.47)),
    dict(type="MultiViewsImgCrop", size=(256, 704), random=False),
    dict(type="MultiViewsImgFlip"),
    dict(type="MultiViewsImgRotate", rot=(-5.4, 5.4)),
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

_map_dataset_common = dict(
    map_path=meta_rootdir,
    pc_range=point_cloud_range,
    bev_size=(bev_h_, bev_w_),
    fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
    padding_value=-10000,
    map_classes=map_classes,
    with_bev_bboxes=False,
    with_ego_bboxes=False,
    with_bev_mask=False,
    use_lidar_gt=use_lidar_gt,
    aux_seg=aux_seg_cfg,
    num_split=2,
    filter_empty_gt=False,
)

_lidar_dataset_common = dict(
    type="NuscenesLidarDataset",
    num_sweeps=num_lidar_sweeps,
    load_dim=lidar_load_dim,
    use_dim=lidar_use_dim,
    filter_empty_gt=False,
    transforms=None,
)


def _sparse_map_merge_dataset(
    base_lmdb_path,
    lidar_lmdb_path,
    transforms,
    test_mode=False,
    map_gt_path=None,
):
    map_dataset_cfg = dict(**_map_dataset_common)
    if map_gt_path is not None:
        map_dataset_cfg["map_gt_path"] = map_gt_path
    return dict(
        type="MergeDataset",
        join_key="sample_token",
        transforms=None,
        datasets=[
            dict(
                type="NuscenesSparseMapDataset",
                data_path=base_lmdb_path,
                test_mode=test_mode,
                transforms=transforms,
                **map_dataset_cfg,
            ),
            dict(
                **_lidar_dataset_common,
                data_path=lidar_lmdb_path,
                test_mode=test_mode,
            ),
        ],
    )


train_dataset = _sparse_map_merge_dataset(
    os.path.join(data_rootdir, "train_lmdb"),
    lidar_train_lmdb,
    train_transforms,
    test_mode=False,
    map_gt_path=map_train_lmdb,
)
val_dataset = _sparse_map_merge_dataset(
    os.path.join(data_rootdir, "val_lmdb"),
    lidar_val_lmdb,
    val_transforms,
    test_mode=True,
    map_gt_path=map_val_lmdb,
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
    for outs in outputs:
        for _, loss in outs.items():
            losses.append(loss)
    return losses


batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    loss_collector=loss_collector,
    enable_amp=enable_amp,
    enable_amp_dtype=enable_amp_dtype,
)

val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    loss_collector=None,
)

calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader["dataset"]["datasets"][0][
    "transforms"
] = val_transforms
calibration_batch_processor = copy.deepcopy(val_batch_processor)

calibration_example_data_loader = copy.deepcopy(calibration_data_loader)
calibration_example_data_loader["num_workers"] = 0

val_example_data_loader = copy.deepcopy(val_data_loader)
val_example_data_loader["num_workers"] = 0

int_infer_data_loader = copy.deepcopy(val_data_loader)
int_infer_data_loader["batch_size"] = 1

_hbm_transforms = [
    dict(type="MultiViewsImgResize", size=resize_shape[1:]),
    dict(type="MultiViewsImgCrop", size=val_data_shape[1:]),
    dict(
        type="MultiViewsImgTransformWrapper",
        transforms=[
            dict(type="PILToTensor"),
        ],
    ),
]

hbm_data_loader = copy.deepcopy(val_data_loader)
hbm_data_loader["batch_size"] = 1
hbm_data_loader["shuffle"] = False
hbm_data_loader["dataset"]["datasets"][0]["transforms"] = _hbm_transforms

hbm_batch_processor = copy.deepcopy(val_batch_processor)

infer_transforms = val_transforms

single_infer_dataset = copy.deepcopy(val_dataset)
single_infer_dataset["datasets"][0]["transforms"] = None
