"""Data loaders and batch processors for MapTR."""

import copy
import os

import torch
from common import (
    aux_seg_cfg,
    batch_size_per_gpu,
    bev_h_,
    bev_w_,
    data_rootdir,
    enable_amp,
    enable_amp_dtype,
    fixed_ptsnum_per_gt_line,
    log_freq,
    map_classes,
    map_train_lmdb,
    map_val_lmdb,
    meta_rootdir,
    point_cloud_range,
    queue_length,
    sd_map_path,
    test_queue_length,
    use_lidar_gt,
)

from heal.data.collates.nusc_collates import collate_nuscenes_sequence

_map_dataset_common = dict(
    map_path=meta_rootdir,
    sd_map_path=sd_map_path,
    pc_range=point_cloud_range,
    bev_size=(bev_h_, bev_w_),
    fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
    padding_value=-10000,
    map_classes=map_classes,
    aux_seg=aux_seg_cfg,
    with_bev_bboxes=False,
    with_ego_bboxes=False,
    with_bev_mask=False,
    use_lidar_gt=use_lidar_gt,
)
if use_lidar_gt:
    _map_dataset_common["map_gt_path"] = map_train_lmdb

_val_map_dataset_common = copy.deepcopy(_map_dataset_common)
if use_lidar_gt:
    _val_map_dataset_common["map_gt_path"] = map_val_lmdb

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="NuscenesMapDataset",
        data_path=os.path.join(data_rootdir, "train_lmdb"),
        test_mode=False,
        queue_length=queue_length,
        **_map_dataset_common,
        transforms=[
            dict(type="MultiViewsImgResize", size=(450, 800)),
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
                    dict(type="Normalize", mean=128.0, std=128.0),
                ],
            ),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    shuffle=False,
    batch_size=batch_size_per_gpu,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_nuscenes_sequence,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="NuscenesMapDataset",
        data_path=os.path.join(data_rootdir, "val_lmdb"),
        test_mode=True,
        queue_length=test_queue_length,
        **_val_map_dataset_common,
        transforms=[
            dict(type="MultiViewsImgResize", size=(450, 800)),
            dict(
                type="MultiViewsImgTransformWrapper",
                transforms=[
                    dict(type="PILToTensor"),
                    dict(type="Pad", divisor=32),
                    dict(type="BgrToYuv444", rgb_input=True),
                    dict(type="Normalize", mean=128.0, std=128.0),
                ],
            ),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_nuscenes_sequence,
)

bpu_val_dataset = dict(
    type="NuscenesMapDataset",
    data_path=os.path.join(data_rootdir, "val_lmdb"),
    test_mode=True,
    queue_length=test_queue_length,
    **_val_map_dataset_common,
    transforms=[
        dict(type="MultiViewsImgResize", size=(450, 800)),
        dict(type="MultiViewsImgPad", padding=(0, 0, 0, 30)),
    ],
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

calibration_batch_processor = copy.deepcopy(val_batch_processor)

val_example_data_loader = copy.deepcopy(val_data_loader)
val_example_data_loader.pop("sampler")
val_example_data_loader["num_workers"] = 0

calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader.pop("sampler")
calibration_data_loader["dataset"]["transforms"] = val_data_loader["dataset"][
    "transforms"
]
calibration_example_data_loader = copy.deepcopy(calibration_data_loader)
calibration_example_data_loader["num_workers"] = 0

single_infer_dataset = copy.deepcopy(val_data_loader["dataset"])
single_infer_dataset["transforms"] = None
single_infer_dataset["queue_length"] = 1

int_infer_data_loader = copy.deepcopy(val_data_loader)
int_infer_data_loader["sampler"] = None
int_infer_data_loader["batch_size"] = 1
int_infer_data_loader["shuffle"] = False
int_infer_data_loader["num_workers"] = 0

_hbm_transforms = [
    dict(type="MultiViewsImgResize", size=(450, 800)),
    dict(
        type="MultiViewsImgTransformWrapper",
        transforms=[
            dict(type="PILToTensor"),
            dict(type="Pad", divisor=32),
        ],
    ),
]
hbm_data_loader = copy.deepcopy(int_infer_data_loader)
hbm_data_loader["dataset"]["transforms"] = _hbm_transforms

hbm_batch_processor = copy.deepcopy(val_batch_processor)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=log_freq,
)
