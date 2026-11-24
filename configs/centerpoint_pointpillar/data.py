"""Data loaders for CenterPoint PointPillars NuScenes."""

import copy
import os

import torch
from common import (
    batch_size_per_gpu,
    class_names,
    lidar_train_lmdb,
    lidar_val_lmdb,
    max_num_points,
    point_cloud_range,
    voxel_size,
)

from heal.data.collates.collates import collate_lidar3d


def loss_collector(outputs):
    return list(outputs.values())


train_dataset = dict(
    type="NuscenesLidarDataset",
    num_sweeps=9,
    data_path=lidar_train_lmdb,
    load_dim=5,
    use_dim=[0, 1, 2, 3, 4],
    pad_empty_sweeps=True,
    remove_close=True,
    use_valid_flag=True,
    classes=class_names,
    transforms=[
        dict(
            type="LidarMultiPreprocess",
            class_names=class_names,
            shuffle_points=True,
            flip_both=True,
            global_rot_noise=[-0.3925, 0.3925],
            global_scale_noise=[0.95, 1.05],
            drop_points_in_gt=False,
        ),
        dict(
            type="ObjectRangeFilter",
            point_cloud_range=point_cloud_range,
        ),
        dict(type="LidarReformat", with_gt=True),
    ],
)

# CBGS scans the full train LMDB via get_cat_ids (tens of GB on JuiceFS).
# Smoke / --pipeline-test only needs 1 step and must skip that scan.
_pipeline_test = os.environ.get("HAT_PIPELINE_TEST", "0") == "1"
_train_loader_dataset = (
    train_dataset
    if _pipeline_test
    else dict(type="CBGSDataset", dataset=train_dataset)
)

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=_train_loader_dataset,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=4,
    pin_memory=False,
    collate_fn=collate_lidar3d,
)

val_dataset = dict(
    type="NuscenesLidarDataset",
    test_mode=True,
    num_sweeps=9,
    data_path=lidar_val_lmdb,
    load_dim=5,
    use_dim=[0, 1, 2, 3, 4],
    pad_empty_sweeps=True,
    remove_close=True,
    classes=class_names,
    transforms=[
        dict(type="LidarReformat", with_gt=False),
    ],
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=val_dataset,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_lidar3d,
)

batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    loss_collector=loss_collector,
)

val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
    loss_collector=None,
)

calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader.pop("sampler")
calibration_data_loader["dataset"] = copy.deepcopy(val_dataset)
calibration_data_loader["batch_size"] = batch_size_per_gpu
calibration_batch_processor = copy.deepcopy(val_batch_processor)

calibration_example_data_loader = copy.deepcopy(calibration_data_loader)
calibration_example_data_loader["batch_size"] = 1

hbir_data_loader = copy.deepcopy(val_data_loader)
hbir_data_loader["batch_size"] = 1
hbir_data_loader["shuffle"] = False

single_infer_dataset = copy.deepcopy(val_dataset)
single_infer_dataset["transforms"] = None

pre_process = dict(
    type="CenterPointPreProcess",
    pc_range=point_cloud_range,
    voxel_size=voxel_size,
    max_voxels_num=(40000,),
    max_points_in_voxel=max_num_points,
    norm_range=[-51.2, -51.2, -5.0, 0.0, 51.2, 51.2, 3.0, 255.0],
    norm_dims=[0, 1, 2, 3],
)
