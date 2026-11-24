import copy
import os

import torch
from common import base_data_dir, class_names
from hat_engine.engine.processors.loss_collector import collect_loss_by_index
from model import pc_range

from heal.data.collates.collates import collate_lidar3d

batch_size_per_gpu = 4

db_sampler = dict(
    type="DataBaseSampler",
    enable=True,
    root_path=base_data_dir,
    db_info_path=os.path.join(base_data_dir, "kitti3d_dbinfos_train.pkl"),
    sample_groups=[dict(Car=15)],
    db_prep_steps=[
        dict(
            type="DBFilterByDifficulty",
            filter_by_difficulty=[-1],
        ),
        dict(
            type="DBFilterByMinNumPoint",
            filter_by_min_num_points=dict(
                Car=5,
            ),
        ),
    ],
    global_random_rotation_range_per_object=[0, 0],
    rate=1.0,
)

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Kitti3D",
        data_path=os.path.join(base_data_dir, "train_lmdb"),
        transforms=[
            dict(
                type="ObjectSample",
                class_names=class_names,
                remove_points_after_sample=False,
                db_sampler=db_sampler,
            ),
            dict(
                type="ObjectNoise",
                gt_rotation_noise=[-0.15707963267, 0.15707963267],
                gt_loc_noise_std=[0.25, 0.25, 0.25],
                global_random_rot_range=[0, 0],
                num_try=100,
            ),
            dict(
                type="PointRandomFlip",
                probability=0.5,
            ),
            dict(
                type="PointGlobalRotation",
                rotation=[-0.78539816, 0.78539816],
            ),
            dict(
                type="PointGlobalScaling",
                min_scale=0.95,
                max_scale=1.05,
            ),
            dict(
                type="ShufflePoints",
                shuffle=True,
            ),
            dict(
                type="ObjectRangeFilter",
                point_cloud_range=pc_range,
            ),
            dict(type="LidarReformat"),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_lidar3d,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Kitti3D",
        data_path=os.path.join(base_data_dir, "val_lmdb"),
        transforms=[
            dict(type="LidarReformat"),
        ],
    ),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_lidar3d,
)

batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    loss_collector=collect_loss_by_index(0),
)

val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
    loss_collector=None,
)

calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader.pop("sampler")  # Calibration do not support DDP or DP
calibration_data_loader["batch_size"] = batch_size_per_gpu * 4
calibration_data_loader["dataset"]["transforms"] = val_data_loader["dataset"][
    "transforms"
]
calibration_batch_processor = copy.deepcopy(val_batch_processor)

hbir_data_loader = copy.deepcopy(val_data_loader)
hbir_data_loader["batch_size"] = 1
hbir_data_loader["shuffle"] = False

single_infer_dataset = copy.deepcopy(val_data_loader["dataset"])
single_infer_dataset["transforms"] = None
