import copy

import torch
from common import (
    batch_size_per_gpu,
    data_num_workers,
    train_data_path,
    val_data_path,
)

from heal.data.collates import collate_2d

train_transforms = [
    dict(
        type="FixedCrop",
        size=(0, 270, 1640, 320),
    ),
    dict(
        type="RandomFlip",
        px=0.5,
        py=0.0,
    ),
    dict(
        type="Resize",
        img_scale=(320, 800),
        multiscale_mode="value",
        keep_ratio=False,
    ),
    dict(
        type="RandomSelectOne",
        transforms=[
            dict(
                type="RGBShift",
                r_shift_limit=(-10, 10),
                g_shift_limit=(-10, 10),
                b_shift_limit=(-10, 10),
                p=1.0,
            ),
            dict(
                type="HueSaturationValue",
                hue_range=(-10, 10),
                sat_range=(-15, 15),
                val_range=(-10, 10),
                p=1.0,
            ),
        ],
        p=0.7,
    ),
    dict(
        type="JPEGCompress",
        p=0.2,
        max_quality=85,
        min_quality=95,
    ),
    dict(
        type="RandomSelectOne",
        transforms=[
            dict(
                type="MeanBlur",
                ksize=3,
                p=1.0,
            ),
            dict(
                type="MedianBlur",
                ksize=3,
                p=1.0,
            ),
        ],
        p=0.2,
    ),
    dict(
        type="RandomBrightnessContrast",
        brightness_limit=(-0.2, 0.2),
        contrast_limit=(-0.0, 0.0),
        p=0.5,
    ),
    dict(
        type="ShiftScaleRotate",
        shift_limit=(-0.1, 0.1),
        scale_limit=(0.8, 1.2),
        rotate_limit=(-10, 10),
        interpolation=1,
        border_mode=0,
        p=0.6,
    ),
    dict(
        type="LaneRandomResizedCrop",
        height=320,
        width=800,
        scale=(0.8, 1.2),
        ratio=(1.7, 2.7),
        p=0.6,
    ),
    dict(
        type="ToTensor",
        to_yuv=False,
    ),
]

val_transforms = [
    dict(
        type="FixedCrop",
        size=(0, 270, 1640, 320),
    ),
    dict(
        type="Resize",
        img_scale=(320, 800),
        multiscale_mode="value",
        keep_ratio=False,
    ),
    dict(
        type="ToTensor",
        to_yuv=False,
    ),
]

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="CuLaneDataset",
        data_path=train_data_path,
        to_rgb=True,
        transforms=train_transforms,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    pin_memory=True,
    shuffle=True,
    num_workers=data_num_workers,
    collate_fn=collate_2d,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="CuLaneDataset",
        data_path=val_data_path,
        to_rgb=True,
        transforms=val_transforms,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    pin_memory=True,
    shuffle=False,
    num_workers=data_num_workers,
    collate_fn=collate_2d,
)

calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader.pop("sampler")
calibration_data_loader["batch_size"] = batch_size_per_gpu * 2
calibration_data_loader["dataset"]["transforms"] = val_transforms

hbir_data_loader = copy.deepcopy(val_data_loader)
hbir_data_loader["batch_size"] = 1
hbir_data_loader["shuffle"] = False

hbm_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="CuLaneDataset",
        data_path=val_data_path,
        to_rgb=True,
        transforms=val_transforms,
    ),
    batch_size=1,
    shuffle=False,
    num_workers=data_num_workers,
    pin_memory=True,
    collate_fn=collate_2d,
)

inference_dataset = dict(
    type="CuLaneDataset",
    data_path=val_data_path,
    to_rgb=True,
)


def loss_collector(outputs: dict):
    losses = []
    for _, loss in outputs.items():
        losses.append(loss)
    return losses


batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    loss_collector=loss_collector,
    batch_transforms=[
        dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
)

val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    loss_collector=None,
    batch_transforms=[
        dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
)

hbm_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    loss_collector=None,
)

deploy_inputs = dict(img=torch.randn((1, 3, 320, 800)))
