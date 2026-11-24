import copy
import os

import torch
from common import (
    batch_size_per_gpu,
    bev_range,
    data_rootdir,
    dataloader_workers,
)

from heal.data.collates.nusc_collates import collate_nuscenes

train_transforms = [
    dict(type="MultiViewsImgResize", size=(792, 1408)),
    dict(type="MultiViewsImgCrop", size=(512, 1408), random=False),
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
            dict(type="Normalize", mean=128.0, std=128.0),
        ],
    ),
]

val_transforms = [
    dict(type="MultiViewsImgResize", size=(792, 1408)),
    dict(type="MultiViewsImgCrop", size=(512, 1408)),
    dict(
        type="MultiViewsImgTransformWrapper",
        transforms=[
            dict(type="PILToTensor"),
            dict(type="BgrToYuv444", rgb_input=True),
            dict(type="Normalize", mean=128.0, std=128.0),
        ],
    ),
]

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="NuscenesBevDataset",
        data_path=os.path.join(data_rootdir, "train_lmdb"),
        transforms=train_transforms,
        with_lidar_bboxes=False,
        with_ego_bboxes=True,
        bev_range=bev_range,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=dataloader_workers,
    pin_memory=True,
    collate_fn=collate_nuscenes,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="NuscenesBevDataset",
        data_path=os.path.join(data_rootdir, "val_lmdb"),
        transforms=val_transforms,
        with_lidar_bboxes=False,
        with_ego_bboxes=True,
        bev_range=bev_range,
    ),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=dataloader_workers,
    pin_memory=True,
    collate_fn=collate_nuscenes,
    sampler=dict(type=torch.utils.data.DistributedSampler),
)

calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader.pop("sampler")
calibration_data_loader["batch_size"] = batch_size_per_gpu * 2
calibration_data_loader["dataset"]["transforms"] = val_data_loader["dataset"][
    "transforms"
]

hbir_data_loader = copy.deepcopy(val_data_loader)
hbir_data_loader["batch_size"] = 1
hbir_data_loader["shuffle"] = False

hbm_data_loader = copy.deepcopy(val_data_loader)
hbm_data_loader["batch_size"] = 1
hbm_data_loader["shuffle"] = False
# Remove BgrToYuv444 and Normalize for HBM NV12 input
hbm_data_loader["dataset"]["transforms"] = [
    dict(type="MultiViewsImgResize", size=(792, 1408)),
    dict(type="MultiViewsImgCrop", size=(512, 1408)),
    dict(
        type="MultiViewsImgTransformWrapper",
        transforms=[
            dict(type="PILToTensor"),
        ],
    ),
]

inference_dataset = dict(
    type="NuscenesBevDataset",
    data_path=os.path.join(data_rootdir, "val_lmdb"),
    with_lidar_bboxes=False,
    with_ego_bboxes=True,
    bev_range=bev_range,
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
)

val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
)

calibration_batch_processor = copy.deepcopy(val_batch_processor)

hbm_batch_processor = copy.deepcopy(val_batch_processor)


def get_deploy_input():
    inputs = {
        "img": torch.randn((6, 3, 512, 1408)),
        "pos_embed": torch.randn((1, 256, 96, 44)),
    }
    return inputs


deploy_inputs = get_deploy_input()


def get_example_input():
    inputs = {
        "img": torch.randn((6, 3, 512, 1408)),
        "ego2img": torch.randn((6, 4, 4)),
    }
    return inputs


example_inputs = get_example_input()
