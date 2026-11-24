"""Data loaders and batch processors."""

import copy
import os

import torch
from common import (
    batch_size_per_gpu,
    data_rootdir,
    map_size,
    meta_rootdir,
    point_cloud_range,
    queue_length,
)

from heal.data.collates.nusc_collates import collate_nuscenes_sequence

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="NuscenesBevSequenceDataset",
        data_path=os.path.join(data_rootdir, "train_lmdb"),
        map_size=map_size,
        map_path=meta_rootdir,
        with_ego_bboxes=True,
        bev_range=point_cloud_range,
        num_seq=queue_length,
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
                    dict(type="PILToTensor"),
                    dict(type="Pad", divisor=32),
                    dict(type="BgrToYuv444", rgb_input=True),
                    dict(type="Normalize", mean=128.0, std=128.0),
                ],
            ),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_nuscenes_sequence,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="NuscenesBevSequenceDataset",
        data_path=os.path.join(data_rootdir, "val_lmdb"),
        map_size=map_size,
        map_path=meta_rootdir,
        with_ego_bboxes=True,
        bev_range=point_cloud_range,
        num_seq=1,
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
    sampler=None,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_nuscenes_sequence,
)


def loss_collector(outputs: dict):
    losses = []
    for output in outputs:
        for _, loss in output.items():
            losses.append(loss)
    return losses


batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    loss_collector=loss_collector,
    enable_amp=True,
    enable_amp_dtype=torch.float16,
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

calibration_example_data_loader = copy.deepcopy(calibration_data_loader)
calibration_example_data_loader["num_workers"] = 0


int_infer_data_loader = copy.deepcopy(val_data_loader)
int_infer_data_loader["sampler"] = None
int_infer_data_loader["batch_size"] = 1
int_infer_data_loader["shuffle"] = False

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
hbm_infer_transforms = _hbm_transforms
hbm_data_loader = copy.deepcopy(int_infer_data_loader)
hbm_data_loader["dataset"]["transforms"] = _hbm_transforms


infer_transforms = [
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
]

single_infer_dataset = copy.deepcopy(val_data_loader["dataset"])
single_infer_dataset["transforms"] = None
single_infer_dataset["num_seq"] = 1
