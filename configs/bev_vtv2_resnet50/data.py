import copy
import os

import torch
from common import batch_size_per_gpu, data_rootdir, dataloader_workers

from heal.data.collates.nusc_collates import collate_nuscenes

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="NuscenesBevDataset",
        data_path=os.path.join(data_rootdir, "train_lmdb"),
        transforms=[
            dict(type="MultiViewsImgResize", size=(396, 704)),
            dict(type="MultiViewsImgCrop", size=(256, 704)),
            dict(type="MultiViewsImgFlip", prob=0.5),
            dict(type="MultiViewsImgRotate", rot=(-5.4, 5.4)),
            dict(type="BevBBoxRotation", rotation_3d_range=(-0.3925, 0.3925)),
            dict(
                type="MultiViewsImgTransformWrapper",
                transforms=[
                    dict(type="PILToTensor"),
                    dict(type="BgrToYuv444", rgb_input=True),
                    dict(type="Normalize", mean=128.0, std=128.0),
                ],
            ),
        ],
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
        transforms=[
            dict(type="MultiViewsImgResize", size=(396, 704)),
            dict(type="MultiViewsImgCrop", size=(256, 704)),
            dict(
                type="MultiViewsImgTransformWrapper",
                transforms=[
                    dict(type="PILToTensor"),
                    dict(type="BgrToYuv444", rgb_input=True),
                    dict(type="Normalize", mean=128.0, std=128.0),
                ],
            ),
        ],
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

qat_data_loader = copy.deepcopy(calibration_data_loader)

int_infer_data_loader = copy.deepcopy(val_data_loader)
int_infer_data_loader["batch_size"] = 1
int_infer_data_loader["shuffle"] = False

hbir_data_loader = copy.deepcopy(int_infer_data_loader)

_hbm_transforms = [
    dict(type="MultiViewsImgResize", size=(396, 704)),
    dict(type="MultiViewsImgCrop", size=(256, 704)),
    dict(
        type="MultiViewsImgTransformWrapper",
        transforms=[dict(type="PILToTensor")],
    ),
]
hbm_data_loader = copy.deepcopy(int_infer_data_loader)
hbm_data_loader["dataset"]["transforms"] = _hbm_transforms

single_infer_dataset = copy.deepcopy(int_infer_data_loader["dataset"])
single_infer_dataset["transforms"] = None


def loss_collector(outputs: dict):
    losses = []
    for _, loss in outputs[1].items():
        losses.append(loss)
    return losses


batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    loss_collector=loss_collector,
    enable_amp=False,
)
val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
)
hbm_batch_processor = copy.deepcopy(val_batch_processor)
