import copy

import torch
from common import (
    batch_size_per_gpu,
    data_path,
    data_shape,
    dataloader_workers,
    enable_amp,
    num_classes,
    scale_factors,
)
from hat_engine.engine.processors.loss_collector import collect_loss_by_index
from torchvision.transforms import InterpolationMode

from heal.data.collates import collate_2d
from heal.data.datasets.cityscapes import CITYSCAPES_LABLE_MAPPINGS

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Cityscapes",
        data_path=data_path + "train_lmdb",
        transforms=[
            dict(type="PILToTensor"),
        ],
        color_space="rgb",
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=dataloader_workers,
    pin_memory=True,
    collate_fn=collate_2d,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Cityscapes",
        data_path=data_path + "val_lmdb",
        transforms=[
            dict(type="PILToTensor"),
        ],
        color_space="rgb",
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=dataloader_workers,
    pin_memory=True,
    collate_fn=collate_2d,
)

val_example_data_loader = copy.deepcopy(val_data_loader)
val_example_data_loader.pop("sampler")
val_example_data_loader["num_workers"] = 0

calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader.pop("sampler")
calibration_data_loader["batch_size"] = batch_size_per_gpu * 2

calibration_example_data_loader = copy.deepcopy(calibration_data_loader)
calibration_example_data_loader["num_workers"] = 0

qat_data_loader = copy.deepcopy(data_loader)

hbir_data_loader = copy.deepcopy(val_data_loader)
hbir_data_loader["batch_size"] = 1
hbir_data_loader["shuffle"] = False
hbir_data_loader.pop("sampler", None)
hbir_data_loader["num_workers"] = 0

hbm_data_loader = copy.deepcopy(hbir_data_loader)

inference_dataset = dict(
    type="Cityscapes",
    data_path=data_path + "val_lmdb",
    transforms=None,
)

batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    batch_transforms=[
        dict(type="LabelRemap", mapping=CITYSCAPES_LABLE_MAPPINGS),
        dict(type="SegOneHot", num_classes=num_classes),
        dict(type="SegResize", size=data_shape[1:]),
        dict(
            type="SegRandomAffine",
            degrees=0,
            scale=(0.5, 2.0),
            interpolation=InterpolationMode.BILINEAR,
            label_fill_value=0,
        ),
        dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
        dict(
            type="Scale",
            scales=scale_factors,
            mode="bilinear",
        ),
    ],
    loss_collector=collect_loss_by_index(1),
    enable_amp=enable_amp,
)

val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    batch_transforms=[
        dict(type="LabelRemap", mapping=CITYSCAPES_LABLE_MAPPINGS),
        dict(type="SegOneHot", num_classes=num_classes),
        dict(type="SegResize", size=data_shape[1:]),
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
    batch_transforms=[
        dict(type="LabelRemap", mapping=CITYSCAPES_LABLE_MAPPINGS),
        dict(type="SegOneHot", num_classes=num_classes),
        dict(type="SegResize", size=data_shape[1:]),
    ],
)
