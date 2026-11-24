import copy

import torch
from common import batch_size_per_gpu

from heal.data.collates import collate_2d

data_path = "./tmp_data/heal_data/mscoco/"
ann_file_val = data_path + "instances_val2017.json"

transform_step1 = [
    dict(type="Resize", img_scale=(896, 896), keep_ratio=True),
    dict(
        type="Mosaic",
        image_size=896,
        mixup=True,
        degrees=0.0,
        translate=0.5,
        scale=0.5,
        shear=0,
        perspective=0.0,
    ),
    dict(
        type="RandomFlip",
        px=0.5,
        py=0,
    ),
    dict(type="AugmentHSV", hgain=0.015, sgain=0.7, vgain=0.4),
    dict(
        type="ToTensor",
        to_yuv=True,
        use_yuv_v2=False,
    ),
    dict(
        type="Normalize",
        mean=128.0,
        std=128.0,
    ),
]

transform_step2 = [
    dict(
        type="Resize",
        img_scale=(896, 896),
        keep_ratio=True,
    ),
    dict(
        type="Pad",
        size=(896, 896),
    ),
    dict(
        type="RandomFlip",
        px=0.5,
        py=0,
    ),
    dict(type="AugmentHSV", hgain=0.015, sgain=0.7, vgain=0.4),
    dict(
        type="ToTensor",
        to_yuv=True,
        use_yuv_v2=False,
    ),
    dict(
        type="Normalize",
        mean=128.0,
        std=128.0,
    ),
]

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="BatchTransformDataset",
        dataset=dict(
            type="Coco",
            data_path=data_path + "train_lmdb/",
        ),
        transforms_cfgs=[
            transform_step1,
            transform_step2,
        ],
        epoch_steps=[385],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
    collate_fn=collate_2d,
)

qat_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Coco",
        data_path=data_path + "train_lmdb/",
        transforms=[
            dict(
                type="Resize",
                img_scale=(896, 896),
                keep_ratio=True,
            ),
            dict(
                type="Pad",
                size=(896, 896),
            ),
            dict(
                type="RandomFlip",
                px=0,
                py=0,
            ),
            dict(type="AugmentHSV", hgain=0.015, sgain=0.7, vgain=0.4),
            dict(
                type="ToTensor",
                to_yuv=True,
                use_yuv_v2=False,
            ),
            dict(
                type="Normalize",
                mean=128.0,
                std=128.0,
            ),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=1,
    pin_memory=True,
    collate_fn=collate_2d,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Coco",
        data_path=data_path + "val_lmdb/",
        transforms=[
            dict(
                type="Resize",
                img_scale=(896, 896),
                keep_ratio=True,
            ),
            dict(
                type="Pad",
                size=(896, 896),
            ),
            dict(
                type="ToTensor",
                to_yuv=True,
                use_yuv_v2=False,
            ),
            dict(
                type="Normalize",
                mean=128.0,
                std=128.0,
            ),
        ],
    ),
    batch_size=batch_size_per_gpu,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    shuffle=False,
    num_workers=1,
    pin_memory=True,
    collate_fn=collate_2d,
)

val_example_data_loader = copy.deepcopy(val_data_loader)
val_example_data_loader.pop("sampler")
val_example_data_loader["num_workers"] = 0

hbir_data_loader = copy.deepcopy(val_data_loader)
hbir_data_loader["batch_size"] = 1
hbir_data_loader["shuffle"] = False

calibration_data_loader = copy.deepcopy(qat_data_loader)
calibration_data_loader.pop("sampler")
calibration_data_loader["batch_size"] = batch_size_per_gpu * 4
calibration_data_loader["dataset"]["transforms"] = val_data_loader["dataset"][
    "transforms"
]

calibration_example_data_loader = copy.deepcopy(calibration_data_loader)
calibration_example_data_loader["num_workers"] = 0


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

hbm_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Coco",
        data_path=data_path + "val_lmdb/",
        transforms=[
            dict(
                type="Resize",
                img_scale=(896, 896),
                keep_ratio=True,
            ),
            dict(
                type="Pad",
                size=(896, 896),
            ),
            dict(
                type="ToTensor",
                to_yuv=False,
            ),
        ],
    ),
    batch_size=1,
    sampler=dict(type=torch.utils.data.DistributedSampler),
    shuffle=False,
    num_workers=1,
    pin_memory=True,
    collate_fn=collate_2d,
)
hbm_batch_processor = copy.deepcopy(val_batch_processor)

inference_dataset = dict(
    type="Coco",
    data_path=data_path + "val_lmdb/",
)
