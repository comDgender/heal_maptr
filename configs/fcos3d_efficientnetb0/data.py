import copy

import torch
from common import batch_size_per_gpu

from heal.data.collates import collate_2d

data_rootdir = "./tmp_data/heal_data/nuscenes/v1.0-trainval"
meta_rootdir = "./tmp_data/heal_data/nuscenes/meta"

train_transforms = [
    dict(
        type="Resize3D",
        img_scale=(896, 512),
        keep_ratio=True,
        interpolation="bilinear",
    ),
    dict(
        type="Pad",
        size=(512, 896),
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
]

val_transforms = [
    dict(
        type="Resize3D",
        img_scale=(896, 512),
        keep_ratio=True,
        interpolation="bilinear",
    ),
    dict(
        type="Pad",
        size=(512, 896),
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
]

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="NuscenesMonoDataset",
        data_path=data_rootdir + "/train_lmdb",
        transforms=train_transforms,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_2d,
)

qat_data_loader = copy.deepcopy(data_loader)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="NuscenesMonoDataset",
        data_path=data_rootdir + "/val_lmdb",
        transforms=val_transforms,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_2d,
)

calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader.pop("sampler")
calibration_data_loader["batch_size"] = batch_size_per_gpu * 4
calibration_data_loader["dataset"]["transforms"] = val_transforms

hbir_data_loader = copy.deepcopy(val_data_loader)
hbir_data_loader["batch_size"] = 1
hbir_data_loader["shuffle"] = False

inference_dataset = dict(
    type="NuscenesMonoDataset",
    data_path=data_rootdir + "/val_lmdb",
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

hbm_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="NuscenesMonoDataset",
        data_path=data_rootdir + "/val_lmdb",
        transforms=[
            dict(
                type="Resize3D",
                img_scale=(896, 512),
                keep_ratio=True,
                interpolation="bilinear",
            ),
            dict(
                type="Pad",
                size=(512, 896),
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

example_inputs = dict(
    img=torch.randn((1, 3, 512, 896)),
    cam2img=[
        [
            torch.tensor(  # noqa: E501
                [[709.2, 0.0, 457.1], [0.0, 1266.0, 491.0], [0.0, 0.0, 1.0]]
            )
        ]
    ],
)

deploy_inputs = dict(img=torch.randn((1, 3, 512, 896)))
