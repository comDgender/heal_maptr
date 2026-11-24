import copy

import torch
from hat_engine.engine.processors.loss_collector import collect_loss_by_index

batch_size_per_gpu = 64
data_path = "./tmp_data/heal_data/imagenet/"

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="ImageNet",
        data_path=data_path + "train_lmdb/",
        transforms=[
            dict(
                type="TorchVisionAdapter",
                interface="RandomResizedCrop",
                size=224,
                scale=(0.08, 1.0),
                ratio=(3.0 / 4.0, 4.0 / 3.0),
            ),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=5,
    pin_memory=True,
)
val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="ImageNet",
        data_path=data_path + "val_lmdb/",
        transforms=[
            dict(type="TorchVisionAdapter", interface="Resize", size=256),
            dict(type="TorchVisionAdapter", interface="CenterCrop", size=224),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=5,
    pin_memory=True,
)

val_example_data_loader = copy.deepcopy(val_data_loader)
val_example_data_loader.pop("sampler")
val_example_data_loader["num_workers"] = 0

calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader.pop("sampler")
calibration_data_loader["batch_size"] = batch_size_per_gpu * 4

calibration_example_data_loader = copy.deepcopy(calibration_data_loader)
calibration_example_data_loader["num_workers"] = 0

batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    batch_transforms=[
        dict(type="TorchVisionAdapter", interface="RandomHorizontalFlip"),
        dict(
            type="TorchVisionAdapter",
            interface="ColorJitter",
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1,
        ),
        dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
    loss_collector=collect_loss_by_index(1),
)
val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    batch_transforms=[
        dict(type="BgrToYuv444", rgb_input=True),
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=128.0,
            std=128.0,
        ),
    ],
    loss_collector=None,
)

hbir_data_loader = copy.deepcopy(val_data_loader)
hbir_data_loader["batch_size"] = 1
hbir_data_loader["shuffle"] = False

hbm_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="ImageNet",
        data_path=data_path + "val_lmdb/",
        out_pil=True,
        transforms=[
            dict(
                type="TorchVisionAdapter",
                interface="Resize",
                size=256,
                interpolation=3,
            ),
            dict(type="TorchVisionAdapter", interface="CenterCrop", size=224),
            dict(type="TorchVisionAdapter", interface="ToTensor"),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
)

hbm_batch_processor = dict(
    type="BasicBatchProcessor",
    batch_transforms=[
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=0.0,
            std=1 / 255.0,
        ),
    ],
    need_grad_update=False,
    loss_collector=None,
)

inference_dataset = dict(
    type="ImageNet",
    data_path=data_path + "val_lmdb/",
    out_pil=True,
)
