import copy

import torch
from common import mixup, num_classes
from hat_engine.engine.processors.loss_collector import collect_loss_by_index

batch_size_per_gpu = 128
data_path = "./tmp_data/heal_data/imagenet/"


image_shape = (3, 224, 224)
auto_augment = True

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="ImageNet",
        data_path=data_path + "train_lmdb/",
        out_pil=True,
        transforms=[
            dict(
                type="TimmTransforms",
                input_size=image_shape[-1],
                is_training=True,
                color_jitter=0.4,
                auto_augment="rand-m9-mstd0.5-inc1" if auto_augment else None,
                re_prob=0.25,
                re_mode="pixel",
                re_count=1,
                mean=[0, 0, 0],
                std=[1.0, 1.0, 1.0],
                interpolation="bicubic",
            ),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    drop_last=True,
)
val_data_loader = dict(
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
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
)


val_example_data_loader = copy.deepcopy(val_data_loader)
val_example_data_loader.pop("sampler")
val_example_data_loader["num_workers"] = 0

calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader.pop("sampler")
calibration_data_loader["batch_size"] = batch_size_per_gpu * 2

calibration_example_data_loader = copy.deepcopy(calibration_data_loader)
calibration_example_data_loader["num_workers"] = 0


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


inference_dataset = dict(
    type="ImageNet",
    data_path=data_path + "val_lmdb/",
    out_pil=True,
)


batch_transforms = [
    dict(
        type="TorchVisionAdapter",
        interface="Normalize",
        mean=0.0,
        std=1 / 255.0,
    ),
    dict(type="BgrToYuv444", rgb_input=True),
    dict(
        type="TorchVisionAdapter",
        interface="Normalize",
        mean=128.0,
        std=128.0,
    ),
]
if mixup:
    batch_transforms.append(
        dict(
            type="TimmMixup",
            mixup_alpha=0.8,
            cutmix_alpha=1.0,
            cutmix_minmax=None,
            prob=1.0,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=0.1,
            num_classes=num_classes,
        ),
    )
batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    batch_transforms=batch_transforms,
    loss_collector=collect_loss_by_index(1),
)

val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    loss_collector=None,
    batch_transforms=[
        dict(
            type="TorchVisionAdapter",
            interface="Normalize",
            mean=0.0,
            std=1 / 255.0,
        ),
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
