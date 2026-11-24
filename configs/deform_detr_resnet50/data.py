import copy

import torch
from common import batch_size_per_gpu, data_path, img_shape

from heal.data.collates.collates import collate_2d, collate_2d_pad

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Coco",
        data_path=data_path + "train_lmdb/",
        transforms=[
            dict(
                type="RandomFlip",
                px=0.5,
                py=0,
            ),
            dict(
                type="Resize",
                img_scale=(800, 1332),
                ratio_range=(0.5, 1.5),
                keep_ratio=True,
            ),
            dict(
                type="RandomCrop",
                size=(800, 1332),
                filter_area=False,
            ),
            dict(
                type="Pad",
                size=(800, 1332),
            ),
            dict(
                type="ToTensor",
                to_yuv=True,
                use_yuv_v2=False,
            ),
            dict(
                type="Normalize",
                mean=128,
                std=128,
            ),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_2d_pad,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Coco",
        data_path=data_path + "val_lmdb/",
        transforms=[
            dict(
                type="Resize",
                img_scale=[(800, 1332)],
                multiscale_mode="value",
                pad_to_keep_ratio=True,
            ),
            dict(
                type="ToTensor",
                to_yuv=True,
                use_yuv_v2=False,
            ),
            dict(
                type="Normalize",
                mean=128,
                std=128,
            ),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_2d,
)

val_example_data_loader = copy.deepcopy(val_data_loader)
val_example_data_loader.pop("sampler", None)
val_example_data_loader["num_workers"] = 0

calibration_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Coco",
        data_path=data_path + "train_lmdb/",
        transforms=[
            dict(
                type="Resize",
                img_scale=[(800, 1332)],
                multiscale_mode="value",
                pad_to_keep_ratio=True,
            ),
            dict(
                type="ToTensor",
                to_yuv=True,
                use_yuv_v2=False,
            ),
            dict(
                type="Normalize",
                mean=128,
                std=128,
            ),
        ],
    ),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_2d,
)

calibration_example_data_loader = copy.deepcopy(calibration_data_loader)
calibration_example_data_loader["num_workers"] = 0

qat_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Coco",
        data_path=data_path + "train_lmdb/",
        transforms=[
            dict(
                type="Resize",
                img_scale=[(800, 1332)],
                multiscale_mode="value",
                pad_to_keep_ratio=True,
            ),
            dict(
                type="ToTensor",
                to_yuv=True,
                use_yuv_v2=False,
            ),
            dict(
                type="Normalize",
                mean=128,
                std=128,
            ),
        ],
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_2d,
)

hbir_data_loader = copy.deepcopy(val_data_loader)
hbir_data_loader["batch_size"] = 1
hbir_data_loader["shuffle"] = False

hbm_data_loader = copy.deepcopy(val_data_loader)
hbm_data_loader["batch_size"] = 1
hbm_data_loader["shuffle"] = False
hbm_data_loader["dataset"]["transforms"] = [
    dict(
        type="Resize",
        img_scale=[(800, 1332)],
        multiscale_mode="value",
        pad_to_keep_ratio=True,
    ),
    dict(
        type="ToTensor",
        to_yuv=False,
        use_yuv_v2=False,
    ),
]

inference_dataset = dict(
    type="Coco",
    data_path=data_path + "val_lmdb/",
)


def loss_collector(outputs: dict):
    losses = []
    for _, loss in outputs.items():
        losses.append(loss)
    return losses


def update_loss(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


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

calibration_batch_processor = copy.deepcopy(val_batch_processor)

hbm_batch_processor = copy.deepcopy(val_batch_processor)


def update_metric(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


deploy_inputs = dict(
    img=torch.randn((1, 3, img_shape[0], img_shape[1])),
)

example_inputs = dict(
    img=torch.randn((1, 3, img_shape[0], img_shape[1])),
    scale_factor=torch.tensor([[1.0, 1.0, 1.0, 1.0]]),
    resized_shape=[(800, 1332, 3)],
    img_name=["test"],
    img_id=[1],
)
