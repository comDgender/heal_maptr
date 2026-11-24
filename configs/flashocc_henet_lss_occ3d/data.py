"""Data and batch-processing config for FlashOcc HENet LSS NuScenes."""

import copy
import os

import torch
from common import (
    base_dataset_type,
    batch_size_per_gpu,
    data_rootdir,
    data_shape,
    occ_dataset_type,
    orig_shape,
    resize_shape,
    step_log_freq,
    task_name,
    val_batch_size_per_gpu,
)
from hat_engine.engine.processors.loss_collector import collect_loss_by_index

from heal.data.collates.nusc_collates import collate_nuscenes

bda_aug_conf = dict(
    rot_lim=(-0.0, 0.0),
    scale_lim=(1.0, 1.0),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
)

scale = float(resize_shape[2]) / float(orig_shape[2])
resize_aug = (-0.06, 0.11)

base_train_path = os.path.join(data_rootdir, "train_lmdb")
base_val_path = os.path.join(data_rootdir, "val_lmdb")
occ_train_path = os.path.join(data_rootdir, "train_occ_lmdb")
occ_val_path = os.path.join(data_rootdir, "val_occ_lmdb")


train_transforms = [
    dict(
        type="MultiViewsImgResize",
        scales=tuple(x + scale for x in resize_aug),
    ),
    dict(type="MultiViewsImgCrop", size=data_shape[1:]),
    dict(type="MultiViewsImgFlip", prob=0.5),
    dict(type="MultiViewsImgRotate", rot=(-5.4, 5.4)),
    dict(
        type="BevFeatureAug",
        bda_aug_conf=bda_aug_conf,
        is_train=True,
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

test_transforms = [
    dict(type="MultiViewsImgResize", size=resize_shape[1:]),
    dict(type="MultiViewsImgCrop", size=data_shape[1:]),
    dict(
        type="MultiViewsImgTransformWrapper",
        transforms=[
            dict(type="PILToTensor"),
            dict(type="BgrToYuv444", rgb_input=True),
            dict(type="Normalize", mean=128.0, std=128.0),
        ],
    ),
]

bup_transforms = [
    dict(type="MultiViewsImgResize", size=resize_shape[1:]),
    dict(type="MultiViewsImgCrop", size=data_shape[1:]),
]

hbm_transforms = [
    dict(type="MultiViewsImgResize", size=resize_shape[1:]),
    dict(type="MultiViewsImgCrop", size=data_shape[1:]),
    dict(
        type="MultiViewsImgTransformWrapper",
        transforms=[
            dict(type="PILToTensor"),
        ],
    ),
]


def _merge_dataset(base_path, occ_path, transforms):
    """flashocc sample = base camera dataset + occ dataset, joined by token."""
    return dict(
        type="MergeDataset",
        join_key="sample_token",
        transforms=transforms,
        datasets=[
            dict(type=base_dataset_type, data_path=base_path),
            dict(type=occ_dataset_type, data_path=occ_path),
        ],
    )


data = dict(
    train=_merge_dataset(base_train_path, occ_train_path, train_transforms),
    val=_merge_dataset(base_val_path, occ_val_path, test_transforms),
    hbm_val=_merge_dataset(base_val_path, occ_val_path, hbm_transforms),
)

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=data["train"],
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=2,
    collate_fn=collate_nuscenes,
    pin_memory=True,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=data["val"],
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=val_batch_size_per_gpu,
    shuffle=False,
    num_workers=2,
    collate_fn=collate_nuscenes,
    pin_memory=True,
)

calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader.pop("sampler")
calibration_data_loader["batch_size"] = batch_size_per_gpu * 4

calibration_example_data_loader = copy.deepcopy(calibration_data_loader)
calibration_example_data_loader["num_workers"] = 0

val_example_data_loader = copy.deepcopy(val_data_loader)
val_example_data_loader.pop("sampler")
val_example_data_loader["num_workers"] = 0

int_infer_data_loader = copy.deepcopy(val_data_loader)
int_infer_data_loader["batch_size"] = 1
int_infer_data_loader["shuffle"] = False

hbir_data_loader = copy.deepcopy(val_data_loader)
hbir_data_loader["batch_size"] = 1
hbir_data_loader["shuffle"] = False

hbm_data_loader = copy.deepcopy(val_data_loader)
hbm_data_loader["batch_size"] = 1
hbm_data_loader["shuffle"] = False
hbm_data_loader["dataset"] = data["hbm_val"]

batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=True,
    loss_collector=collect_loss_by_index(1),
    enable_amp=True,
    enable_amp_dtype=torch.float16,
)

val_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    loss_collector=None,
)


def update_loss(metrics, batch, model_out):
    for metric in metrics:
        metric.update(model_out[1])


def val_update_metric_func(metrics, batch, model_outs):
    gt_semantics = batch["gt_occ_info"]["voxel_semantics"][0].squeeze()
    mask_camera = batch["gt_occ_info"]["mask_camera"][0].squeeze()
    semantics_pred = model_outs[1]["occ_pre"].squeeze()

    masked_semantics_gt = gt_semantics[mask_camera]
    masked_semantics_pred = semantics_pred[mask_camera]

    results = {
        "label": masked_semantics_gt.reshape(-1),
        "preds": masked_semantics_pred.reshape(-1),
    }
    for metric in metrics:
        metric.update(**results)


metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_loss,
    step_log_freq=step_log_freq,
    epoch_log_freq=1,
    log_prefix=task_name,
)

val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=val_update_metric_func,
    step_log_freq=1000000,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)

single_infer_dataset = copy.deepcopy(int_infer_data_loader["dataset"])
single_infer_dataset["transforms"] = None
