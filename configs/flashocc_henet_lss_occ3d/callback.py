"""Callbacks for FlashOcc HENet LSS Occ3D NuScenes."""

import copy

from common import ckpt_dir, occ3d_seg_class, training_step, val_log_interval
from data import (
    metric_updater,
    val_batch_processor,
    val_data_loader,
    val_metric_updater,
)
from model import model

stat_callback = dict(
    type="StatsMonitor",
    log_freq=500,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,
    mode="max",
)

grad_callback = dict(
    type="GradScale",
    module_and_scale=[],
    clip_grad_norm=35,
    clip_norm_type=2,
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    val_interval=val_log_interval,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=model,
    val_on_train_end=True,
)

calibration_ckpt_callback = copy.deepcopy(ckpt_callback)
calibration_ckpt_callback["save_interval"] = 1

calibration_val_callback = copy.deepcopy(val_callback)
calibration_val_callback["val_interval"] = 1
calibration_val_callback["val_on_train_end"] = False

val_metrics = dict(
    type="MeanIOU",
    seg_class=occ3d_seg_class,
    ignore_index=17,
)

bpu_eval_metric = dict(
    type="MeanIOU",
    seg_class=occ3d_seg_class,
    ignore_index=17,
)

__all__ = [
    "stat_callback",
    "ckpt_callback",
    "grad_callback",
    "val_callback",
    "calibration_ckpt_callback",
    "calibration_val_callback",
    "val_metrics",
    "bpu_eval_metric",
    "metric_updater",
]
