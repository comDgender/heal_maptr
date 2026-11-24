"""Trainer definitions for MapTR HENet-tiny BEVFormer NuScenes.

Training stages: float -> calibration (no qat_trainer).
"""

import copy

import torch
from callback import (
    ckpt_callback,
    grad_callback,
    loss_show_update,
    val_callback,
    val_map_metric,
)
from common import (
    calibration_step,
    float_epochs,
    float_lr,
    pretrained_model_dir,
)
from data import (
    batch_processor,
    calibration_batch_processor,
    calibration_data_loader,
    data_loader,
    stat_callback,
)
from model import calib_model, model
from qconfig import (
    calibration_train_model_convert_pipeline,
    calibration_val_model_convert_pipeline,
)

calibration_val_callback = copy.deepcopy(val_callback)
calibration_val_callback["val_interval"] = 1
calibration_val_callback["val_on_train_end"] = False
calibration_val_callback["model_convert_pipeline"] = (
    calibration_val_model_convert_pipeline
)

calibration_ckpt_callback = copy.deepcopy(ckpt_callback)
calibration_ckpt_callback["save_interval"] = 1

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=pretrained_model_dir,
                allow_miss=True,
                ignore_extra=True,
                verbose=True,
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        params={"backbone": dict(lr_mult=0.1)},
        lr=float_lr,
        weight_decay=0.1,
    ),
    batch_processor=batch_processor,
    device=None,
    num_epochs=float_epochs,
    callbacks=[
        stat_callback,
        loss_show_update,
        grad_callback,
        dict(
            type="CosineAnnealingLrUpdater",
            warmup_len=500,
            warmup_by="step",
            warmup_lr_ratio=1.0 / 3,
            warmup_lr_begin2ratio=True,
            step_log_interval=500,
            stop_lr=3e-3 * float_lr,
        ),
        val_callback,
        ckpt_callback,
    ],
    sync_bn=True,
    train_metrics=dict(type="LossShow"),
    val_metrics=[val_map_metric],
)

calibration_trainer = dict(
    type="Calibrator",
    model=calib_model,
    model_convert_pipeline=calibration_train_model_convert_pipeline,
    data_loader=calibration_data_loader,
    batch_processor=calibration_batch_processor,
    num_steps=calibration_step,
    device=None,
    callbacks=[
        stat_callback,
        calibration_ckpt_callback,
        calibration_val_callback,
    ],
    val_metrics=[val_map_metric],
    log_interval=max(1, calibration_step // 10),
)
