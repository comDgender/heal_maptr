"""Trainer definitions."""

import copy

import torch
from callback import (
    ckpt_callback,
    grad_callback,
    loss_show_update,
    stat_callback,
    val_callback,
    val_nuscenes_metric,
)
from common import calibration_step, float_epochs, float_lr, qat_epochs, qat_lr
from data import (
    batch_processor,
    calibration_batch_processor,
    calibration_data_loader,
    data_loader,
)
from model import deploy_model, model
from qconfig import (
    calibration_train_model_convert_pipeline,
    calibration_val_model_convert_pipeline,
    qat_train_model_convert_pipeline,
    qat_val_model_convert_pipeline,
)

calibration_val_callback = copy.deepcopy(val_callback)
calibration_val_callback["val_on_train_end"] = True
calibration_val_callback["model_convert_pipeline"] = (
    calibration_val_model_convert_pipeline
)

calibration_ckpt_callback = copy.deepcopy(ckpt_callback)
calibration_ckpt_callback["save_interval"] = 1

qat_val_callback = copy.deepcopy(val_callback)
qat_val_callback["val_interval"] = 1
qat_val_callback["model_convert_pipeline"] = qat_val_model_convert_pipeline

qat_ckpt_callback = copy.deepcopy(ckpt_callback)
qat_ckpt_callback["save_interval"] = 1

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=(
                    "./tmp_pretrained_models/resnet50/"
                    "float-checkpoint-best.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        params={"backbone": dict(lr_mult=0.1)},
        lr=float_lr,
        weight_decay=0.01,
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
            step_log_interval=500,
            stop_lr=2e-4 * 1e-3,
        ),
        val_callback,
        ckpt_callback,
    ],
    sync_bn=True,
    train_metrics=dict(type="LossShow"),
    val_metrics=[val_nuscenes_metric],
)

calibration_trainer = dict(
    type="Calibrator",
    model=model,
    skip_step=2,
    model_convert_pipeline=calibration_train_model_convert_pipeline,
    data_loader=calibration_data_loader,
    batch_processor=calibration_batch_processor,
    num_steps=calibration_step,
    device=None,
    callbacks=[
        stat_callback,
        calibration_val_callback,
        calibration_ckpt_callback,
    ],
    val_metrics=[val_nuscenes_metric],
    log_interval=calibration_step / 10,
)

qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=qat_train_model_convert_pipeline,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        lr=qat_lr,
        weight_decay=0.01,
    ),
    batch_processor=batch_processor,
    device=None,
    num_epochs=qat_epochs,
    callbacks=[
        stat_callback,
        loss_show_update,
        grad_callback,
        dict(
            type="StepDecayLrUpdater", lr_decay_id=[5], step_log_interval=500
        ),
        qat_val_callback,
        qat_ckpt_callback,
    ],
    sync_bn=True,
    train_metrics=dict(type="LossShow"),
    val_metrics=[val_nuscenes_metric],
)

int_infer_trainer = dict(
    type="Trainer",
    model=deploy_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(type="Float2QAT", convert_mode="jit-strip"),
            dict(type="QAT2Quantize", convert_mode="jit-strip"),
        ],
    ),
    data_loader=None,
    optimizer=None,
    batch_processor=None,
    num_epochs=0,
    device=None,
)
