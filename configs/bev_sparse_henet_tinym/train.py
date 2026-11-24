"""Trainer definitions for SparseBEV HENet-tiny NuScenes."""

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
from common import calibration_step, float_lr, num_steps
from data import (
    batch_processor,
    calibration_batch_processor,
    calibration_data_loader,
    data_loader,
)
from model import model, test_model
from qconfig import (
    calibration_train_model_convert_pipeline,
    calibration_val_model_convert_pipeline,
)

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=(
                    "./tmp_pretrained_models/henet_tinym/"
                    "float-checkpoint-best.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
                verbose=True,
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        params={"backbone": dict(lr=float_lr)},
        eps=1e-8,
        betas=(0.9, 0.999),
        lr=float_lr,
        weight_decay=0.001,
    ),
    batch_processor=batch_processor,
    num_steps=num_steps,
    stop_by="step",
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(type="ExponentialMovingAverage"),
        grad_callback,
        dict(
            type="CosineAnnealingLrUpdater",
            warmup_len=500,
            warmup_by="step",
            warmup_lr_ratio=1.0 / 3,
            step_log_interval=500,
            update_by="step",
            min_lr_ratio=1e-3,
        ),
        val_callback,
        ckpt_callback,
    ],
    train_metrics=dict(type="LossShow"),
    sync_bn=True,
    val_metrics=[val_nuscenes_metric],
)

# Calibration validation uses Pattern A (cf.
# bevformer_tiny_resnet50_detection): a float `test_model`
# (enable_dn=False) is converted once to a calibration quantized graph via
# `calibration_val_model_convert_pipeline` (Float2Calibration) at callback
# init, then each validation deepcopies that graph and loads the *current*
# calibration training weights (init_with_train_model=True) — including the
# observer scales produced during calibration. This evaluates the actual
# calibration product instead of rebuilding from the float checkpoint.
calibration_val_callback = copy.deepcopy(val_callback)
calibration_val_callback["val_model"] = test_model
calibration_val_callback["init_with_train_model"] = True
calibration_val_callback["model_convert_pipeline"] = (
    calibration_val_model_convert_pipeline
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
    callbacks=[calibration_val_callback, ckpt_callback],
    log_interval=max(1, calibration_step // 10),
    val_metrics=[val_nuscenes_metric],
)
