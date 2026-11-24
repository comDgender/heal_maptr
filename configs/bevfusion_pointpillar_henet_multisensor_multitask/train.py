"""Trainer definitions for BevFusion multisensor multitask."""

import os
from functools import partial

import torch
from callback import (
    ckpt_callback,
    grad_callback,
    loss_show_update,
    stat_callback,
    val_callback,
    val_metrics,
)
from common import (
    calibration_step,
    camera_ckpt_dir,
    float_epochs,
    float_lr,
    lidar_ckpt_dir,
)
from data import (
    batch_processor,
    calibration_batch_processor,
    calibration_data_loader,
    data_loader,
)
from hat_engine.utils.checkpoint import update_state_dict_by_add_prefix
from model import model
from qconfig import calibration_train_model_convert_pipeline

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    camera_ckpt_dir, "float-checkpoint-last.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
                verbose=True,
            ),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    lidar_ckpt_dir, "float-checkpoint-last.pth.tar"
                ),
                state_dict_update_func=partial(
                    update_state_dict_by_add_prefix, prefix="lidar_net."
                ),
                allow_miss=True,
                ignore_extra=True,
                verbose=True,
            ),
        ],
    ),
    optimizer=dict(
        type=torch.optim.AdamW,
        betas=(0.95, 0.99),
        params={
            "camera_net.backbone": dict(lr_mult=0.1),
        },
        lr=float_lr,
        weight_decay=0.01,
    ),
    batch_processor=batch_processor,
    data_loader=data_loader,
    num_epochs=float_epochs,
    stop_by="epoch",
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(
            type="CosineAnnealingLrUpdater",
            warmup_len=500,
            warmup_by="step",
            warmup_lr_ratio=1.0 / 3,
            step_log_interval=500,
            stop_lr=2e-4 * 1e-3,
        ),
        grad_callback,
        val_callback,
        ckpt_callback,
    ],
    sync_bn=True,
    train_metrics=dict(type="LossShow"),
    val_metrics=val_metrics,
)

calibration_trainer = dict(
    type="Calibrator",
    model=model,
    model_convert_pipeline=calibration_train_model_convert_pipeline,
    data_loader=calibration_data_loader,
    batch_processor=calibration_batch_processor,
    num_steps=calibration_step,
    device=None,
    callbacks=[
        stat_callback,
        val_callback,
        ckpt_callback,
    ],
    val_metrics=val_metrics,
    log_interval=20,
)
