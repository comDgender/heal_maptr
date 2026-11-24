"""Training entry configuration for FlashOcc HENet LSS Occ3D NuScenes."""

import torch
from callback import (
    calibration_ckpt_callback,
    calibration_val_callback,
    ckpt_callback,
    grad_callback,
    metric_updater,
    stat_callback,
    val_callback,
    val_metrics,
)
from common import num_epochs
from data import batch_processor, calibration_data_loader, data_loader
from model import model
from qconfig import calibration_train_model_convert_pipeline

calibration_step = 100
calibration_batch_processor = dict(
    type="BasicBatchProcessor",
    need_grad_update=False,
    loss_collector=None,
)

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=(
                    "./tmp_pretrained_models/henet_tinym/"
                    "float-checkpoint-best.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
                ignore_tensor_shape=True,
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        lr=1e-4,
        weight_decay=1e-2,
    ),
    batch_processor=batch_processor,
    num_epochs=num_epochs,
    device=None,
    callbacks=[
        stat_callback,
        grad_callback,
        dict(
            type="CosineAnnealingLrUpdater",
            warmup_len=500,
            warmup_by="step",
            warmup_lr_ratio=1.0 / 3,
            step_log_interval=500,
            stop_lr=2e-4 * 1e-3,
        ),
        metric_updater,
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
        calibration_val_callback,
        calibration_ckpt_callback,
    ],
    val_metrics=val_metrics,
    log_interval=calibration_step / 10,
)
