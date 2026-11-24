import os

import torch
import torch.nn as nn
from callback import (
    ckpt_callback,
    loss_show_update,
    stat_callback,
    val_callback,
    val_metrics,
)
from common import ckpt_dir, qat_mode
from data import (
    batch_processor,
    calibration_batch_processor,
    calibration_data_loader,
    calibration_step,
    data_loader,
    qat_data_loader,
)
from model import cali_model, model
from qconfig import float2calibration, float2qat

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type="custom_param_optimizer",
        optim_cls=torch.optim.AdamW,
        optim_cfgs=dict(lr=5e-4, weight_decay=1e-4),
        custom_param_mapper={
            "bias": dict(weight_decay=0.0),
            "norm_types": dict(weight_decay=0.0),
            nn.Embedding: dict(weight_decay=0.0),
        },
    ),
    batch_processor=batch_processor,
    num_epochs=75,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(
            type="CosLrUpdater",
            warmup_len=1,
            warmup_by="epoch",
            step_log_interval=500,
        ),
        val_callback,
        ckpt_callback,
    ],
    sync_bn=True,
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=val_metrics,
)


calibration_trainer = dict(
    type="Calibrator",
    model=cali_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode=qat_mode,
        qconfig_params=dict(
            activation_calibration_observer="mse",
        ),
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-last.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
                verbose=True,
            ),
            float2calibration,
        ],
    ),
    data_loader=calibration_data_loader,
    batch_processor=calibration_batch_processor,
    num_steps=calibration_step,
    device=None,
    callbacks=[
        ckpt_callback,
        val_callback,
    ],
    log_interval=calibration_step / 10,
    val_metrics=val_metrics,
)


qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=cali_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode=qat_mode,
        converters=[
            float2qat,
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-last.pth.tar"
                ),
                verbose=True,
                allow_miss=True,
                ignore_extra=True,
            ),
        ],
    ),
    data_loader=qat_data_loader,
    optimizer=dict(
        type="custom_param_optimizer",
        optim_cls=torch.optim.AdamW,
        optim_cfgs=dict(lr=8 * 5e-6, weight_decay=1e-4),  # gpu: 8*4090
        custom_param_mapper={
            "bias": dict(weight_decay=0.0),
            "norm_types": dict(weight_decay=0.0),
            nn.Embedding: dict(weight_decay=0.0),
        },
    ),
    batch_processor=batch_processor,
    num_epochs=15,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(
            type="StepDecayLrUpdater",
            lr_decay_id=[4],
            step_log_interval=500,
        ),
        val_callback,
        ckpt_callback,
    ],
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=val_metrics,
)
