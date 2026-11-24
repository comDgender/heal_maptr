"""Trainer definitions."""

import copy
import os

import torch
from callback import (
    ckpt_callback,
    grad_callback,
    loss_show_update,
    stat_callback,
    val_callback,
    val_nuscenes_metric,
)
from common import (
    calibration_step,
    ckpt_dir,
    convert_mode,
    float_epochs,
    qat_epochs,
    qat_mode,
)
from data import (
    batch_processor,
    calibration_batch_processor,
    calibration_data_loader,
    data_loader,
)
from model import example_inputs, model
from qconfig import calib_qconfig_setter, qat_qconfig_setter

calibration_val_callback = copy.deepcopy(val_callback)
calibration_val_callback["val_on_train_end"] = True

calibration_ckpt_callback = copy.deepcopy(ckpt_callback)
calibration_ckpt_callback["save_interval"] = 1

qat_val_callback = copy.deepcopy(val_callback)
qat_val_callback["val_interval"] = 1

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        betas=(0.95, 0.99),
        lr=2e-4,
        weight_decay=0.01,
    ),
    batch_processor=batch_processor,
    num_epochs=float_epochs,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(
            type="CyclicLrUpdater",
            target_ratio=(10, 1e-4),
            cyclic_times=1,
            step_ratio_up=0.4,
            step_log_interval=200,
        ),
        grad_callback,
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
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode=qat_mode,
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-last.pth.tar"
                ),
                allow_miss=True,
                verbose=True,
            ),
            dict(
                type="Float2Calibration",
                convert_mode=convert_mode,
                example_inputs=example_inputs,
                qconfig_setter=calib_qconfig_setter,
            ),
        ],
    ),
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
    log_interval=max(1, calibration_step // 10),
)

qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode=qat_mode,
        qconfig_params=dict(
            activation_qat_qkwargs=dict(averaging_constant=0),
            weight_qat_qkwargs=dict(averaging_constant=1),
        ),
        converters=[
            dict(
                type="Float2QAT",
                convert_mode=convert_mode,
                example_inputs=example_inputs,
                qconfig_setter=qat_qconfig_setter,
            ),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-last.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        weight_decay=0.0,
        lr=2e-4,
        momentum=0.9,
    ),
    batch_processor=batch_processor,
    num_epochs=qat_epochs,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(
            type="CyclicLrUpdater",
            target_ratio=(10, 1e-4),
            cyclic_times=1,
            step_ratio_up=0.4,
            step_log_interval=200,
        ),
        grad_callback,
        qat_val_callback,
        ckpt_callback,
    ],
    sync_bn=True,
    train_metrics=dict(type="LossShow"),
    val_metrics=[val_nuscenes_metric],
)
