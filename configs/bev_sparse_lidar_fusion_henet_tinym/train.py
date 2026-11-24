"""Trainer definitions for SparseBEV lidar fusion HENet-tiny NuScenes."""

import copy
import os
from functools import partial

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
    float_lr,
    num_steps,
    pretrain_ckpt_dir,
    qat_lr,
)
from data import (
    batch_processor,
    calibration_batch_processor,
    calibration_data_loader,
    data_loader,
)
from hat_engine.utils.checkpoint import update_state_dict_by_add_prefix
from model import model
from qconfig import (
    calibration_train_model_convert_pipeline,
    qat_train_model_convert_pipeline,
    qat_val_model_convert_pipeline,
)

_pretrain_ckpt = os.path.join(
    pretrain_ckpt_dir, "float-checkpoint-best.pth.tar"
)

qat_model = copy.deepcopy(model)
qat_model["head"]["enable_dn"] = False

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=_pretrain_ckpt,
                allow_miss=True,
                ignore_extra=True,
                ignore_tensor_shape=True,
                verbose=True,
            ),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=_pretrain_ckpt,
                state_dict_update_func=partial(
                    update_state_dict_by_add_prefix, prefix="lidar_net."
                ),
                allow_miss=True,
                ignore_extra=True,
                ignore_tensor_shape=True,
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

_calibration_trainer_common = dict(
    type="Calibrator",
    model=model,
    data_loader=calibration_data_loader,
    batch_processor=calibration_batch_processor,
    num_steps=calibration_step,
    device=None,
    callbacks=[val_callback, ckpt_callback],
    log_interval=max(1, calibration_step // 10),
    val_metrics=[val_nuscenes_metric],
)

calibration_trainer = dict(
    **_calibration_trainer_common,
    model_convert_pipeline=calibration_train_model_convert_pipeline,
)

qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=qat_model,
    model_convert_pipeline=qat_train_model_convert_pipeline,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        eps=1e-8,
        betas=(0.9, 0.999),
        params={"backbone": dict(lr=qat_lr)},
        lr=qat_lr,
        weight_decay=1e-3,
    ),
    batch_processor=batch_processor,
    num_steps=int(num_steps * 0.1),
    stop_by="step",
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        grad_callback,
        dict(
            type="StepDecayLrUpdater",
            lr_decay_id=[int(num_steps * 0.1 * 0.4)],
            step_log_interval=100,
        ),
        val_callback,
        ckpt_callback,
    ],
    train_metrics=dict(type="LossShow"),
    val_metrics=[val_nuscenes_metric],
)

qat_val_callback = copy.deepcopy(val_callback)
qat_val_callback["val_interval"] = 100
qat_val_callback["model_convert_pipeline"] = qat_val_model_convert_pipeline
