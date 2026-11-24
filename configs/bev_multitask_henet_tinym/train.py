# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import os
from functools import partial

import torch
from callback import (
    ckpt_callback,
    freeze_model_callback,
    grad_callback,
    loss_show_update,
    stage1_val_metrics,
    stage2_val_metrics,
    stage3_val_metrics,
    stat_callback,
    val_callback,
    val_metrics,
)
from common import (
    ckpt_dir,
    float_lr_stage1,
    float_lr_stage2,
    float_lr_stage3,
    num_steps_stage1,
    num_steps_stage2,
    num_steps_stage3,
    pretrain_ckpt_dir,
)
from data import (
    batch_processor,
    calibration_batch_processor,
    calibration_data_loader,
    data_loader,
)
from hat_engine.utils.checkpoint import update_state_dict_by_add_prefix
from model import model, model_stage1, model_stage2, model_stage3
from qconfig import calibration_train_model_convert_pipeline

# stage1: dynamic + static + occ + discrete map
float_stage1_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model_stage1,
    data_loader=data_loader,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    pretrain_ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
                state_dict_update_func=partial(
                    update_state_dict_by_add_prefix, prefix="camera_net."
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
            "task_decoders.dynamic_objects_det": dict(lr_mult=2),
            "task_decoders.occ": dict(lr_mult=2),
            "task_decoders.discrete_map": dict(lr_mult=2),
        },
        lr=float_lr_stage1,
        weight_decay=0.1,
    ),
    batch_processor=batch_processor,
    num_steps=num_steps_stage1,
    stop_by="step",
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
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=stage1_val_metrics,
)


# stage2: maptr only (frozen backbone/neck/view_transformer/fuse)
float_stage2_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model_stage2,
    data_loader=data_loader,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float_stage1-checkpoint-last.pth.tar"
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
        lr=float_lr_stage2,
        weight_decay=0.1,
    ),
    batch_processor=batch_processor,
    num_steps=num_steps_stage2,
    stop_by="step",
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
        freeze_model_callback,
    ],
    sync_bn=True,
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=stage2_val_metrics,
)


# stage3: full tasks
float_stage3_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model_stage3,
    data_loader=data_loader,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float_stage1-checkpoint-last.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
                verbose=True,
            ),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float_stage2-checkpoint-last.pth.tar"
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
            "camera_net.neck": dict(lr_mult=0.5),
            "camera_net.view_transformer": dict(lr_mult=0.5),
            "fuse_module": dict(lr_mult=0.5),
            "task_decoders.dynamic_objects_det": dict(lr_mult=0.1),
            "task_decoders.occ": dict(lr_mult=0.1),
            "task_decoders.static_objects_det": dict(lr_mult=0.1),
            "task_decoders.discrete_map": dict(lr_mult=0.1),
            "task_decoders.maptr": dict(lr_mult=3),
        },
        lr=float_lr_stage3,
        weight_decay=0.1,
    ),
    batch_processor=batch_processor,
    num_steps=num_steps_stage3,
    stop_by="step",
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
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=stage3_val_metrics,
)


# --- calibration ---
calibration_trainer = dict(
    type="Calibrator",
    model=model,
    model_convert_pipeline=calibration_train_model_convert_pipeline,
    data_loader=calibration_data_loader,
    batch_processor=calibration_batch_processor,
    skip_step=2,
    num_steps=20,
    device=None,
    callbacks=[
        stat_callback,
        val_callback,
        ckpt_callback,
    ],
    val_metrics=val_metrics,
    log_interval=20,
)
