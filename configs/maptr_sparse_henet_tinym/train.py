"""Trainer definitions for sparse MapTR.

Training stages: float -> calibration (no qat_trainer).
"""

import copy
import os

import torch
from callback import (
    ckpt_callback,
    grad_callback,
    loss_show_update,
    stat_callback,
    val_callback,
    val_map_metric,
)
from common import (
    calibration_step,
    ckpt_dir,
    deploy_inputs,
    float_backbone_lr,
    float_lr,
    log_freq,
    num_steps,
    pretrained_model_dir,
)
from data import (
    batch_processor,
    calibration_batch_processor,
    calibration_data_loader,
    data_loader,
)
from model import model
from qconfig import cali_qconfig_setter

calibration_val_callback = copy.deepcopy(val_callback)
calibration_val_callback["val_interval"] = 1
calibration_val_callback["val_on_train_end"] = False
calibration_val_callback["model_convert_pipeline"] = dict(
    type="ModelConvertPipeline",
    qat_mode="with_bn",
    converters=[
        dict(
            type="Float2QAT",
            convert_mode="jit-strip",
            # Use deploy_inputs (has projection_mat) instead of a real batch —
            # SparseMapHead's compiler_model path requires that key.
            example_inputs=copy.deepcopy(deploy_inputs),
            qconfig_setter=cali_qconfig_setter,
        ),
    ],
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
                verbose=False,
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        params={"backbone": dict(lr=float_backbone_lr)},
        lr=float_lr,
        weight_decay=0.1,
    ),
    batch_processor=batch_processor,
    device=None,
    num_steps=num_steps,
    stop_by="step",
    callbacks=[
        stat_callback,
        loss_show_update,
        grad_callback,
        dict(
            type="CosineAnnealingLrUpdater",
            warmup_len=500,
            warmup_by="step",
            warmup_lr_ratio=1.0 / 3,
            step_log_interval=log_freq,
            update_by="step",
            min_lr_ratio=1e-3,
        ),
        val_callback,
        ckpt_callback,
    ],
    sync_bn=True,
    train_metrics=dict(type="LossShow"),
    val_metrics=[val_map_metric],
)

# Calibration: collect activation stats and save quantized scales/weights.
calibration_trainer = dict(
    type="Calibrator",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="with_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
                ignore_extra=True,
                allow_miss=True,
                verbose=True,
                check_hash=False,
            ),
            dict(
                type="Float2Calibration",
                convert_mode="jit-strip",
                # Same as val convert: deploy_inputs carries projection_mat
                # required by SparseMapHead when preparing_qat=True.
                example_inputs=copy.deepcopy(deploy_inputs),
                qconfig_setter=cali_qconfig_setter,
            ),
            dict(type="FixWeightQScale"),
        ],
    ),
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
