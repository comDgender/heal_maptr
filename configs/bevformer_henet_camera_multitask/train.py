"""Trainer definitions for BevFormer HENet camera-only det+occ pretrain."""

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
from common import float_epochs, float_lr, pretrain_ckpt_dir
from data import batch_processor, data_loader
from hat_engine.utils.checkpoint import update_state_dict_by_add_prefix
from model import model

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
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
        },
        lr=float_lr,
        weight_decay=0.01,
    ),
    data_loader=data_loader,
    batch_processor=batch_processor,
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
