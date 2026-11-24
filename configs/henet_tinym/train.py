import copy

import torch
from callback import (
    ckpt_callback,
    float_lr_callback,
    metric_updater,
    stat_callback,
    val_callback,
)
from data import (
    batch_processor,
    calibration_data_loader,
    data_loader,
    val_batch_processor,
)
from model import model
from qconfig import calibration_train_model_convert_pipeline

lr = 2e-3
float_train_weight_decay = 0.05
float_train_epochs = 300

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        eps=1e-8,
        betas=(0.9, 0.999),
        lr=lr,
        weight_decay=float_train_weight_decay,
    ),
    batch_processor=batch_processor,
    num_epochs=float_train_epochs,
    device=None,
    callbacks=[
        stat_callback,
        float_lr_callback,
        metric_updater,
        val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="LossShow"),
    ],
    val_metrics=[
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=5),
    ],
)

calibration_batch_processor = copy.deepcopy(val_batch_processor)
calibration_step = 100

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
    val_metrics=[
        dict(type="Accuracy"),
        dict(type="TopKAccuracy", top_k=5),
    ],
    log_interval=calibration_step / 10,
)
