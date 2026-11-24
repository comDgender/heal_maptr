import copy

import torch
from callback import calibration_val_callback as _calibration_val_callback
from callback import (
    ckpt_callback,
    loss_show_update,
    stat_callback,
    val_callback,
)
from common import base_lr, num_epochs
from data import (
    batch_processor,
    calibration_data_loader,
    data_loader,
    val_batch_processor,
)
from model import deploy_model, model
from qconfig import calibration_train_model_convert_pipeline

calibration_batch_processor = copy.deepcopy(val_batch_processor)
calibration_step = 10

# Attach the Float2Calibration pipeline so the deepcopied val_model (with
# GaNetDecoder) is converted to the same qat state as the trainer's
# deploy_model before loading its state_dict. Mirrors
# motr_efficientnetb3/train.py:43-47.
calibration_val_callback = copy.deepcopy(_calibration_val_callback)
calibration_val_callback["model_convert_pipeline"] = copy.deepcopy(
    calibration_train_model_convert_pipeline
)

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.Adam,
        params={"weight": dict(weight_decay=4e-5)},
        lr=base_lr,
    ),
    batch_processor=batch_processor,
    stop_by="epoch",
    num_epochs=num_epochs,
    device=None,
    sync_bn=True,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(
            type="CosLrUpdater",
            warmup_len=1,
            warmup_by="epoch",
            step_log_interval=10,
        ),
        val_callback,
        ckpt_callback,
    ],
    train_metrics=[
        dict(type="LossShow"),
    ],
    val_metrics=[
        dict(type="CulaneF1Score"),
    ],
)

calibration_trainer = dict(
    type="Calibrator",
    model=deploy_model,
    model_convert_pipeline=calibration_train_model_convert_pipeline,
    data_loader=calibration_data_loader,
    batch_processor=calibration_batch_processor,
    num_steps=calibration_step,
    device=None,
    callbacks=[
        stat_callback,
        calibration_val_callback,
        ckpt_callback,
    ],
    val_metrics=[
        dict(type="CulaneF1Score"),
    ],
    log_interval=calibration_step / 10,
)
