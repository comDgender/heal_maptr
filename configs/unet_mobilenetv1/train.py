import copy

import torch
from callback import (
    calibration_val_callback,
    ckpt_callback,
    float_val_callback,
    qat_val_callback,
    stat_callback,
)
from common import (
    qat_lr,
    qat_train_epochs,
    start_lr,
    train_epochs,
    weight_decay,
)
from data import (
    batch_processor,
    calibration_data_loader,
    data_loader,
    qat_data_loader,
    val_batch_processor,
)
from model import model
from qconfig import (
    calibration_train_model_convert_pipeline,
    calibration_val_model_convert_pipeline,
    qat_train_model_convert_pipeline,
)

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=weight_decay)},
        lr=start_lr,
        momentum=0.9,
    ),
    batch_processor=batch_processor,
    device=None,
    num_epochs=train_epochs,
    callbacks=[
        stat_callback,
        dict(
            type="CosLrUpdater",
            warmup_len=2,
            warmup_by="epoch",
            step_log_interval=100,
        ),
        float_val_callback,
        ckpt_callback,
    ],
    sync_bn=True,
)

calibration_batch_processor = copy.deepcopy(val_batch_processor)
calibration_step = 10

calibration_val_callback_cfg = {
    **calibration_val_callback,
    "model_convert_pipeline": calibration_val_model_convert_pipeline,
}

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
        calibration_val_callback_cfg,
        ckpt_callback,
    ],
    log_interval=1,
)

qat_val_callback_cfg = {
    **qat_val_callback,
    "model_convert_pipeline": calibration_val_model_convert_pipeline,
}

qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=qat_train_model_convert_pipeline,
    data_loader=qat_data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=weight_decay)},
        lr=qat_lr,
        momentum=0.9,
    ),
    batch_processor=batch_processor,
    device=None,
    num_epochs=qat_train_epochs,
    callbacks=[
        stat_callback,
        dict(
            type="StepDecayLrUpdater",
            lr_decay_id=[20],
            lr_decay_factor=0.1,
        ),
        qat_val_callback_cfg,
        ckpt_callback,
    ],
    sync_bn=True,
)
