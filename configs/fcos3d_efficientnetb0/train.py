import copy

import torch
from callback import (
    ckpt_callback,
    loss_show_update,
    stat_callback,
    val_callback,
)
from data import (
    batch_processor,
    calibration_data_loader,
    data_loader,
    meta_rootdir,
    val_batch_processor,
)
from model import model
from qconfig import (
    calibration_train_model_convert_pipeline,
    qat_train_model_convert_pipeline,
)

calibration_batch_processor = copy.deepcopy(val_batch_processor)
calibration_step = 100

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=(
                    "./tmp_pretrained_models/efficientnet/"
                    "float-checkpoint-best.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        params={"weight": dict(weight_decay=0.01)},
        lr=2e-4,
    ),
    batch_processor=batch_processor,
    num_epochs=48,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(
            type="CyclicLrUpdater",
            target_ratio=(10, 1e-4),
            cyclic_times=1,
            step_ratio_up=0.4,
            step_log_interval=500,
        ),
        val_callback,
        ckpt_callback,
    ],
    sync_bn=True,
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=dict(
        type="NuscenesMonoMetric",
        data_root=meta_rootdir,
        version="v1.0-trainval",
    ),
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
        val_callback,
        ckpt_callback,
    ],
    val_metrics=dict(
        type="NuscenesMonoMetric",
        data_root=meta_rootdir,
        version="v1.0-trainval",
    ),
    log_interval=calibration_step / 10,
)

qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=qat_train_model_convert_pipeline,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        params={"weight": dict(weight_decay=0.01)},
        lr=1e-6,
    ),
    batch_processor=batch_processor,
    num_epochs=10,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        val_callback,
        ckpt_callback,
    ],
    sync_bn=True,
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=dict(
        type="NuscenesMonoMetric",
        data_root=meta_rootdir,
        version="v1.0-trainval",
    ),
)
