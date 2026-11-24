import torch
from callback import (
    ckpt_callback,
    loss_show_update,
    stat_callback,
    val_callback,
    val_nuscenes_metric,
)
from data import (
    batch_processor,
    calibration_batch_processor,
    calibration_data_loader,
    data_loader,
)
from model import model
from qconfig import (
    calibration_train_model_convert_pipeline,
    qat_train_model_convert_pipeline,
)

calibration_step = 10

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=(
                    "./tmp_pretrained_models/"
                    # CI mounts nash as tmp_pretrained_models; name below
                    # matches fcos3d_efficientnetb3_pretrain_legacy.
                    "fcos3d_efficientnetb3_pretrain_legacy/"
                    "float-checkpoint-best.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
                #  ignore_tensor_shape=True,
                verbose=True,
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        params={
            "backbone": dict(lr=3e-5),
        },
        eps=1e-8,
        betas=(0.9, 0.999),
        lr=3e-4,
        weight_decay=0.01,
    ),
    batch_processor=batch_processor,
    num_epochs=36,
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
            stop_lr=1e-6,
        ),
        val_callback,
        ckpt_callback,
    ],
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=[val_nuscenes_metric],
    sync_bn=True,
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
        val_callback,
        ckpt_callback,
    ],
    val_metrics=[val_nuscenes_metric],
    log_interval=calibration_step / 10,
)

qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=qat_train_model_convert_pipeline,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        eps=1e-8,
        betas=(0.9, 0.999),
        params={
            "backbone": dict(lr=3e-6),
        },
        lr=3e-5,
        weight_decay=0.01,
    ),
    batch_processor=batch_processor,
    num_epochs=10,
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
    val_metrics=[val_nuscenes_metric],
)
