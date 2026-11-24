import torch
from callback import (
    ckpt_callback,
    loss_show_update,
    stat_callback,
    val_callback,
    val_nuscenes_metric,
)
from common import device_ids, tensorboard_log_path
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

# 200 total steps (25/gpu x batch 4 x 8 gpus = 800 frames). The previous 40
# left only 160 frames: with fix_scale the QAT run is frozen on that range for
# all 10 epochs, and 339/488 activation observers clipped on real val data.
# HistogramObserver (see qconfig.py) needs the larger sample count to build a
# meaningful distribution; MSE was the better choice at 40 steps.
calibration_step = 200 // len(device_ids)

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
                    "fcos3d_efficientnetb3_pretrain_legacy/"
                    "float-checkpoint-best.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
                verbose=True,
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        params={
            "backbone": dict(lr=2e-4),
        },
        eps=1e-8,
        betas=(0.9, 0.999),
        lr=2e-4,
        weight_decay=0.05,
    ),
    batch_processor=batch_processor,
    num_epochs=48,
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
        dict(
            type="TensorBoard",
            save_dir=tensorboard_log_path,
            loss_name_reg="^(cls|reg).*",
            update_freq=50,
            update_by="step",
        ),
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
        dict(
            type="TensorBoard",
            save_dir=tensorboard_log_path,
            update_freq=10,
            update_by="step",
        ),
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
            "backbone": dict(lr=2e-5),
        },
        lr=2e-5,
        weight_decay=0.01,
    ),
    batch_processor=batch_processor,
    num_epochs=14,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(
            type="StepDecayLrUpdater",
            # 10->14 epochs with decay moved 6->8: the previous run peaked at
            # epoch 4 and then sat in 0.3777~0.3792 for epochs 5-9, i.e. it had
            # converged after the decay. Extending only the tail adds nothing;
            # the extra budget goes to the high-lr phase (0-7) instead.
            lr_decay_id=[8],
            step_log_interval=500,
        ),
        val_callback,
        ckpt_callback,
        dict(
            type="TensorBoard",
            save_dir=tensorboard_log_path,
            loss_name_reg="^(cls|reg).*",
            update_freq=50,
            update_by="step",
        ),
    ],
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=[val_nuscenes_metric],
)
