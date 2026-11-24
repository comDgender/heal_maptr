import copy
import os

import torch
from callback import (
    bn_callback,
    ckpt_callback,
    grad_callback,
    loss_show_update,
    stat_callback,
    val_callback,
)
from common import ckpt_dir, val_gt
from data import (
    batch_processor,
    calibration_data_loader,
    data_loader,
    qat_train_dataloader,
    val_batch_processor,
)
from model import model, train_model
from qconfig import (
    calibration_train_model_convert_pipeline,
    qat_train_model_convert_pipeline,
)

# Calibration-specific batch processor (no grad) and step count.
# Defined here (not in data.py) because they are training-only concerns,
# matching fcos3d_efficientnetb0/train.py convention.
calibration_batch_processor = copy.deepcopy(val_batch_processor)
# 500 steps: HistogramObserver needs enough samples to fill 2048 bins.
calibration_step = 500

# Calib/QAT val callbacks: deepcopy val_callback and attach the same
# Float2Calibration / Float2QAT pipeline so val_model matches trainer.
calibration_val_callback = copy.deepcopy(val_callback)
calibration_val_callback["val_interval"] = 1
calibration_val_callback["model_convert_pipeline"] = copy.deepcopy(
    calibration_train_model_convert_pipeline
)

qat_val_callback = copy.deepcopy(val_callback)
qat_val_callback["val_interval"] = 1
qat_val_callback["model_convert_pipeline"] = copy.deepcopy(
    qat_train_model_convert_pipeline
)

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=train_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=(
                    "./tmp_pretrained_models/fcos_efficientnetb3/float-checkpoint-best.pth.tar"  # noqa
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
            "backbone": dict(lr=2e-5),
            "sampling_offsets": dict(lr=2e-5),
        },
        eps=1e-8,
        betas=(0.9, 0.999),
        lr=2e-4,
        weight_decay=1e-4,
    ),
    batch_processor=batch_processor,
    num_epochs=250,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        grad_callback,
        dict(
            type="StepDecayLrUpdater",
            lr_decay_id=[200],
            step_log_interval=500,
        ),
        bn_callback,
        val_callback,
        ckpt_callback,
    ],
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=dict(
        type="MotMetric",
        gt_dir=val_gt,
        save_prefix=os.path.join(ckpt_dir, "metric_out/float_train"),
    ),
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
        calibration_val_callback,
        ckpt_callback,
    ],
    log_interval=calibration_step / 10,
    val_metrics=dict(
        type="MotMetric",
        gt_dir=val_gt,
        save_prefix=os.path.join(ckpt_dir, "metric_out/calib_train"),
    ),
)

qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=train_model,
    model_convert_pipeline=qat_train_model_convert_pipeline,
    data_loader=qat_train_dataloader,
    # ignore_qat_buffers=True: avoid DDP broadcast_buffers packing QAT
    # scale buffers into 1.19e-7 subnormals (triggers stubs.py:62).
    ignore_qat_buffers=True,
    optimizer=dict(
        type=torch.optim.AdamW,
        eps=1e-8,
        betas=(0.9, 0.999),
        lr=1e-5,
        weight_decay=1e-4,
    ),
    batch_processor=batch_processor,
    stop_by="epoch",
    num_epochs=20,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        grad_callback,
        dict(
            type="StepDecayLrUpdater",
            lr_decay_id=[10],
            step_log_interval=500,
        ),
        qat_val_callback,
        ckpt_callback,
    ],
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=dict(
        type="MotMetric",
        gt_dir=val_gt,
        save_prefix=os.path.join(ckpt_dir, "metric_out/qat_train"),
    ),
)
