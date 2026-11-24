import os

import torch
from callback import (
    ckpt_callback,
    loss_show_update,
    stat_callback,
    val_callback,
)
from common import ckpt_dir, class_names
from data import (
    batch_processor,
    calibration_batch_processor,
    calibration_data_loader,
    data_loader,
)
from model import example_inputs, model
from qconfig import calib_qconfig_setter, qat_qconfig_setter

float_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        betas=(0.95, 0.99),
        lr=5e-4,
        weight_decay=0.01,
    ),
    batch_processor=batch_processor,
    num_epochs=300,  # 160,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(
            type="CyclicLrUpdater",
            target_ratio=(10, 1e-4),
            cyclic_times=1,
            step_ratio_up=0.4,
            step_log_interval=50,
        ),
        val_callback,
        ckpt_callback,
    ],
    sync_bn=True,
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=dict(
        type="Kitti3DMetricDet",
        compute_aos=True,
        current_classes=class_names,
        difficultys=[0, 1, 2],
    ),
)


calibration_step = 50

calibration_trainer = dict(
    type="Calibrator",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
            ),
            dict(
                type="Float2Calibration",
                convert_mode="jit-strip",
                example_inputs=example_inputs,
                qconfig_setter=calib_qconfig_setter,
            ),
        ],
    ),
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
        type="Kitti3DMetricDet",
        compute_aos=True,
        current_classes=class_names,
        difficultys=[0, 1, 2],
    ),
    log_interval=calibration_step / 10,
)

qat_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        # activation scale freeze via fix_scale=True on setter; the old
        # qconfig_params was dead code under the setter path.
        converters=[
            dict(
                type="Float2QAT",
                convert_mode="jit-strip",
                example_inputs=example_inputs,
                qconfig_setter=qat_qconfig_setter,
            ),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.SGD,
        params={"weight": dict(weight_decay=0.0)},
        lr=1e-8,
        momentum=0.9,
    ),
    batch_processor=batch_processor,
    num_epochs=50,
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(
            type="StepDecayLrUpdater",
            lr_decay_id=[25],
            step_log_interval=50,
        ),
        val_callback,
        ckpt_callback,
    ],
    train_metrics=dict(
        type="LossShow",
    ),
    val_metrics=dict(
        type="Kitti3DMetricDet",
        compute_aos=True,
        current_classes=class_names,
        difficultys=[0, 1, 2],
    ),
)
