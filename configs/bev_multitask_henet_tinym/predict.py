# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import os

from callback import (
    stage1_val_metrics,
    stage2_val_metrics,
    stage3_val_metrics,
    val_metric_updater,
    val_metrics,
)
from common import ckpt_dir
from data import (
    hbm_batch_processor,
    hbm_data_loader,
    int_infer_data_loader,
    val_batch_processor,
    val_data_loader,
)
from model import (
    hbir_infer_model,
    hbm_infer_model,
    model,
    model_stage1,
    model_stage2,
    model_stage3,
)
from qconfig import calibration_predict_model_convert_pipeline

float_stage1_predictor = dict(
    type="Predictor",
    model=model_stage1,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float_stage1-checkpoint-last.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
                verbose=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=stage1_val_metrics,
    callbacks=[
        val_metric_updater,
    ],
    log_interval=100,
)

float_stage2_predictor = dict(
    type="Predictor",
    model=model_stage2,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float_stage2-checkpoint-last.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
                verbose=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=stage2_val_metrics,
    callbacks=[
        val_metric_updater,
    ],
    log_interval=100,
)

float_stage3_predictor = dict(
    type="Predictor",
    model=model_stage3,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float_stage3-checkpoint-best.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
                verbose=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=stage3_val_metrics,
    callbacks=[
        val_metric_updater,
    ],
    log_interval=100,
)

# Backward-compatible alias → stage3
float_predictor = float_stage3_predictor

# --- calibration predictor ---
calibration_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=calibration_predict_model_convert_pipeline,
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=val_metrics,
    callbacks=[
        val_metric_updater,
    ],
    log_interval=100,
)

# --- HBIR: quantized_infer.bc (same YUV444 + Normalize as float val) ---
hbir_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=int_infer_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    metrics=val_metrics,
    callbacks=[val_metric_updater],
    log_interval=1,
)

# --- HBM: compile/model.hbm; RGB uint8 only (no BgrToYuv444 / Normalize) ---
hbm_predictor = dict(
    type="Predictor",
    model=hbm_infer_model,
    data_loader=hbm_data_loader,
    batch_processor=hbm_batch_processor,
    device=None,
    metrics=val_metrics,
    callbacks=[val_metric_updater],
    log_interval=1,
)
