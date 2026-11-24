"""Prediction configuration for SparseBEV HENet-tiny NuScenes."""

import os

from callback import val_metric_updater, val_nuscenes_metric
from common import ckpt_dir, convert_mode, qat_mode
from data import (
    hbm_batch_processor,
    hbm_data_loader,
    int_infer_data_loader,
    val_batch_processor,
    val_data_loader,
)
from model import eval_trace_inputs, hbir_infer_model, hbm_infer_model, model
from qconfig import calibration_qconfig_setter

float_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
                load_ema_model=True,
                allow_miss=True,
                ignore_extra=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[val_nuscenes_metric],
    callbacks=[val_metric_updater],
    log_interval=50,
)

calibration_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode=qat_mode,
        converters=[
            dict(
                type="Float2Calibration",
                convert_mode=convert_mode,
                example_inputs=eval_trace_inputs,
                qconfig_setter=calibration_qconfig_setter,
            ),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-last.pth.tar"
                ),
                ignore_extra=True,
                allow_miss=True,
                verbose=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[val_nuscenes_metric],
    callbacks=[val_metric_updater],
    log_interval=50,
)

int_infer_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=int_infer_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    metrics=[val_nuscenes_metric],
    callbacks=[val_metric_updater],
    log_interval=1,
)

# HBIR eval after export_hbir (``quantized_infer.bc``).
hbir_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=int_infer_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    metrics=[val_nuscenes_metric],
    callbacks=[val_metric_updater],
    log_interval=1,
)

# On-board BPU eval after compile (``compile/model.hbm``).
hbm_predictor = dict(
    type="Predictor",
    model=hbm_infer_model,
    data_loader=hbm_data_loader,
    batch_processor=hbm_batch_processor,
    device=None,
    metrics=[val_nuscenes_metric],
    callbacks=[val_metric_updater],
    log_interval=1,
)
