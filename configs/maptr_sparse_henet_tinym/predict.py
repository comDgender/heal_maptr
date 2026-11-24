"""Predictor definitions for sparse MapTR.

Quantization flow: float -> calibration -> hbir -> hbm (no QAT training/predict).  # noqa: E501
"""

from callback import val_map_metric, val_metric_updater
from data import (
    hbm_batch_processor,
    hbm_data_loader,
    int_infer_data_loader,
    val_batch_processor,
    val_data_loader,
)
from model import hbir_infer_model, hbm_infer_model, test_model
from qconfig import (
    calibration_predict_model_convert_pipeline,
    float_predict_model_convert_pipeline,
)

float_predictor = dict(
    type="Predictor",
    model=test_model,
    model_convert_pipeline=float_predict_model_convert_pipeline,
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[val_map_metric],
    callbacks=[val_metric_updater],
    log_interval=100,
)

# Quantized eval: loads calibration-checkpoint-best.
calibration_predictor = dict(
    type="Predictor",
    model=test_model,
    model_convert_pipeline=calibration_predict_model_convert_pipeline,
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[val_map_metric],
    callbacks=[val_metric_updater],
    log_interval=100,
)

# HBIR eval after export_hbir (``quantized_infer.bc``).
hbir_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=int_infer_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    metrics=[val_map_metric],
    callbacks=[val_metric_updater],
    log_interval=10,
)

# On-board BPU eval after compile (``compile/model.hbm``).
hbm_predictor = dict(
    type="Predictor",
    model=hbm_infer_model,
    data_loader=hbm_data_loader,
    batch_processor=hbm_batch_processor,
    device=None,
    metrics=[val_map_metric],
    callbacks=[val_metric_updater],
    log_interval=10,
)
