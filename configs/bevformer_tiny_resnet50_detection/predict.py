"""Predictor definitions."""

from callback import val_metric_updater, val_nuscenes_metric
from data import (
    hbm_data_loader,
    int_infer_data_loader,
    val_batch_processor,
    val_data_loader,
)
from model import hbir_infer_model, hbm_infer_model, test_model
from qconfig import (
    calibration_predict_model_convert_pipeline,
    float_predict_model_convert_pipeline,
    qat_predict_model_convert_pipeline,
)

float_predictor = dict(
    type="Predictor",
    model=test_model,
    model_convert_pipeline=float_predict_model_convert_pipeline,
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[val_nuscenes_metric],
    callbacks=[val_metric_updater],
    log_interval=1,
)

calibration_predictor = dict(
    type="Predictor",
    model=test_model,
    model_convert_pipeline=calibration_predict_model_convert_pipeline,
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[val_nuscenes_metric],
    callbacks=[val_metric_updater],
    log_interval=1,
)

qat_predictor = dict(
    type="Predictor",
    model=test_model,
    model_convert_pipeline=qat_predict_model_convert_pipeline,
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[val_nuscenes_metric],
    callbacks=[val_metric_updater],
    log_interval=1,
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

hbm_predictor = dict(
    type="Predictor",
    model=hbm_infer_model,
    data_loader=hbm_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    metrics=[val_nuscenes_metric],
    callbacks=[val_metric_updater],
    log_interval=1,
)
