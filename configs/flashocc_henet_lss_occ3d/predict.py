"""Prediction configuration for FlashOcc HENet LSS Occ3D NuScenes."""

from callback import val_metrics
from data import (
    hbir_data_loader,
    hbm_data_loader,
    int_infer_data_loader,
    val_batch_processor,
    val_data_loader,
    val_metric_updater,
)
from model import (
    float_load_checkpoint_pipeline,
    hbir_infer_model,
    hbm_infer_model,
    model,
)
from qconfig import calibration_predict_model_convert_pipeline

float_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=float_load_checkpoint_pipeline,
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=val_metrics,
    callbacks=[val_metric_updater],
    log_interval=1,
)

calibration_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=calibration_predict_model_convert_pipeline,
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=val_metrics,
    callbacks=[val_metric_updater],
    log_interval=1,
)

int_infer_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=int_infer_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    metrics=val_metrics,
    callbacks=[val_metric_updater],
    log_interval=1,
)

align_bpu_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=int_infer_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    metrics=val_metrics,
    callbacks=[val_metric_updater],
    log_interval=1,
)

hbir_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=hbir_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    metrics=val_metrics,
    callbacks=[val_metric_updater],
    log_interval=1,
)

hbm_predictor = dict(
    type="Predictor",
    model=hbm_infer_model,
    data_loader=hbm_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    metrics=val_metrics,
    callbacks=[val_metric_updater],
    log_interval=1,
)
