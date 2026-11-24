from callback import val_metric_updater, val_nuscenes_metric
from data import (
    hbir_data_loader,
    hbm_batch_processor,
    hbm_data_loader,
    val_batch_processor,
    val_data_loader,
)
from model import hbir_model, hbir_model_convert_pipeline, hbm_model, model
from qconfig import (
    calibration_predict_model_convert_pipeline,
    qat_predict_model_convert_pipeline,
)

float_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=hbir_model_convert_pipeline,
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[val_nuscenes_metric],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

calibration_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=calibration_predict_model_convert_pipeline,
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[val_nuscenes_metric],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

qat_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=qat_predict_model_convert_pipeline,
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[val_nuscenes_metric],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

hbir_predictor = dict(
    type="Predictor",
    model=hbir_model,
    data_loader=hbir_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    metrics=[val_nuscenes_metric],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)

hbm_predictor = dict(
    type="Predictor",
    model=hbm_model,
    data_loader=hbm_data_loader,
    batch_processor=hbm_batch_processor,
    device=None,
    metrics=[val_nuscenes_metric],
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)
