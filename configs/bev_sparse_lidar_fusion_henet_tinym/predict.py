"""Prediction configuration for SparseBEV lidar fusion."""

import copy

from callback import val_metric_updater, val_nuscenes_metric
from data import (
    hbm_batch_processor,
    hbm_data_loader,
    int_infer_data_loader,
    val_batch_processor,
    val_data_loader,
)
from model import hbir_infer_model, hbm_infer_model, model
from qconfig import (
    calibration_predict_model_convert_pipeline,
    float_checkpoint,
    qat_val_model_convert_pipeline,
)

qat_model = copy.deepcopy(model)
qat_model["head"]["enable_dn"] = False

_predictor_common = dict(
    type="Predictor",
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=[val_nuscenes_metric],
    callbacks=[val_metric_updater],
    log_interval=50,
)

float_predictor = dict(
    **_predictor_common,
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=float_checkpoint,
                allow_miss=True,
                ignore_extra=True,
            ),
        ],
    ),
)

calibration_predictor = dict(
    **_predictor_common,
    model=model,
    model_convert_pipeline=calibration_predict_model_convert_pipeline,
)

qat_predictor = dict(
    **_predictor_common,
    model=qat_model,
    model_convert_pipeline=qat_val_model_convert_pipeline,
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
