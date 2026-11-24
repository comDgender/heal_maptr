"""Predictor definitions for sparse multitask det + MapTR + FlashOcc."""

from callback import val_metric_updater, val_metrics, val_metrics_stage2
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
    test_model,
    test_model_stage2,
)
from qconfig import (
    calibration_predict_model_convert_pipeline,
    float_merged_predict_model_convert_pipeline,
    float_stage1_predict_model_convert_pipeline,
    float_stage2_predict_model_convert_pipeline,
)

_predictor_common = dict(
    type="Predictor",
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    callbacks=[val_metric_updater],
    log_interval=50,
)

float_stage1_predictor = dict(
    **_predictor_common,
    model=test_model,
    model_convert_pipeline=float_stage1_predict_model_convert_pipeline,
    metrics=val_metrics,
)

float_stage2_predictor = dict(
    **_predictor_common,
    model=test_model_stage2,
    model_convert_pipeline=float_stage2_predict_model_convert_pipeline,
    metrics=val_metrics_stage2,
)

# Full model (det + om + occ) loaded from the merged checkpoint — evaluates
# the complete model produced by the `float_merge` train stage. Stage name
# `float_merge` aligns with `float_merge_trainer` (run via
# `predict.py -s float_merge`).
float_merge_predictor = dict(
    **_predictor_common,
    model=test_model,
    model_convert_pipeline=float_merged_predict_model_convert_pipeline,
    metrics=val_metrics,
)

calibration_predictor = dict(
    **_predictor_common,
    model=test_model,
    model_convert_pipeline=calibration_predict_model_convert_pipeline,
    metrics=val_metrics,
)

int_infer_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=int_infer_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    metrics=val_metrics,
    callbacks=[val_metric_updater],
    log_interval=10,
)

# Backward-compatible alias.
hbir_predictor = int_infer_predictor

# On-board BPU eval after compile (``compile/model.hbm``).
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
