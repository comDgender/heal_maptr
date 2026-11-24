import os

from callback import stat_callback, val_metric_updater
from common import ckpt_dir, val_gt
from data import (
    hbm_batch_processor,
    hbm_data_loader,
    int_infer_data_loader,
    val_batch_processor,
    val_data_loader,
)
from model import hbir_model, hbm_model, model
from qconfig import (
    calibration_predict_model_convert_pipeline,
    qat_predict_model_convert_pipeline,
)

float_predictor = dict(
    type="Predictor",
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
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=dict(
        type="MotMetric",
        gt_dir=val_gt,
        save_prefix=os.path.join(ckpt_dir, "metric_out/float_predict"),
    ),
    callbacks=[
        stat_callback,
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
    metrics=dict(
        type="MotMetric",
        gt_dir=val_gt,
        save_prefix=os.path.join(ckpt_dir, "metric_out/calib_predict"),
    ),
    callbacks=[
        stat_callback,
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
    metrics=dict(
        type="MotMetric",
        gt_dir=val_gt,
        save_prefix=os.path.join(ckpt_dir, "metric_out/qat_predict"),
    ),
    callbacks=[
        stat_callback,
        val_metric_updater,
    ],
    log_interval=50,
)

hbir_predictor = dict(
    type="Predictor",
    model=hbir_model,
    data_loader=[int_infer_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=dict(
        type="MotMetric",
        gt_dir=val_gt,
        save_prefix=os.path.join(ckpt_dir, "metric_out/int_predict"),
    ),
    callbacks=[
        stat_callback,
        val_metric_updater,
    ],
    log_interval=50,
)

hbm_predictor = dict(
    type="Predictor",
    model=hbm_model,
    data_loader=[hbm_data_loader],
    batch_processor=hbm_batch_processor,
    device=None,
    metrics=dict(
        type="MotMetric",
        gt_dir=val_gt,
        save_prefix=os.path.join(ckpt_dir, "metric_out/hbm_predict"),
    ),
    callbacks=[
        stat_callback,
        val_metric_updater,
    ],
    log_interval=50,
)
