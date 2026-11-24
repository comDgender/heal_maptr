import os

from callback import val_metric_updater
from common import ann_file, ckpt_dir
from data import (
    hbir_data_loader,
    hbm_batch_processor,
    hbm_data_loader,
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
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir,
                    "float-checkpoint-best.pth.tar",
                ),
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=dict(
        type="COCODetectionMetric",
        ann_file=ann_file,
    ),
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
    metrics=dict(
        type="COCODetectionMetric",
        ann_file=ann_file,
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=100,
)

qat_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=qat_predict_model_convert_pipeline,
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=dict(
        type="COCODetectionMetric",
        ann_file=ann_file,
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

hbir_predictor = dict(
    type="Predictor",
    model=hbir_model,
    data_loader=[hbir_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=dict(
        type="COCODetectionMetric",
        ann_file=ann_file,
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)

hbm_predictor = dict(
    type="Predictor",
    model=hbm_model,
    data_loader=[hbm_data_loader],
    batch_processor=hbm_batch_processor,
    device=None,
    metrics=dict(
        type="COCODetectionMetric",
        ann_file=ann_file,
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)
