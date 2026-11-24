import copy
import os

from callback import val_metric_updater, val_metrics
from common import ckpt_dir, qat_mode
from data import int_infer_data_loader, val_batch_processor, val_data_loader
from model import cali_model, deploy_inputs, hbir_infer_model
from qconfig import float2calibration, float2qat

# predictor
float_predictor = dict(
    type="Predictor",
    model=cali_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-last.pth.tar"
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
    metrics=val_metrics,
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

calibration_predictor = dict(
    type="Predictor",
    model=cali_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode=qat_mode,
        qconfig_params=dict(
            activation_calibration_observer="mse",
        ),
        converters=[
            float2calibration,
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-last.pth.tar"
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
    metrics=val_metrics,
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

qat_predictor = dict(
    type="Predictor",
    model=cali_model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode=qat_mode,
        converters=[
            float2qat,
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-last.pth.tar"
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
    metrics=val_metrics,
    callbacks=[
        val_metric_updater,
    ],
    log_interval=50,
)

# -------------------------- int validation config --------------------------
hbir_predictor = dict(
    type="Predictor",
    model=hbir_infer_model,
    data_loader=int_infer_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    metrics=val_metrics,
    callbacks=[
        val_metric_updater,
    ],
    log_interval=10,
)

# ----------------------- hbm infer predictor config -----------------------
# QCNetOEHbmInfer wraps an HbmModule (.hbm) and reuses the streaming-infer
# preprocess/postprocess of the hbir model. Requires board/hbm_infer runtime.
hbm_infer_model = copy.deepcopy(hbir_infer_model)
hbm_infer_model["type"] = "QCNetOEHbmInfer"
hbm_infer_model["ir_model"] = dict(
    type="HbmModule",
    model_path=os.path.join(ckpt_dir, "compile", "model.hbm"),
    host="10.35.72.2",
)
hbm_infer_model["example_data"] = deploy_inputs

hbm_predictor = dict(
    type="Predictor",
    model=hbm_infer_model,
    data_loader=int_infer_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    metrics=val_metrics,
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)
