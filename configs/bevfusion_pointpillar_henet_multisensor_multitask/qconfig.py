"""Quantization configs for BevFusion multisensor multitask."""

import os

from common import ckpt_dir, convert_mode, qat_mode
from data import calibration_example_data_loader
from horizon_plugin_pytorch.dtype import qint8, qint16
from horizon_plugin_pytorch.quantization import get_qconfig, observer_v2
from horizon_plugin_pytorch.quantization.qconfig_setter import (
    ConvDtypeTemplate,
    MatmulDtypeTemplate,
    ModuleNameTemplate,
    QconfigSetter,
    SensitivityTemplate,
)
from model import deploy_inputs


def get_custom_qconfig(is_calib):
    return QconfigSetter(
        reference_qconfig=get_qconfig(
            observer=(
                observer_v2.HistogramObserver
                if is_calib
                else observer_v2.MinMaxObserver
            )
        ),
        templates=[
            ModuleNameTemplate({"": qint8}),
            MatmulDtypeTemplate(input_dtypes=[qint8, qint8]),
            ConvDtypeTemplate(input_dtype=qint8, weight_dtype=qint8),
            # Output of the object query / reference-point projections needs
            # int16 activation (was previously set in the model's
            # `set_qconfig`; moved here so all quant config lives in the
            # template, matching the sibling configs).
            ModuleNameTemplate(
                {
                    "bev_decoders.0.quant_object_query_embed": {
                        "output": qint16
                    },
                    "bev_decoders.0.reference_points": {"dtype": qint16},
                    "bev_decoders.0.decoder.layers.5.sa.matmul": {
                        "output": qint16
                    },
                },
            ),
            SensitivityTemplate(
                [
                    ("lidar_net.neck.blocks.0.1.2", "output"),
                    ("lidar_net.neck.blocks.0.2.2", "output"),
                    ("lidar_net.neck.blocks.0.3.2", "output"),
                    ("lidar_net.neck.blocks.0.4.2", "output"),
                ],
                topk_or_ratio=1.0,
            ),
        ],
        enable_optimize=True,
        save_dir=os.path.join(ckpt_dir, "qconfig_setting"),
        custom_qconfig_mapping=None,
    )


qat_qconfig_setter = get_custom_qconfig(False)
calib_qconfig_setter = get_custom_qconfig(True)

float2calibration = dict(
    type="Float2Calibration",
    convert_mode=convert_mode,
    example_data_loader=calibration_example_data_loader,
    qconfig_setter=calib_qconfig_setter,
)

float2qat = dict(
    type="Float2QAT",
    convert_mode=convert_mode,
    example_data_loader=calibration_example_data_loader,
    qconfig_setter=qat_qconfig_setter,
)

calibration_train_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
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
        dict(type="RepModel2Deploy"),
        float2calibration,
        dict(type="FixWeightQScale"),
    ],
)

# ---------------------------------------------------------------------------
# Predict pipelines used by predict.py's float/calibration predictors. The
# quant_analysis pipeline below mirrors
# ``calibration_predict_model_convert_pipeline``.
# ---------------------------------------------------------------------------
float_predict_model_convert_pipeline = dict(
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
        dict(type="RepModel2Deploy"),
    ],
)

calibration_predict_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(type="RepModel2Deploy"),
        float2qat,
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "calibration-checkpoint-last.pth.tar"
            ),
            verbose=True,
            allow_miss=True,
            ignore_extra=True,
        ),
    ],
)

deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            example_inputs=deploy_inputs,
            qconfig_setter=qat_qconfig_setter,
        ),
    ],
)

real_deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(type="RepModel2Deploy"),
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            example_inputs=deploy_inputs,
            qconfig_setter=qat_qconfig_setter,
        ),
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
)

# The analysis model mirrors the calibration predict pipeline
# (RepModel2Deploy + Float2QAT(val) + calibration checkpoint) so quant_analysis
# measures the same quantized model that actually ships, not a float-only
# fake-quant built from the float checkpoint. This requires a prior calibration
# stage to have produced ``calibration-checkpoint-last.pth.tar``.
quant_analysis_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(type="RepModel2Deploy"),
        float2qat,
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
)
