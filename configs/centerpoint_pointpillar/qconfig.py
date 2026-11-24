"""Quantization configs for CenterPoint PointPillars NuScenes."""

import copy
import os

from common import ckpt_dir, convert_mode, qat_mode
from horizon_plugin_pytorch.dtype import qint8, qint16
from horizon_plugin_pytorch.quantization import get_qconfig, observer_v2
from horizon_plugin_pytorch.quantization.qconfig_setter import (
    ConvDtypeTemplate,
    MatmulDtypeTemplate,
    ModuleNameTemplate,
    QconfigSetter,
)
from model import example_inputs


def get_custom_qconfig(is_calib):
    return QconfigSetter(
        reference_qconfig=get_qconfig(
            observer=(
                observer_v2.MSEObserver
                if is_calib
                else observer_v2.MinMaxObserver
            )
        ),
        templates=[
            ModuleNameTemplate({"": qint8}),
            MatmulDtypeTemplate(input_dtypes=[qint8, qint16]),
            ConvDtypeTemplate(input_dtype=qint8, weight_dtype=qint8),
        ],
        enable_optimize=True,
        save_dir=os.path.join(ckpt_dir, "qconfig_setting"),
        custom_qconfig_mapping=None,
    )


qat_qconfig_setter = get_custom_qconfig(False)
calib_qconfig_setter = get_custom_qconfig(True)

calibration_train_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(
            type="Float2Calibration",
            convert_mode=convert_mode,
            qconfig_setter=calib_qconfig_setter,
        ),
    ],
)

# Used when Validation.val_model is set; train val uses train-model deepcopy.
calibration_val_model_convert_pipeline = copy.deepcopy(
    calibration_train_model_convert_pipeline
)
calibration_val_model_convert_pipeline["converters"] = [
    dict(
        type="Float2Calibration",
        convert_mode=convert_mode,
        example_inputs=example_inputs,
        qconfig_setter=calib_qconfig_setter,
    ),
]

qat_train_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    qconfig_params=dict(
        activation_qat_qkwargs=dict(averaging_constant=0),
        weight_qat_qkwargs=dict(averaging_constant=1),
    ),
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            qconfig_setter=qat_qconfig_setter,
        ),
    ],
)

qat_val_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            example_inputs=example_inputs,
            state="val",
            qconfig_setter=qat_qconfig_setter,
        ),
    ],
)
