import os

from common import ckpt_dir, convert_mode
from data import deploy_inputs
from horizon_plugin_pytorch.dtype import qint8
from horizon_plugin_pytorch.quantization import get_qconfig, observer_v2
from horizon_plugin_pytorch.quantization.qconfig_setter import (
    ConvDtypeTemplate,
    MatmulDtypeTemplate,
    ModuleNameTemplate,
    QconfigSetter,
)

cali_qconfig_setter = QconfigSetter(
    reference_qconfig=get_qconfig(observer=(observer_v2.MSEObserver)),
    templates=[
        ModuleNameTemplate(
            {"": {"output": qint8, "weight": qint8}},
            enable_propagate=True,
        ),
        MatmulDtypeTemplate(
            input_dtypes=[qint8, qint8],
        ),
        ConvDtypeTemplate(
            input_dtype=qint8,
            weight_dtype=qint8,
        ),
    ],
    enable_optimize=True,
    save_dir="./qconfig_setting",
)

qat_qconfig_setter = QconfigSetter(
    reference_qconfig=get_qconfig(
        observer=(observer_v2.MinMaxObserver), fix_scale=True
    ),
    templates=[
        ModuleNameTemplate(
            {"": {"output": qint8, "weight": qint8}}, enable_propagate=True
        ),
        MatmulDtypeTemplate(
            input_dtypes=[qint8, qint8],
        ),
        ConvDtypeTemplate(
            input_dtype=qint8,
            weight_dtype=qint8,
        ),
    ],
    enable_optimize=True,
    save_dir="./qconfig_setting",
)

float2calibration = dict(
    type="Float2Calibration",
    convert_mode=convert_mode,
    example_inputs=deploy_inputs,
    qconfig_setter=cali_qconfig_setter,
)

float2qat = dict(
    type="Float2QAT",
    convert_mode=convert_mode,
    example_inputs=deploy_inputs,
    qconfig_setter=qat_qconfig_setter,
)

calibration_train_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "float-checkpoint-best.pth.tar"
            ),
            allow_miss=True,
            verbose=True,
        ),
        float2calibration,
    ],
)

calibration_predict_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        float2calibration,
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "calibration-checkpoint-best.pth.tar"
            ),
            ignore_extra=True,
        ),
    ],
)

qat_train_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    # activation scale freeze is handled by fix_scale=True on the setter's
    # reference_qconfig; qconfig_params was dead code under the setter path.
    converters=[
        float2qat,
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "calibration-checkpoint-best.pth.tar"
            ),
        ),
    ],
)

qat_predict_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        float2qat,
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "qat-checkpoint-best.pth.tar"
            ),
            ignore_extra=True,
        ),
    ],
)

real_deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        float2qat,
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "qat-checkpoint-best.pth.tar"
            ),
            ignore_extra=True,
            allow_miss=True,
            verbose=True,
        ),
    ],
)
