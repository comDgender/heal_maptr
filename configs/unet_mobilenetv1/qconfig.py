import copy
import os

import torch
from common import ckpt_dir, convert_mode, data_shape
from horizon_plugin_pytorch.dtype import qint8
from horizon_plugin_pytorch.quantization import get_qconfig, observer_v2
from horizon_plugin_pytorch.quantization.qconfig_setter import (
    ConvDtypeTemplate,
    MatmulDtypeTemplate,
    ModuleNameTemplate,
    QconfigSetter,
)

fx_example_inputs = dict(img=torch.randn((1,) + data_shape))


def get_qconfig_setter(is_calibration=True):
    return QconfigSetter(
        reference_qconfig=get_qconfig(
            observer=(
                observer_v2.HistogramObserver
                if is_calibration
                else observer_v2.MinMaxObserver
            ),
            fix_scale=False if is_calibration else True,
        ),
        templates=[
            ModuleNameTemplate({"": qint8}),
            MatmulDtypeTemplate(
                input_dtypes=[qint8, qint8],
                prefix=None,
            ),
            ConvDtypeTemplate(
                input_dtype=qint8,
                weight_dtype=qint8,
                prefix=None,
            ),
            ModuleNameTemplate(
                {
                    "backbone.quant": {"dtype": qint8, "threshold": 1.0},
                    "head.output_convs.stride4.0": {"dtype": qint8},
                    "head.output_convs.stride8.0": {"dtype": qint8},
                    "head.output_convs.stride16.0": {"dtype": qint8},
                    "head.output_convs.stride32.0": {"dtype": qint8},
                    "head.output_convs.stride64.0": {"dtype": qint8},
                },
                freeze=True,
            ),
        ],
        enable_optimize=True,
        save_dir=os.path.join(ckpt_dir, "qconfig_setting"),
        custom_qconfig_mapping=None,
    )


cali_qconfig_setter = get_qconfig_setter(True)
qat_qconfig_setter = get_qconfig_setter(False)

calibration_train_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "float-checkpoint-best.pth.tar"
            ),
            verbose=True,
        ),
        dict(
            type="Float2Calibration",
            convert_mode=convert_mode,
            example_inputs=fx_example_inputs,
            qconfig_setter=cali_qconfig_setter,
        ),
    ],
)

calibration_predict_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            example_inputs=fx_example_inputs,
            qconfig_setter=copy.deepcopy(qat_qconfig_setter),
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "calibration-checkpoint-best.pth.tar"
            ),
            ignore_extra=True,
            verbose=True,
        ),
    ],
)

qat_train_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            example_inputs=fx_example_inputs,
            qconfig_setter=copy.deepcopy(qat_qconfig_setter),
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir,
                "calibration-checkpoint-best.pth.tar",
            ),
            ignore_extra=True,
            verbose=True,
        ),
    ],
)

qat_predict_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            example_inputs=fx_example_inputs,
            qconfig_setter=copy.deepcopy(qat_qconfig_setter),
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "qat-checkpoint-best.pth.tar"
            ),
            ignore_extra=True,
            verbose=True,
        ),
    ],
)

deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            qconfig_setter=copy.deepcopy(qat_qconfig_setter),
            example_inputs=fx_example_inputs,
        ),
    ],
)

calibration_val_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            qconfig_setter=copy.deepcopy(qat_qconfig_setter),
            example_inputs=fx_example_inputs,
        ),
    ],
)
