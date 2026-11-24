import copy
import os

from common import ckpt_dir, convert_mode
from horizon_plugin_pytorch.dtype import qint8
from horizon_plugin_pytorch.quantization import get_qconfig, observer_v2
from horizon_plugin_pytorch.quantization.qconfig_setter import (
    ConvDtypeTemplate,
    MatmulDtypeTemplate,
    ModuleNameTemplate,
    QconfigSetter,
)
from model import deploy_inputs, example_inputs


def get_qconfig_setter(is_calibration=True):
    return QconfigSetter(
        reference_qconfig=get_qconfig(
            observer=(
                observer_v2.HistogramObserver
                if is_calibration is True
                else observer_v2.MinMaxObserver
            ),
            fix_scale=False if is_calibration else True,
        ),
        templates=[
            ModuleNameTemplate({"": qint8}),
            ModuleNameTemplate(
                {"backbone.quant": {"dtype": qint8, "threshold": 1.0}},
            ),
            MatmulDtypeTemplate(
                input_dtypes=[qint8, qint8],
                prefix=None,
            ),
            ConvDtypeTemplate(
                input_dtype=qint8,
                weight_dtype=qint8,
                prefix=None,
            ),
        ],
        enable_optimize=True,
        save_dir=os.path.join(ckpt_dir, "qconfig_setting"),
        custom_qconfig_mapping=None,
    )


calibration_qconfig_setter = get_qconfig_setter(True)
predict_qconfig_setter = get_qconfig_setter(False)

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
        dict(
            type="Float2Calibration",
            convert_mode=convert_mode,
            qconfig_setter=copy.deepcopy(calibration_qconfig_setter),
            example_inputs=example_inputs,
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
            qconfig_setter=copy.deepcopy(predict_qconfig_setter),
            example_inputs=example_inputs,
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

qat_predict_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            qconfig_setter=copy.deepcopy(predict_qconfig_setter),
            example_inputs=example_inputs,
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

qat_train_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            qconfig_setter=copy.deepcopy(predict_qconfig_setter),
            example_inputs=example_inputs,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "calibration-checkpoint-best.pth.tar"
            ),
            verbose=True,
            ignore_extra=True,
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
            qconfig_setter=copy.deepcopy(predict_qconfig_setter),
            example_inputs=deploy_inputs,
        ),
    ],
)

export_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            qconfig_setter=copy.deepcopy(predict_qconfig_setter),
            example_inputs=deploy_inputs,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "qat-checkpoint-best.pth.tar"
            ),
            ignore_extra=True,
        ),
    ],
)
