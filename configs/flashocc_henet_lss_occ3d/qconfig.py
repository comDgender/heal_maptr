"""Quantization configuration and model-convert pipelines."""

import copy
import os

from common import ckpt_dir, convert_mode
from data import val_example_data_loader
from horizon_plugin_pytorch.dtype import qint8, qint16
from horizon_plugin_pytorch.quantization import get_qconfig, observer_v2
from horizon_plugin_pytorch.quantization.qconfig_setter import (
    ConvDtypeTemplate,
    EqualizeInOutScaleTemplate,
    MatmulDtypeTemplate,
    ModuleNameTemplate,
    QconfigSetter,
)
from model import deploy_inputs, example_inputs

int16_qconfig = dict()
int16_qconfig["view_transformer.quant_stub"] = {"dtype": qint16}
int16_qconfig["view_transformer.dquant_stub"] = {"dtype": qint16}

qtemplates = [
    ModuleNameTemplate({"": qint8}),
    ModuleNameTemplate(
        int16_qconfig,
        freeze=True,
    ),
    MatmulDtypeTemplate(
        input_dtypes=[qint8, qint8],
    ),
    ConvDtypeTemplate(
        input_dtype=qint8,
        weight_dtype=qint8,
    ),
]

cali_qconfig_setter = QconfigSetter(
    reference_qconfig=get_qconfig(observer=observer_v2.MSEObserver),
    templates=qtemplates,
    enable_optimize=True,
    save_dir=ckpt_dir,
    custom_qconfig_mapping=None,
    optimize_templates_kwargs={
        EqualizeInOutScaleTemplate: {"only_fuse_relu_after_conv": True}
    },
)

qat_qconfig_setter = cali_qconfig_setter

calibration_train_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "float-checkpoint-best.pth.tar"
            ),
            ignore_extra=True,
            verbose=True,
        ),
        dict(
            type="Float2Calibration",
            convert_mode=convert_mode,
            example_inputs=example_inputs,
            qconfig_setter=copy.deepcopy(cali_qconfig_setter),
        ),
        dict(
            type="FixWeightQScale",
        ),
    ],
)

calibration_val_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(
            type="Float2Calibration",
            convert_mode=convert_mode,
            example_data_loader=val_example_data_loader,
            qconfig_setter=copy.deepcopy(cali_qconfig_setter),
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
            example_inputs=example_inputs,
            qconfig_setter=copy.deepcopy(cali_qconfig_setter),
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

deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            example_inputs=deploy_inputs,
            qconfig_setter=copy.deepcopy(qat_qconfig_setter),
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "calibration-checkpoint-best.pth.tar"
            ),
            ignore_extra=True,
            allow_miss=True,
            verbose=True,
        ),
    ],
)
