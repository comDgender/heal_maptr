import os

from common import ckpt_dir
from data import deploy_inputs
from horizon_plugin_pytorch.dtype import qint8
from horizon_plugin_pytorch.quantization import get_qconfig, observer_v2
from horizon_plugin_pytorch.quantization.qconfig_setter import (
    ConvDtypeTemplate,
    MatmulDtypeTemplate,
    ModuleNameTemplate,
    QconfigSetter,
)

convert_mode = "jit-strip"

q_templates = [
    ModuleNameTemplate({"": qint8}),
    MatmulDtypeTemplate(
        input_dtypes=[qint8, qint8],
    ),
    ConvDtypeTemplate(
        input_dtype=qint8,
        weight_dtype=qint8,
    ),
]

cali_qconfig_setter = QconfigSetter(
    reference_qconfig=get_qconfig(observer=(observer_v2.MSEObserver)),
    templates=q_templates,
    enable_optimize=True,
    save_dir=ckpt_dir,
    custom_qconfig_mapping=None,
)
predict_qconfig_setter = QconfigSetter(
    reference_qconfig=get_qconfig(
        observer=(observer_v2.MinMaxObserver), fix_scale=True
    ),
    templates=q_templates,
    enable_optimize=True,
    save_dir=ckpt_dir,
    custom_qconfig_mapping=None,
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
            verbose=True,
        ),
        dict(
            type="Float2Calibration",
            convert_mode=convert_mode,
            example_inputs=deploy_inputs,
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
            example_inputs=deploy_inputs,
            qconfig_setter=predict_qconfig_setter,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "calibration-checkpoint-best.pth.tar"
            ),
            verbose=True,
        ),
    ],
)

deploy_model_convert_pipeline = calibration_predict_model_convert_pipeline
