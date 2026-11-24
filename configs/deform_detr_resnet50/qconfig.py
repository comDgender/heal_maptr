import copy
import os

from common import ckpt_dir, convert_mode, qat_mode
from data import example_inputs
from horizon_plugin_pytorch.dtype import qint8, qint16
from horizon_plugin_pytorch.quantization import get_qconfig, observer_v2
from horizon_plugin_pytorch.quantization.qconfig_setter import (
    ConvDtypeTemplate,
    MatmulDtypeTemplate,
    ModuleNameTemplate,
    QconfigSetter,
)


def get_qconfig_setter(is_calibration=True):
    return QconfigSetter(
        reference_qconfig=get_qconfig(
            observer=(
                observer_v2.MSEObserver
                if is_calibration is True
                else observer_v2.MinMaxObserver
            ),
            fix_scale=False,
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
                    "coord_add.0": {"dtype": qint16},
                    "coord_add.1": {"dtype": qint16},
                    "coord_add.2": {"dtype": qint16},
                    "coord_add.3": {"dtype": qint16},
                    "coord_add.4": {"dtype": qint16},
                    "coord_add.5": {"dtype": qint16},
                    "tmp_cat.0": {"dtype": qint16},
                    "tmp_cat.1": {"dtype": qint16},
                    "tmp_cat.2": {"dtype": qint16},
                    "tmp_cat.3": {"dtype": qint16},
                    "tmp_cat.4": {"dtype": qint16},
                    "tmp_cat.5": {"dtype": qint16},
                    "bbox_embed.0.layers.2": {"dtype": qint16},
                },
            ),
        ],
        enable_optimize=True,
        save_dir="./qconfig_setting",
        custom_qconfig_mapping=None,
    )


cali_qconfig_setter = get_qconfig_setter(True)
qat_qconfig_setter = get_qconfig_setter(False)


calibration_train_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    qconfig_params=dict(
        activation_calibration_observer="mse",
    ),
    converters=[
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "float-checkpoint-best.pth.tar"
            ),
        ),
        dict(
            type="Float2Calibration",
            convert_mode=convert_mode,
            qconfig_setter=copy.deepcopy(cali_qconfig_setter),
            example_inputs=example_inputs,
        ),
    ],
)

calibration_predict_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            qconfig_setter=copy.deepcopy(qat_qconfig_setter),
            example_inputs=example_inputs,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "calibration-checkpoint-best.pth.tar"
            ),
            allow_miss=True,
            ignore_extra=True,
            verbose=True,
        ),
    ],
)

qat_train_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            qconfig_setter=copy.deepcopy(qat_qconfig_setter),
            example_inputs=example_inputs,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "calibration-checkpoint-best.pth.tar"
            ),
            allow_miss=True,
            ignore_extra=True,
            verbose=True,
        ),
    ],
)

qat_predict_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            qconfig_setter=copy.deepcopy(qat_qconfig_setter),
            example_inputs=example_inputs,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "qat-checkpoint-best.pth.tar"
            ),
            allow_miss=True,
            ignore_extra=True,
            verbose=True,
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
            qconfig_setter=copy.deepcopy(qat_qconfig_setter),
            example_inputs=example_inputs,
        ),
    ],
)
