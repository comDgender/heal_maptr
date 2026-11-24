"""Quantization config and model convert pipelines for sparse MapTR.

Quantization flow (calibration-only, no QAT training):
    float -> calibration -> predict / deploy / export

Sparse MapTR does not need a separate QAT stage; calibration accuracy is
sufficient. All post-calibration pipelines load ``calibration-checkpoint-best``.  # noqa: E501
"""

import copy
import os

from common import ckpt_dir, convert_mode, deploy_inputs, qat_mode, task_name
from horizon_plugin_pytorch.dtype import qint8, qint16
from horizon_plugin_pytorch.quantization import get_qconfig, observer_v2
from horizon_plugin_pytorch.quantization.qconfig_setter import (
    ConvDtypeTemplate,
    MatmulDtypeTemplate,
    ModuleNameTemplate,
    QconfigSetter,
    SensitivityTemplate,
)


def get_output_qconfig(dtype, max_val):
    return {
        "dtype": dtype,
        "threshold": {"output": max_val},
    }


# Decoder layers with large dynamic range: keep key ops at int16.
fixed_int16_qconfig = {}
for layer in range(2, 36, 6):
    fixed_int16_qconfig[f"bev_decoders.0.decoder.layers.{layer}.point_cat"] = (
        get_output_qconfig(qint16, 60)
    )
    fixed_int16_qconfig[
        f"bev_decoders.0.decoder.layers.{layer}.reciprocal_op"
    ] = get_output_qconfig(qint16, 10)
    fixed_int16_qconfig[f"bev_decoders.0.decoder.layers.{layer}.point_sum"] = (
        get_output_qconfig(qint16, 60)
    )
    fixed_int16_qconfig[f"bev_decoders.0.decoder.layers.{layer}.point_mul"] = (
        get_output_qconfig(qint16, 1.1)
    )

q_templates = [
    ModuleNameTemplate({"": qint8}),
    ModuleNameTemplate(fixed_int16_qconfig, freeze=True),
    MatmulDtypeTemplate(input_dtypes=[qint8, qint8]),
    ConvDtypeTemplate(input_dtype=qint8, weight_dtype=qint8),
    SensitivityTemplate(
        [
            ("bev_decoders.0.decoder.quant_anchor", "output"),
            ("bev_decoders.0.decoder.mat_quant_stub", "output"),
            *[
                (
                    f"bev_decoders.0.decoder.layers.{i}.kps_generator.learnable_fc",  # noqa: E501
                    "output",
                )
                for i in [2, 8, 14, 20, 26, 32]
            ],
            *[
                (
                    f"bev_decoders.0.decoder.layers.{i}.kps_generator._generated_add_0",  # noqa: E501
                    "output",
                )
                for i in [2, 8, 14, 20, 26, 32]
            ],
            *[
                (
                    f"bev_decoders.0.decoder.layers.{i}.kps_generator._generated_mul_0",  # noqa: E501
                    "output",
                )
                for i in [2, 8, 14, 20, 26, 32]
            ],
            *[
                (
                    f"bev_decoders.0.decoder.layers.{i}.kps_generator._generated_cat_0",  # noqa: E501
                    "output",
                )
                for i in [2, 8, 14, 20, 26, 32]
            ],
            *[
                (
                    f"bev_decoders.0.decoder.layers.{i}.kps_generator.quant_fix_height",  # noqa: E501
                    "output",
                )
                for i in [2, 8, 14, 20, 26, 32]
            ],
            *[
                (
                    f"bev_decoders.0.decoder.layers.{i}.kps_generator._generated_cat_1",  # noqa: E501
                    "output",
                )
                for i in [2, 8, 14, 20, 26, 32]
            ],
            *[
                (
                    f"bev_decoders.0.decoder.layers.{i}.kps_generator._generated_add_1",  # noqa: E501
                    "output",
                )
                for i in [2, 8, 14, 20, 26, 32]
            ],
            *[
                (f"bev_decoders.0.decoder.layers.{i}.point_matmul", "output")
                for i in [2, 8, 14, 20, 26, 32]
            ],
            *[
                (f"bev_decoders.0.decoder.layers.{i}.point_matmul", "input-0")
                for i in [2, 8, 14, 20, 26, 32]
            ],
            *[
                (f"bev_decoders.0.decoder.layers.{i}.point_matmul", "input-1")
                for i in [2, 8, 14, 20, 26, 32]
            ],
            *[
                (f"bev_decoders.0.decoder.layers.{i}.layers.10", "output")
                for i in [5, 11, 17, 23, 29]
            ],
            *[
                (
                    f"bev_decoders.0.decoder.layers.{i}.layers.11.scale_quant_stub",  # noqa: E501
                    "output",
                )
                for i in [5, 11, 17, 23, 29]
            ],
            *[
                (f"bev_decoders.0.decoder.layers.{i}.layers.11.mul", "output")
                for i in [5, 11, 17, 23, 29]
            ],
            *[
                (
                    f"bev_decoders.0.decoder.layers.{i}._generated_add_1",
                    "output",
                )
                for i in [5, 11, 17, 23, 29]
            ],
        ],
        # Keep all listed sensitive nodes at float precision during cali search.  # noqa: E501
        topk_or_ratio=1.0,
    ),
]

# Calibration qconfig: MSEObserver for scale search during cali training.
cali_qconfig_setter = QconfigSetter(
    reference_qconfig=get_qconfig(observer=(observer_v2.MSEObserver)),
    templates=q_templates,
    enable_optimize=True,
    save_dir=os.path.join(ckpt_dir, "cali_train_qconfig"),
    custom_qconfig_mapping=None,
)

# Alias for API compatibility; sparse MapTR reuses cali qconfig (no QAT).
qat_qconfig_setter = cali_qconfig_setter

float_predict_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    converters=[
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "float-checkpoint-best.pth.tar"
            ),
            ignore_extra=True,
            verbose=True,
        ),
    ],
)

# Val callback during calibration: build QAT graph with cali qconfig only.
calibration_val_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            example_inputs=copy.deepcopy(deploy_inputs),
            qconfig_setter=cali_qconfig_setter,
        ),
    ],
)

# Predict / deploy / export: Float2QAT + calibration checkpoint.
calibration_predict_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            example_inputs=copy.deepcopy(deploy_inputs),
            qconfig_setter=cali_qconfig_setter,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "calibration-checkpoint-best.pth.tar"
            ),
            allow_miss=True,
            ignore_extra=True,
            verbose=True,
            check_hash=False,
        ),
    ],
)

# Backward-compatible alias: ``-s qat`` uses the same calibration pipeline.
qat_predict_model_convert_pipeline = copy.deepcopy(
    calibration_predict_model_convert_pipeline
)

deploy_model_convert_pipeline = copy.deepcopy(
    calibration_predict_model_convert_pipeline
)

export_model_convert_pipeline = copy.deepcopy(
    calibration_predict_model_convert_pipeline
)

compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march="NASH_M",
    name=task_name,
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    debug=True,
    input_source=["pyramid", "ddr"],
    opt="O2",
    split_dim=dict(inputs={"0": [0, 6]}),
)
