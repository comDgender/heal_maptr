"""Quantization config and model convert pipelines."""

import os

from common import ckpt_dir, convert_mode, qat_mode
from horizon_plugin_pytorch.dtype import qint8, qint16
from horizon_plugin_pytorch.quantization.fake_quantize_base import (
    FakeQuantizeBase,
)

FakeQuantizeBase._enable_fp16_compute = False
from horizon_plugin_pytorch.quantization import (  # noqa: E402
    get_qconfig,
    observer_v2,
)
from horizon_plugin_pytorch.quantization.qconfig_setter import (  # noqa: E402
    ConvDtypeTemplate,
    MatmulDtypeTemplate,
    ModuleNameTemplate,
    QconfigSetter,
    SensitivityTemplate,
)
from model import eval_trace_inputs  # noqa: E402

int16_max_point_muls = [
    f"head.layers.{layer}.point_mul" for layer in range(3, 39, 7)
]
int16_max_reciprocal_op = [
    f"head.layers.{layer}.reciprocal_op" for layer in range(3, 39, 7)
]
int16_point_cat = [
    f"head.layers.{layer}.point_cat" for layer in range(3, 39, 7)
]
float32_ops = [
    f"head.layers.{layer}.cls_layers" for layer in range(6, 35, 7)
] + [f"head.layers.{layer}.quality_layers" for layer in range(6, 35, 7)]


def get_qconfig_setter(is_calibration=True):
    return QconfigSetter(
        reference_qconfig=get_qconfig(
            observer=(
                observer_v2.MSEObserver
                if is_calibration
                else observer_v2.MinMaxObserver
            ),
            fix_scale=False,
        ),
        templates=[
            ModuleNameTemplate({"": qint8}),
            MatmulDtypeTemplate(input_dtypes=[qint8, qint8], prefix=None),
            ConvDtypeTemplate(
                input_dtype=qint8,
                weight_dtype=qint8,
                prefix=None,
            ),
            ModuleNameTemplate(
                {
                    m: {"dtype": qint16, "threshold": 1.1}
                    for m in int16_max_point_muls
                }
            ),
            ModuleNameTemplate(
                {
                    m: {"dtype": qint16, "threshold": 11}
                    for m in int16_max_reciprocal_op
                }
            ),
            ModuleNameTemplate(
                {
                    m: {"dtype": qint16, "threshold": 60}
                    for m in int16_point_cat
                }
            ),
            ModuleNameTemplate(
                {
                    "backbone.quant": {"dtype": qint8, "threshold": 1.0},
                    "head.fc_before": {"dtype": qint16, "threshold": 5.0},
                    "head.fc_after": {"dtype": qint16, "threshold": 50.0},
                    "head.instance_bank.anchor_quant_stub": {
                        "dtype": qint16,
                        "threshold": 60,
                    },
                    "head.instance_bank.anchor_cat": {"output": qint16},
                    "head.layers.41.layers.11.scale_quant_stub": {
                        "output": qint16
                    },
                    "head.layers.3.kps_generator.offset": {"output": qint16},
                    "head.layers.10.kps_generator.offset": {"output": qint16},
                    "head.layers.17.kps_generator.offset": {"output": qint16},
                    "head.layers.24.kps_generator.offset": {"output": qint16},
                    "head.layers.31.kps_generator.offset": {"output": qint16},
                    "head.layers.38.kps_generator.offset": {"output": qint16},
                    "head.layers.3.kps_generator.keypoints_add": {
                        "output": qint16
                    },
                    "head.layers.10.kps_generator.keypoints_add": {
                        "output": qint16
                    },
                    "head.layers.17.kps_generator.keypoints_add": {
                        "output": qint16
                    },
                    "head.layers.24.kps_generator.keypoints_add": {
                        "output": qint16
                    },
                    "head.layers.31.kps_generator.keypoints_add": {
                        "output": qint16
                    },
                    "head.layers.38.kps_generator.keypoints_add": {
                        "output": qint16
                    },
                    "head.layers.3.point_quant_stub": {"output": qint16},
                    "head.layers.10.point_quant_stub": {"output": qint16},
                    "head.layers.17.point_quant_stub": {"output": qint16},
                    "head.layers.24.point_quant_stub": {"output": qint16},
                    "head.layers.31.point_quant_stub": {"output": qint16},
                    "head.layers.38.point_quant_stub": {"output": qint16},
                }
            ),
            ModuleNameTemplate({m: None for m in float32_ops}, freeze=True),
            SensitivityTemplate(
                [
                    ("head.anchor_encoder.vel_fc.1", "output"),
                    ("backbone.stages.3.block.5.act.lut", "output"),
                    ("backbone.stages.3.block.5.add", "output"),
                    ("backbone.stage_norm.3", "output"),
                    ("head.fc_before", "output"),
                    ("head.fc_after", "output"),
                    ("head.instance_bank.anchor_quant_stub", "output"),
                    ("head.instance_bank.anchor_cat", "output"),
                    ("head.layers.41.layers.11.scale_quant_stub", "output"),
                    ("head.layers.41.layers.11.mul", "output"),
                    ("head.layers.41.add2", "output"),
                    ("head.layers.38.camera_encoder.0", "input"),
                    ("head.layers.31.camera_encoder.0", "input"),
                    ("head.layers.24.camera_encoder.0", "input"),
                    ("head.layers.17.camera_encoder.0", "input"),
                    ("head.layers.3.camera_encoder.0", "input"),
                    ("head.layers.10.camera_encoder.0", "input"),
                    ("head.layers.3.point_cat", "output"),
                    ("head.layers.10.point_cat", "output"),
                    ("head.layers.17.point_cat", "output"),
                    ("head.layers.24.point_cat", "output"),
                    ("head.layers.31.point_cat", "output"),
                    ("head.layers.38.point_cat", "output"),
                    ("head.layers.3.point_matmul", "output"),
                    ("head.layers.10.point_matmul", "output"),
                    ("head.layers.17.point_matmul", "output"),
                    ("head.layers.24.point_matmul", "output"),
                    ("head.layers.31.point_matmul", "output"),
                    ("head.layers.38.point_matmul", "output"),
                    ("head.layers.3.point_matmul", "input-0"),
                    ("head.layers.10.point_matmul", "input-0"),
                    ("head.layers.17.point_matmul", "input-0"),
                    ("head.layers.24.point_matmul", "input-0"),
                    ("head.layers.31.point_matmul", "input-0"),
                    ("head.layers.38.point_matmul", "input-0"),
                    ("head.layers.3.point_matmul", "input-1"),
                    ("head.layers.10.point_matmul", "input-1"),
                    ("head.layers.17.point_matmul", "input-1"),
                    ("head.layers.24.point_matmul", "input-1"),
                    ("head.layers.31.point_matmul", "input-1"),
                    ("head.layers.38.point_matmul", "input-1"),
                    ("head.layers.6.layers.10", "output"),
                    ("head.layers.13.layers.10", "output"),
                    ("head.layers.20.layers.10", "output"),
                    ("head.layers.27.layers.10", "output"),
                    ("head.layers.34.layers.10", "output"),
                    ("head.layers.6.layers.11.scale_quant_stub", "output"),
                    ("head.layers.13.layers.11.scale_quant_stub", "output"),
                    ("head.layers.20.layers.11.scale_quant_stub", "output"),
                    ("head.layers.27.layers.11.scale_quant_stub", "output"),
                    ("head.layers.34.layers.11.scale_quant_stub", "output"),
                    ("head.layers.6.layers.11.mul", "output"),
                    ("head.layers.13.layers.11.mul", "output"),
                    ("head.layers.20.layers.11.mul", "output"),
                    ("head.layers.27.layers.11.mul", "output"),
                    ("head.layers.34.layers.11.mul", "output"),
                    ("head.layers.6.add2", "output"),
                    ("head.layers.13.add2", "output"),
                    ("head.layers.20.add2", "output"),
                    ("head.layers.27.add2", "output"),
                    ("head.layers.34.add2", "output"),
                ],
                topk_or_ratio=1.0,
            ),
        ],
        enable_optimize=True,
        save_dir="./qconfig_setting",
        custom_qconfig_mapping=None,
    )


calibration_qconfig_setter = get_qconfig_setter(True)

calibration_train_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "float-checkpoint-best.pth.tar"
            ),
            load_ema_model=True,
            allow_miss=True,
            ignore_extra=True,
            verbose=True,
        ),
        dict(
            type="Float2Calibration",
            convert_mode=convert_mode,
            example_inputs=eval_trace_inputs,
            qconfig_setter=calibration_qconfig_setter,
        ),
        dict(type="DisableSyncConverter"),
    ],
)

# Pattern A: build the calibration quantized graph from the float test_model
# (no LoadCheckpoint — the weights, including the observer scales produced
# during calibration, are loaded into this graph from the live training model
# via `init_with_train_model=True` at each validation). Mirrors the
# bevformer_tiny_resnet50_detection calibration_val pipeline shape.
calibration_val_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(
            type="Float2Calibration",
            convert_mode=convert_mode,
            example_inputs=eval_trace_inputs,
            qconfig_setter=calibration_qconfig_setter,
        ),
    ],
)

# Calibration inference/predict pipeline: build the calibration quantized
# graph from the float model, then load the calibration product (fake-quant
# weights + observer scales from calibration-checkpoint-last). Used by
# tools/inference.py -s calibration and predict.py's calibration_predictor.
calibration_predict_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(
            type="Float2Calibration",
            convert_mode=convert_mode,
            example_inputs=eval_trace_inputs,
            qconfig_setter=calibration_qconfig_setter,
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

compile_cfg = dict(
    march="NASH_M",
    name="bev_sparse_henet_tinym_model",
    out_dir=os.path.join(ckpt_dir, "compile"),
    hbm=os.path.join(ckpt_dir, "compile", "model.hbm"),
    layer_details=True,
    debug=True,
    input_source="pyramid, ddr, ddr, ddr, ddr, ddr",
    opt="O2",
    split_dim=dict(inputs={"0": [0, 6]}),
)

# Deploy/export pipeline: build a QAT (fake-quant) graph from the float
# `deploy_model` using the calibration qconfig setter (same observer/qconfig as
# calibration — bev_sparse has no separate qat stage), then load the
# calibration product (fake-quant weights + observer scales). HbirExporter
# converts this fake-quant model to a quantized hbir. Mirrors the
# flashocc_henet_lss_occ3d deploy pipeline (qat_qconfig_setter =
# cali_qconfig_setter pattern).
deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            example_inputs=eval_trace_inputs,
            state="val",
            qconfig_setter=calibration_qconfig_setter,
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
