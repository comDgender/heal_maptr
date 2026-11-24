"""Quantization config and model convert pipelines."""

import copy
import os

from common import ckpt_dir, convert_mode, march, qat_mode, task_name
from horizon_plugin_pytorch.dtype import qint8, qint16
from horizon_plugin_pytorch.quantization import get_qconfig, observer_v2
from horizon_plugin_pytorch.quantization.qconfig_setter import (
    ConvDtypeTemplate,
    MatmulDtypeTemplate,
    ModuleNameTemplate,
    QconfigSetter,
    SensitivityTemplate,
)
from model import deploy_inputs, eval_trace_inputs

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
int16_inputs = (
    [
        "head.anchor_encoder.pos_fc.0",
        "head.anchor_encoder.size_fc.0",
        "head.anchor_encoder.yaw_fc.0",
        "head.anchor_encoder.vel_fc.0",
    ]
    + [f"head.layers.{i}.camera_encoder.0" for i in range(3, 39, 7)]
    + [
        f"head.layers.{i}.attn.v_proj"
        for i in [0, 1, 7, 8, 14, 15, 21, 22, 28, 29, 35, 36]
    ]
)


# Most sensitive ops across the classification/prediction outputs, inlined from
# the quant_analysis tables (output_classification_L1_sensitive_ops.pt /
# output_prediction_L1_sensitive_ops.pt under ckpt_dir/quant_analysis) so the
# qconfig no longer loads .pt files at runtime. Each entry is
# (module_name, io_type); kept in descending sensitivity order. Duplicates
# across outputs are kept once.
sensitive_ops = [
    ("lidar_net.neck.conv_add.0", "activation"),
    ("lidar_net.backbone.stages.2.block.7.pwconv1.1.lut", "activation"),
    ("lidar_net.backbone.patch_embed.1.2", "activation"),
    ("lidar_net.backbone.patch_embed.0.2", "activation"),
    ("lidar_net.neck.conv_add.1", "activation"),
    ("lidar_net.neck.conv_extract.3.0", "activation"),
    ("lidar_net.backbone.stages.2.block.7.add", "activation"),
    ("head.layers.29.attn.matmul", "activation"),
    ("lidar_net.backbone.patch_embed.1.0", "weight"),
    ("lidar_net.backbone.stages.3.block.5.add", "activation"),
    ("lidar_net.backbone.stages.0.block.0.pwconv1", "activation"),
    ("lidar_net.backbone.stage_norm.2", "activation"),
    ("lidar_net.neck.fpn_conv.2.0", "activation"),
    ("lidar_net.neck.fpn_conv.1.0", "activation"),
    ("head.anchor_encoder.vel_fc.1", "activation"),
    ("lidar_net.backbone.stages.0.block.0.dwconv.0", "activation"),
    ("lidar_net.backbone.stages.0.block.0.add", "activation"),
    ("head.layers.22.attn.matmul", "activation"),
    ("head.anchor_encoder.cat", "activation"),
    ("head.anchor_encoder.pos_fc.11.bias_add", "activation"),
    ("lidar_net.backbone.stages.0.block.3.add", "activation"),
    ("lidar_net.reader.pfn_layers.0.relu", "activation"),
    ("head.layers.1.attn.softmax.mul", "activation"),
    ("head.layers.41.add2", "activation"),
    ("lidar_net.reader.pfn_layers.0.linear", "weight"),
    ("lidar_net.backbone.stages.0.block.3.dwconv.0", "activation"),
]


q_template = [
    ModuleNameTemplate({"": qint8}),
    MatmulDtypeTemplate(input_dtypes=[qint8, qint8], prefix=None),
    ConvDtypeTemplate(
        input_dtype=qint8,
        weight_dtype=qint8,
        prefix=None,
    ),
    ModuleNameTemplate(
        {m: {"dtype": qint16, "threshold": 1.1} for m in int16_max_point_muls}
    ),
    ModuleNameTemplate(
        {
            m: {"dtype": qint16, "threshold": 11}
            for m in int16_max_reciprocal_op
        }
    ),
    ModuleNameTemplate(
        {m: {"dtype": qint16, "threshold": 60} for m in int16_point_cat}
    ),
    ModuleNameTemplate(
        {
            "backbone.quant": {"dtype": qint8, "threshold": 1.0},
            "head.fc_before": {
                "dtype": {"input": qint16, "output": qint16},
                "threshold": {"output": 5},
            },
            "head.fc_after": {"dtype": qint16, "threshold": 50.0},
            "head.mat_quant_stub": {"output": qint16},
            "head.instance_bank.anchor_quant_stub": {
                "dtype": qint16,
                "threshold": 60,
            },
            "head.instance_bank.anchor_cat": {"output": qint16},
            "head.layers.3.kps_generator.offset": {"output": qint16},
            "head.layers.10.kps_generator.offset": {"output": qint16},
            "head.layers.17.kps_generator.offset": {"output": qint16},
            "head.layers.24.kps_generator.offset": {"output": qint16},
            "head.layers.31.kps_generator.offset": {"output": qint16},
            "head.layers.38.kps_generator.offset": {"output": qint16},
            "head.layers.3.kps_generator.keypoints_add": {"output": qint16},
            "head.layers.10.kps_generator.keypoints_add": {"output": qint16},
            "head.layers.17.kps_generator.keypoints_add": {"output": qint16},
            "head.layers.24.kps_generator.keypoints_add": {"output": qint16},
            "head.layers.31.kps_generator.keypoints_add": {"output": qint16},
            "head.layers.38.kps_generator.keypoints_add": {"output": qint16},
            "head.layers.3.point_quant_stub": {"output": qint16},
            "head.layers.10.point_quant_stub": {"output": qint16},
            "head.layers.17.point_quant_stub": {"output": qint16},
            "head.layers.24.point_quant_stub": {"output": qint16},
            "head.layers.31.point_quant_stub": {"output": qint16},
            "head.layers.38.point_quant_stub": {"output": qint16},
        }
    ),
    ModuleNameTemplate(
        {m: {"input": qint16, "output": qint8} for m in int16_inputs}
    ),
    SensitivityTemplate(sensitive_ops, topk_or_ratio=1.0),
    ModuleNameTemplate({m: None for m in float32_ops}, freeze=True),
]

_qconfig_setter_common = dict(
    enable_optimize=True,
    save_dir=os.path.join(ckpt_dir, "qconfig_setting"),
    custom_qconfig_mapping=None,
)

calibration_qconfig_setter = QconfigSetter(
    reference_qconfig=get_qconfig(observer=(observer_v2.HistogramObserver)),
    templates=q_template,
    **_qconfig_setter_common,
)
qat_qconfig_setter = QconfigSetter(
    reference_qconfig=get_qconfig(
        observer=(observer_v2.MinMaxObserver), fix_scale=True
    ),
    templates=q_template,
    **_qconfig_setter_common,
)

float_checkpoint = os.path.join(ckpt_dir, "float-checkpoint-best.pth.tar")
calibration_checkpoint = os.path.join(
    ckpt_dir, "calibration-checkpoint-last.pth.tar"
)

_load_checkpoint_common = dict(
    allow_miss=True,
    ignore_extra=True,
    verbose=True,
)

_calibration_load_checkpoint = dict(
    type="LoadCheckpoint",
    checkpoint_path=float_checkpoint,
    **_load_checkpoint_common,
)

_qconfig_params = dict(
    activation_calibration_observer="mse",
)

calibration_train_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    qconfig_params=_qconfig_params,
    converters=[
        _calibration_load_checkpoint,
        dict(
            type="Float2Calibration",
            convert_mode=convert_mode,
            example_inputs=eval_trace_inputs,
            qconfig_setter=calibration_qconfig_setter,
        ),
        dict(type="DisableSyncConverter"),
    ],
)

# Build a fake-quant analysis model straight from the float checkpoint with the
# calibration qconfig. quant_analysis only finds bad cases / verifies the
# qconfig (sensitive ops are inlined above), so it no longer depends on a
# prior calibration checkpoint and can run any time after float training.
quant_analysis_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        _calibration_load_checkpoint,
        dict(
            type="Float2Calibration",
            convert_mode=convert_mode,
            example_inputs=eval_trace_inputs,
            qconfig_setter=calibration_qconfig_setter,
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
            example_inputs=eval_trace_inputs,
            state="val",
            qconfig_setter=qat_qconfig_setter,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=calibration_checkpoint,
            **_load_checkpoint_common,
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
            example_inputs=eval_trace_inputs,
            state="train",
            qconfig_setter=qat_qconfig_setter,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=calibration_checkpoint,
            allow_miss=True,
            ignore_extra=True,
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
            example_inputs=eval_trace_inputs,
            state="val",
            qconfig_setter=qat_qconfig_setter,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "qat-checkpoint-best.pth.tar"
            ),
            allow_miss=True,
            ignore_extra=True,
        ),
    ],
)


def state_dict_update_func(state_dict):
    new_state_dict = copy.deepcopy(state_dict)
    for k in list(new_state_dict.keys()):
        if "weights_quant_stub" in k:
            new_state_dict.pop(k)
    return new_state_dict


export_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            example_inputs=deploy_inputs,
            state="train",
            qconfig_setter=qat_qconfig_setter,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "qat-checkpoint-best.pth.tar"
            ),
            state_dict_update_func=state_dict_update_func,
            allow_miss=True,
            ignore_extra=True,
            verbose=True,
        ),
    ],
)

compile_cfg = dict(
    march=march,
    name=task_name + "_model",
    out_dir=os.path.join(ckpt_dir, "compile"),
    hbm=os.path.join(ckpt_dir, "compile", "model.hbm"),
    layer_details=True,
    debug=True,
    input_source="pyramid, ddr, ddr, ddr, ddr, ddr",
    opt="O2",
    split_dim=dict(inputs={"0": [0, 6]}),
    enable_vpu=True,
)

deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            example_inputs=deploy_inputs,
            state="val",
            qconfig_setter=qat_qconfig_setter,
        ),
    ],
)
