"""Quantization config and model convert pipelines for sparse multitask."""

import copy
import os

import torch
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

int16_max_point_muls = [
    f"head.det.layers.{layer}.point_mul" for layer in range(3, 39, 7)
] + [f"head.om.decoder.layers.{layer}.point_mul" for layer in range(2, 33, 6)]
int16_max_reciprocal_op = [
    f"head.det.layers.{layer}.reciprocal_op" for layer in range(3, 39, 7)
]
int16_add_ops = [f"head.det.layers.{layer}.add2" for layer in range(6, 35, 7)]
float32_ops = [
    f"head.det.layers.{layer}.cls_layers" for layer in range(6, 35, 7)
] + [f"head.det.layers.{layer}.quality_layers" for layer in range(6, 35, 7)]


# Sensitive ops promoted to qint16, explicitly enumerated from the
# quant_analysis sensitivity tables (source: nash_org/<task>/quant_analysis/
# *_L1_sensitive_ops.pt). Previously these were loaded at runtime from .pt
# files in ckpt_dir/quant_analysis via the topk_or_ratio dict below; the ops
# are now hardcoded so qconfig no longer depends on those files existing.
# Each list is the first topk rows of the corresponding sensitivity table
# (sorted high->low L1 sensitivity); sens_type is the table's fake_stub_type
# ("activation" -> SensitivityTemplate maps it to "output" internally, same as
# the .pt path). topk_or_ratio=len(list) so all listed ops are promoted.
_det_cls_sensitive_ops = [
    ("head.det.anchor_encoder.pos_fc.0", "input-0"),
    ("head.det.fc_after", "activation"),
    ("head.det.anchor_encoder.pos_fc.1", "activation"),
    ("head.det.anchor_encoder.vel_fc.0", "input-0"),
    ("head.det.layers.23.bias_add", "activation"),
    ("backbone.patch_embed.1.2", "activation"),
    ("head.det.layers.36.attn.softmax.mul", "activation"),
    ("head.det.layers.2.bias_add", "activation"),
    ("head.det.layers.29.attn.matmul", "activation"),
    ("head.det.layers.6.layers.9.bias_add", "activation"),
    ("head.det.layers.1.query_cat", "activation"),
    ("backbone.stages.3.block.5.add", "activation"),
    ("head.det.anchor_encoder.pos_fc.2.bias_add", "activation"),
    ("head.det.anchor_encoder.pos_fc.11.bias_add", "activation"),
    ("head.det.layers.15.attn.softmax.mul", "activation"),
    ("head.det.anchor_encoder.vel_fc.1", "activation"),
    ("head.det.layers.3.weight_softmax.mul", "activation"),
    ("backbone.stages.3.block.5.act.lut", "activation"),
    ("head.det.layers.6.layers.8", "activation"),
    ("head.det.layers.16.bias_add", "activation"),
]
_det_pred_sensitive_ops = [
    ("head.det.layers.41.add2", "activation"),
    ("head.det.fc_after", "activation"),
    ("head.det.anchor_encoder.pos_fc.0", "input-0"),
    ("head.det.anchor_encoder.vel_fc.0", "input-0"),
    ("backbone.patch_embed.1.2", "activation"),
    ("head.det.layers.8.attn.softmax.mul", "activation"),
    ("backbone.stages.3.block.5.add", "activation"),
    ("head.det.layers.3.weight_softmax.mul", "activation"),
    ("head.det.anchor_encoder.pos_fc.5.input_mean.mean", "activation"),
    ("head.det.anchor_encoder.pos_fc.1", "activation"),
    ("head.det.anchor_encoder.vel_fc.1", "activation"),
    ("backbone.stages.3.block.5.act.lut", "activation"),
    ("backbone.patch_embed.0.2", "activation"),
    ("backbone.stages.3.block.1.add", "activation"),
    ("backbone.patch_embed.0.0", "weight"),
    ("backbone.stages.3.block.3.act.lut", "activation"),
    ("head.det.layers.22.attn.softmax.mul", "activation"),
    ("backbone.downsample_block.2.proj.0", "activation"),
    ("backbone.stages.3.block.0.add", "activation"),
    ("backbone.patch_embed.1.0", "weight"),
]
_det_quality_sensitive_ops = [
    ("head.det.anchor_encoder.pos_fc.0", "input-0"),
    ("head.det.fc_after", "activation"),
    ("head.det.anchor_encoder.vel_fc.0", "input-0"),
    ("head.det.layers.8.attn.softmax.mul", "activation"),
    ("head.det.anchor_encoder.pos_fc.1", "activation"),
    ("head.det.anchor_encoder.pos_fc.5.input_mean.mean", "activation"),
    ("backbone.patch_embed.1.2", "activation"),
    ("head.det.anchor_encoder.vel_fc.1", "activation"),
    ("backbone.stages.3.block.5.add", "activation"),
    ("head.det.anchor_encoder.size_fc.0", "input-0"),
]
_om_pts_sensitive_ops = [
    ("head.om.decoder.layers.2.weight_softmax.mul", "activation"),
    ("head.om.decoder.layers.8.weight_softmax.mul", "activation"),
    ("head.om.decoder.layers.14.weight_softmax.mul", "activation"),
    ("backbone.patch_embed.1.2", "activation"),
    ("head.om.decoder.layers.2.feat_mul", "activation"),
]
_om_bbox_sensitive_ops = [
    ("head.om.decoder.layers.2.weight_softmax.mul", "activation"),
    ("head.om.decoder.layers.8.weight_softmax.mul", "activation"),
    ("head.om.decoder.layers.14.weight_softmax.mul", "activation"),
    ("backbone.patch_embed.1.2", "activation"),
    ("head.om.decoder.layers.2.feat_mul", "activation"),
]
_om_cls_sensitive_ops = [
    ("head.om.decoder.layers.2.weight_softmax.mul", "activation"),
    ("head.om.decoder.layers.8.weight_softmax.mul", "activation"),
    ("backbone.patch_embed.1.2", "activation"),
    ("head.om.decoder.layers.14.weight_softmax.mul", "activation"),
    ("head.om.decoder.layers.20.weight_softmax.mul", "activation"),
]

sensitivity_templates = [
    SensitivityTemplate(
        _det_cls_sensitive_ops, topk_or_ratio=len(_det_cls_sensitive_ops)
    ),
    SensitivityTemplate(
        _det_pred_sensitive_ops, topk_or_ratio=len(_det_pred_sensitive_ops)
    ),
    SensitivityTemplate(
        _det_quality_sensitive_ops,
        topk_or_ratio=len(_det_quality_sensitive_ops),
    ),
    SensitivityTemplate(
        _om_pts_sensitive_ops, topk_or_ratio=len(_om_pts_sensitive_ops)
    ),
    SensitivityTemplate(
        _om_bbox_sensitive_ops, topk_or_ratio=len(_om_bbox_sensitive_ops)
    ),
    SensitivityTemplate(
        _om_cls_sensitive_ops, topk_or_ratio=len(_om_cls_sensitive_ops)
    ),
]

q_template_core = [
    ModuleNameTemplate({"": qint8}),
    MatmulDtypeTemplate(input_dtypes=[qint8, qint8]),
    ConvDtypeTemplate(input_dtype=qint8, weight_dtype=qint8),
    ModuleNameTemplate(
        {m: {"dtype": qint8, "threshold": 1.0} for m in ["backbone.quant"]},
    ),
    ModuleNameTemplate(
        {m: {"dtype": qint16, "threshold": 1.1} for m in int16_max_point_muls},
    ),
    ModuleNameTemplate(
        {
            m: {"dtype": qint16, "threshold": 10}
            for m in int16_max_reciprocal_op
        },
    ),
    ModuleNameTemplate(
        {m: {"dtype": qint16, "threshold": 50} for m in ["head.det.fc_after"]},
    ),
    ModuleNameTemplate(
        {
            m: {"dtype": qint16, "threshold": 10}
            for m in ["head.det.fc_before"]
        },
    ),
    ModuleNameTemplate(
        {
            m: {"input": [qint16, qint16], "output": qint16}
            for m in int16_add_ops
        },
    ),
]
float32_freeze_template = ModuleNameTemplate(
    {m: None for m in float32_ops}, freeze=True
)
q_template_base = q_template_core + [float32_freeze_template]
q_templates = (
    q_template_core + sensitivity_templates + [float32_freeze_template]
)

_qconfig_setter_common = dict(
    reference_qconfig=get_qconfig(observer=(observer_v2.HistogramObserver)),
    enable_optimize=True,
    save_dir=os.path.join(ckpt_dir, "qconfig_setting"),
    custom_qconfig_mapping=None,
)

quant_analysis_qconfig_setter = QconfigSetter(
    templates=q_template_base,
    **_qconfig_setter_common,
)
calibration_qconfig_setter = QconfigSetter(
    templates=q_templates,
    **_qconfig_setter_common,
)

qat_qconfig_setter = QconfigSetter(
    reference_qconfig=get_qconfig(
        observer=(observer_v2.MinMaxObserver), fix_scale=True
    ),
    templates=q_templates,
    enable_optimize=True,
    save_dir=os.path.join(ckpt_dir, "qconfig_setting"),
    custom_qconfig_mapping=None,
)

# Example inputs for tracing (Float2Calibration / Float2QAT val). During
# trace `preparing_qat=True`, and the three heads read different metas keys:
#   - det (SparseBEVHead, compiler_model=False):
#     metas["img"] + metas["lidar2img"]
#     (gen_projection_mat; MemoryBank.get hits the self.metas is None
#     reset branch, so no timestamp/cached needed)
#   - occ (FlashOccHead/LSSTransformer, compiler_model=False):
#     metas["img"] + metas["lidar2img"]
#     (_get_homography, homo_key="lidar2img")
#   - om  (SparseMapHead, is_deploy=True via preparing_qat,
#     compiler_model=True):
#     metas["img"] + metas["projection_mat"]
#     (instance_bank num_temp_instances=0 so it returns early, no
#     cached_anchor/feature/mask needed)
# So the trace input set must carry BOTH lidar2img (det/occ) and
# projection_mat (om). We union the float example inputs
# (img/lidar2global/lidar2img) with the deploy inputs
# (projection_mat/cached_anchor/cached_feature/points0/points1);
# heads only read the keys they need, extras are harmless.
example_inputs = {
    "img": torch.randn((6, 3, 256, 704)),
    "lidar2global": torch.randn((1, 4, 4)),
    "lidar2img": torch.randn((6, 4, 4)),
    "projection_mat": deploy_inputs["projection_mat"],
    "cached_anchor": deploy_inputs["cached_anchor"],
    "cached_feature": deploy_inputs["cached_feature"],
    "points0": deploy_inputs["points0"],
    "points1": deploy_inputs["points1"],
}

float_stage1_checkpoint = os.path.join(
    ckpt_dir, "float_stage1-checkpoint-best.pth.tar"
)
float_stage2_checkpoint = os.path.join(
    ckpt_dir, "float_stage2-checkpoint-best.pth.tar"
)
# Merged checkpoint = stage1 (det + backbone) + stage2 (om + occ). Produced by
# the `float_merge` train stage (loads stage1 all + stage2 om/occ, saves
# `-last`). calibration / quant_analysis / merged predict read this complete
# checkpoint. (The Checkpoint callback's on_loop_end saves `-last`, not
# `-best`, for num_epochs=0 skip-loop stages — hence `-last` here.)
float_merged_checkpoint = os.path.join(
    ckpt_dir, "float_merged-checkpoint-last.pth.tar"
)
calibration_checkpoint = os.path.join(
    ckpt_dir, "calibration-checkpoint-last.pth.tar"
)

# Stage2 trainer reads the stage1 ckpt; calibration / quant_analysis read the
# merged ckpt (stage1 det + stage2 om/occ).
float_checkpoint = float_stage1_checkpoint

_load_checkpoint_common = dict(
    allow_miss=True,
    ignore_extra=True,
    verbose=True,
)

_calibration_load_checkpoint = dict(
    type="LoadCheckpoint",
    checkpoint_path=float_merged_checkpoint,
    load_ema_model=False,
    **_load_checkpoint_common,
)

calibration_train_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        _calibration_load_checkpoint,
        dict(
            type="Float2Calibration",
            convert_mode=convert_mode,
            example_inputs=example_inputs,
            qconfig_setter=calibration_qconfig_setter,
        ),
    ],
)

quant_analysis_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        _calibration_load_checkpoint,
        dict(
            type="Float2Calibration",
            convert_mode=convert_mode,
            example_inputs=example_inputs,
            qconfig_setter=quant_analysis_qconfig_setter,
        ),
    ],
)

calibration_val_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            # Float2QAT traces with `preparing_qat=True`, which makes every
            # head take its `compiler_model=True` branch
            # (SparseMapPerceptionDecoder sets
            # is_deploy = self.is_deploy or get_value("preparing_qat")).
            # That branch reads `metas["projection_mat"]`
            # (+ cached_anchor/feature for det, points0/1 for occ), so the
            # trace inputs must be the deploy input set — NOT the float
            # example_inputs (img/lidar2global/lidar2img) which lacks
            # projection_mat and would KeyError. Matches
            # export_model_convert_pipeline below.
            example_inputs=example_inputs,
            state="val",
            qconfig_setter=qat_qconfig_setter,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=calibration_checkpoint,
            allow_miss=True,
            ignore_extra=True,
            verbose=True,
        ),
    ],
)

float_stage1_predict_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    converters=[
        dict(
            type="LoadCheckpoint",
            checkpoint_path=float_stage1_checkpoint,
            load_ema_model=True,
            **_load_checkpoint_common,
        ),
    ],
)

float_stage2_predict_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    converters=[
        dict(
            type="LoadCheckpoint",
            checkpoint_path=float_stage2_checkpoint,
            load_ema_model=True,
            **_load_checkpoint_common,
        ),
    ],
)

float_merged_predict_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    converters=[
        dict(
            type="LoadCheckpoint",
            checkpoint_path=float_merged_checkpoint,
            load_ema_model=False,
            **_load_checkpoint_common,
        ),
    ],
)

calibration_predict_model_convert_pipeline = copy.deepcopy(
    calibration_val_model_convert_pipeline
)

export_model_convert_pipeline = dict(
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
        dict(
            type="LoadCheckpoint",
            checkpoint_path=calibration_checkpoint,
            allow_miss=True,
            ignore_extra=True,
            verbose=True,
        ),
    ],
)

deploy_model_convert_pipeline = export_model_convert_pipeline

compile_cfg = dict(
    march="NASH_M",
    name=task_name + "_model",
    input_source=["pyramid", "ddr", "ddr", "ddr", "ddr", "ddr"],
    opt="O2",
    split_dim=dict(inputs={"0": [0, 6]}),
    enable_vpu=True,
    debug=True,
)
