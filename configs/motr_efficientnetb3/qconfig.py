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

# qint16 module list, migrated from each module's set_qconfig.
_NUM_LAYERS = 6

_head_qint16 = [
    "head.query_tracked_ref_pts_quant",
    "head.ref_quant",
    "head.cat_tracked_empty_refs",
    "head.query_tracked_quant",
    "head.query_embed_weight_quant",
    "head.cat_tracked_empty_query",
    "head.transformer.sigmoid",
    "head.transformer.reference_points_encoder_quant",
]

_decoder_top_qint16 = [
    "head.transformer.decoder.new_reference_points_sigmoids",
    "head.transformer.decoder.mul_pos_scale",
    "head.transformer.decoder.reference_points_add",
]

# linear1 is ConvModule2d; qconfig applies to its Conv2d ([0]).
_enc_layer_qint16 = [
    "norm1",
    "norm2",
    "linear1.0",
    "linear2",
    "add_pos",
    "add_att",
    "add_ffn",
]
_enc_attn_qint16 = [
    "attention_weights",
    "attention_weights_mul",
    "attention_weights_sum",
    "softmax",
    "cat",
]
# MotrDeformAttn sampling_* are ModuleList, whole list qint16.
_enc_attn_qint16_lists = [
    "sampling_offsets",
    "sampling_add2",
    "sampling_mul1",
    "sampling_mul2",
    "sampling_add1",
    "sampling_mul3",
]

_dec_layer_qint16 = [
    "norm1",
    "norm2",
    "linear1.0",
    "linear2",
    "norm3",
    "self_attn.matmul",
    "self_attn.attn_matmul",
    "self_attn.out_proj",
    "update_attn.matmul",
    "update_attn.attn_matmul",
    "update_attn.out_proj",
]

_sinepos_qint16 = ["head.transformer.decoder.query_sineembed_pos"]

_qim_qint16 = [
    "track_embed.self_attn.matmul",
    "track_embed.self_attn.attn_matmul",
    "track_embed.self_attn.out_proj",
    "track_embed.linear1",
    "track_embed.linear2",
    "track_embed.linear_pos1",
    "track_embed.linear_pos2",
    "track_embed.norm1",
    "track_embed.norm2",
    "track_embed.activation1",
    "track_embed.activation2",
    "track_embed.add_q_out",
    "track_embed.add_tgt_tgt2",
    "track_embed.add_tgt_tgt2_1",
    "track_embed.add_pos_pos2",
    "track_embed.norm_pos",
    "track_embed.embed_quant",
    "track_embed.query_pos_quant",
]

qint16_names = list(_head_qint16) + list(_decoder_top_qint16)
for i in range(_NUM_LAYERS):
    for m in _enc_layer_qint16:
        qint16_names.append(f"head.transformer.encoder.layers.{i}.{m}")
    for m in _enc_attn_qint16:
        qint16_names.append(
            f"head.transformer.encoder.layers.{i}.self_attn.{m}"
        )
    for lst in _enc_attn_qint16_lists:
        qint16_names.append(
            f"head.transformer.encoder.layers.{i}.self_attn.{lst}"
        )
for i in range(_NUM_LAYERS):
    for m in _dec_layer_qint16:
        qint16_names.append(f"head.transformer.decoder.layers.{i}.{m}")
    for m in _enc_attn_qint16:
        qint16_names.append(
            f"head.transformer.decoder.layers.{i}.cross_attn.{m}"
        )
    for lst in _enc_attn_qint16_lists:
        qint16_names.append(
            f"head.transformer.decoder.layers.{i}.cross_attn.{lst}"
        )
qint16_names += _sinepos_qint16
# query_pos_scale.{i} is Sequential; last ConvModule2d is [1], Conv2d [0].
for i in range(_NUM_LAYERS - 1):
    qint16_names.append(f"head.transformer.decoder.query_pos_scale.{i}.1.0")
# bbox_embed.{i} is Sequential of 3 ConvModule2d; last is [2], Conv2d [0].
for i in range(_NUM_LAYERS):
    qint16_names.append(f"head.transformer.decoder.bbox_embed.{i}.2.0")
qint16_names += _qim_qint16

# Conv2d requires "weight" key; FloatFunctional ignores it.
qint16_templates = {
    m: {"dtype": {"output": qint16, "weight": qint8}} for m in qint16_names
}
# SinPos sin_model / cos_model stay qint8.
for i in range(_NUM_LAYERS):
    qint16_templates[
        f"head.transformer.decoder.query_sineembed_pos.{i}.sin_model"
    ] = {"dtype": {"output": qint8, "weight": qint8}}
    qint16_templates[
        f"head.transformer.decoder.query_sineembed_pos.{i}.cos_model"
    ] = {"dtype": {"output": qint8, "weight": qint8}}

# FloatFunctional [sum]/[add] ops get overridden back to qint8 by
# enable_optimize; force qint16 with a separate freeze=True template.
_force_qint16_names = []
for i in range(_NUM_LAYERS):
    _force_qint16_names.append(
        f"head.transformer.encoder.layers.{i}.self_attn.attention_weights_sum"
    )
    _force_qint16_names.append(
        f"head.transformer.decoder.layers.{i}.cross_attn.attention_weights_sum"
    )
    _force_qint16_names.append(f"head.transformer.encoder.layers.{i}.add_pos")
force_qint16_templates = {
    m: {"dtype": {"output": qint16}} for m in _force_qint16_names
}

# Pin mask QuantStub scale=1.0 via FixedScaleObserver
# (threshold=128 -> scale=128/-qint8.min=1.0). freeze=True prevents
# enable_optimize from downgrading to MinMaxObserver, whose drift
# triggers stubs.py:62 scale-conflict on val reload.
_mask_fixed_scale_names = [
    "head.query_embed_mask_quant",
    "head.query_tracked_mask_quant",
    "track_embed.query_mask_quant",
    "track_embed.self_attn.attn_mask_quant",
]
for i in range(_NUM_LAYERS):
    _mask_fixed_scale_names.append(
        f"head.transformer.decoder.layers.{i}.self_attn.attn_mask_quant"
    )
    _mask_fixed_scale_names.append(
        f"head.transformer.decoder.layers.{i}.update_attn.attn_mask_quant"
    )
mask_fixed_scale_templates = {
    n: {"dtype": qint8, "threshold": 128.0} for n in _mask_fixed_scale_names
}


qconfig_setter = QconfigSetter(
    reference_qconfig=get_qconfig(
        observer=(
            observer_v2.HistogramObserver
            if os.environ.get("HAT_TRAINING_STEP") == "calibration"
            else observer_v2.MinMaxObserver
        ),
        # fix_scale=True: QAT freezes activation scale at calib value.
        # Requires horizon-plugin-pytorch >= 3.4.1 (v3.3.0-3.4.0 silently
        # dropped averaging_constant; use the v8_callback config on older
        # plugin).
        fix_scale=os.environ.get("HAT_TRAINING_STEP") != "calibration",
    ),
    templates=[
        ModuleNameTemplate({"": qint8}, enable_propagate=True),
        # backbone input stub: threshold=1.0 -> scale=1/127, aligns with
        # efficientnet.py QuantStub(scale=1/128) from_float path.
        ModuleNameTemplate(
            {"backbone.quant": {"dtype": qint8, "threshold": 1.0}},
        ),
        MatmulDtypeTemplate(input_dtypes=[qint8, qint8]),
        ConvDtypeTemplate(input_dtype=qint8, weight_dtype=qint8),
        ModuleNameTemplate(qint16_templates),
        # Freeze decoder reference_points path so FuseConvAddTemplate can't
        # downgrade qint16 output — otherwise hbir outputs_coords range
        # collapses and tracker drifts on multi-frame seqs.
        ModuleNameTemplate(
            {
                f"head.transformer.decoder.bbox_embed.{i}.2.0": {
                    "dtype": {"output": qint16, "weight": qint8}
                }
                for i in range(_NUM_LAYERS)
            },
            freeze=True,
        ),
        ModuleNameTemplate(
            {
                "head.transformer.decoder.reference_points_add": {
                    "dtype": {"output": qint16, "weight": qint8}
                },
                "head.transformer.decoder.new_reference_points_sigmoids": {
                    "dtype": {"output": qint16, "weight": qint8}
                },
            },
            freeze=True,
        ),
        ModuleNameTemplate(force_qint16_templates, freeze=True),
        ModuleNameTemplate(mask_fixed_scale_templates, freeze=True),
    ],
    enable_optimize=True,
    save_dir=os.path.join(ckpt_dir, "qconfig_setting"),
)


calibration_train_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    # qconfig_params removed: dead code under setter path (setter builds
    # kwargs from reference_qconfig, ignores hat_engine global qconfig).
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
            qconfig_setter=qconfig_setter,
            example_inputs=example_inputs,
        ),
    ],
)

calibration_predict_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(
            type="Float2Calibration",
            convert_mode=convert_mode,
            qconfig_setter=qconfig_setter,
            example_inputs=example_inputs,
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

qat_train_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    # qconfig_params removed: setter drops averaging_constant. Scale is
    # frozen by the QATScaleFreezer callback on qat_trainer instead.
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            qconfig_setter=qconfig_setter,
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
            qconfig_setter=qconfig_setter,
            example_inputs=example_inputs,
        ),
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
