"""Quantization config and model convert pipelines."""

import copy
import os

import torch
from common import ckpt_dir, convert_mode, deploy_inputs, reset_prev_frame_info
from horizon_plugin_pytorch.dtype import qint8, qint16
from horizon_plugin_pytorch.quantization import (
    QConfig,
    get_qconfig,
    observer_v2,
)
from horizon_plugin_pytorch.quantization.fake_quantize import FakeQuantize
from horizon_plugin_pytorch.quantization.qconfig_setter import (
    ConvDtypeTemplate,
    MatmulDtypeTemplate,
    ModuleNameTemplate,
    QconfigSetter,
)


def set_fuse_mode(self):
    from horizon_plugin_pytorch.quantization.fx.fusion_patterns import FuseMode

    return FuseMode.BNAddReLU


QconfigSetter.get_fuse_mode = set_fuse_mode

int16_qconfig = dict()
int16_qconfig["view_transformer.quant_hybird_ref_2d"] = {"output": qint16}
int16_qconfig["view_transformer.quant_norm_coords"] = {"output": qint16}
int16_qconfig["view_transformer.quant_queries_rebatch_grid"] = {
    "output": qint16
}
int16_qconfig["view_transformer.quant_restore_bev_grid"] = {"output": qint16}
int16_qconfig["view_transformer.quant_reference_points_rebatch"] = {
    "output": qint16
}
int16_qconfig["bev_decoders.0.reference_points"] = {"output": qint16}
int16_qconfig["bev_decoders.0.sigmoid"] = {"output": qint16}
int16_qconfig["bev_decoders.0.decoder.reference_points_add1"] = {
    "output": qint16
}
int16_qconfig["bev_decoders.0.decoder.reference_points_add2"] = {
    "output": qint16
}
int16_qconfig["bev_decoders.0.decoder.reference_points_cat"] = {
    "output": qint16
}
int16_qconfig["bev_decoders.0.decoder.new_reference_points_sigmoids"] = {
    "output": qint16
}

for layer in range(3):
    int16_qconfig[f"view_transformer.encoder.layers.{layer}.sa.add_res"] = {
        "output": qint16
    }
    int16_qconfig[
        f"view_transformer.encoder.layers.{layer}.sa.output_proj"
    ] = {"output": qint16}
    int16_qconfig[
        f"view_transformer.encoder.layers.{layer}.sa.sampling_offsets"
    ] = {"output": qint16}
    int16_qconfig[
        f"view_transformer.encoder.layers.{layer}.sa.quant_shape"
    ] = {"output": qint16}
    int16_qconfig[
        f"view_transformer.encoder.layers.{layer}.sa.norm_offset"
    ] = {"output": qint16}
    int16_qconfig[f"view_transformer.encoder.layers.{layer}.sa.add_offset"] = {
        "output": qint16
    }
    int16_qconfig[f"view_transformer.encoder.layers.{layer}.sa.add1"] = {
        "output": qint16
    }
    int16_qconfig[f"view_transformer.encoder.layers.{layer}.sa.mul1"] = {
        "output": qint16
    }
    int16_qconfig[f"view_transformer.encoder.layers.{layer}.ca.add_res"] = {
        "output": qint16
    }
    int16_qconfig[
        f"view_transformer.encoder.layers.{layer}.ca.output_proj"
    ] = {"output": qint16}
    int16_qconfig[
        f"view_transformer.encoder.layers.{layer}.ca.deformable_attention.sampling_offsets"  # noqa: E501
    ] = {"output": qint16}
    int16_qconfig[
        f"view_transformer.encoder.layers.{layer}.ca.deformable_attention.quant_shape"  # noqa: E501
    ] = {"output": qint16}
    int16_qconfig[
        f"view_transformer.encoder.layers.{layer}.ca.deformable_attention.norm_offset"  # noqa: E501
    ] = {"output": qint16}
    int16_qconfig[
        f"view_transformer.encoder.layers.{layer}.ca.deformable_attention.add_offset"  # noqa: E501
    ] = {"output": qint16}
    int16_qconfig[
        f"view_transformer.encoder.layers.{layer}.ca.deformable_attention.add1"
    ] = {"output": qint16}
    int16_qconfig[
        f"view_transformer.encoder.layers.{layer}.ca.deformable_attention.mul1"
    ] = {"output": qint16}
    int16_qconfig[f"view_transformer.encoder.layers.{layer}.ffn.add"] = {
        "output": qint16
    }
    int16_qconfig[f"view_transformer.encoder.layers.{layer}.ffn.ffn2"] = {
        "output": qint16
    }

for layer in range(6):
    int16_qconfig[f"bev_decoders.0.cls_branches.{layer}.0"] = {
        "output": qint16
    }
    int16_qconfig[f"bev_decoders.0.cls_branches.{layer}.3"] = {
        "output": qint16
    }
    int16_qconfig[f"bev_decoders.0.reg_branches.{layer}.4"] = {
        "output": qint16
    }
    int16_qconfig[f"bev_decoders.0.decoder.layers.{layer}.add_self_attn"] = {
        "output": qint16
    }
    int16_qconfig[f"bev_decoders.0.decoder.layers.{layer}.ca.add_res"] = {
        "output": qint16
    }
    int16_qconfig[f"bev_decoders.0.decoder.layers.{layer}.ca.output_proj"] = {
        "output": qint16
    }
    int16_qconfig[
        f"bev_decoders.0.decoder.layers.{layer}.ca.sampling_offsets"
    ] = {"output": qint16}
    int16_qconfig[f"bev_decoders.0.decoder.layers.{layer}.ca.quant_shape"] = {
        "output": qint16
    }
    int16_qconfig[f"bev_decoders.0.decoder.layers.{layer}.ca.norm_offset"] = {
        "output": qint16
    }
    int16_qconfig[f"bev_decoders.0.decoder.layers.{layer}.ca.add_offset"] = {
        "output": qint16
    }
    int16_qconfig[f"bev_decoders.0.decoder.layers.{layer}.ca.add1"] = {
        "output": qint16
    }
    int16_qconfig[f"bev_decoders.0.decoder.layers.{layer}.ca.mul1"] = {
        "output": qint16
    }
    int16_qconfig[f"bev_decoders.0.decoder.layers.{layer}.ffn.add"] = {
        "output": qint16
    }
    int16_qconfig[f"bev_decoders.0.decoder.layers.{layer}.ffn.ffn2"] = {
        "output": qint16
    }

int16_qconfig.pop("bev_decoders.0.reg_branches.5.4")

None_qconfig_template = dict()
None_qconfig_template["view_transformer.positional_encoding"] = None
None_qconfig_template["view_transformer.bev_embedding"] = None
None_qconfig_template["bev_decoders.0.query_embedding"] = None

for layer in range(6):
    None_qconfig_template[f"bev_decoders.0.cls_branches.{layer}.2"] = {
        "activation": None
    }
    None_qconfig_template[f"bev_decoders.0.cls_branches.{layer}.5"] = {
        "activation": None
    }

q_templates = [  # 2. 模板，仅关注 dtype，按次序生效，因此模板的顺序很重要
    # 2.1. 基础配置部分
    ModuleNameTemplate({"": qint8}),  # 全局 feat int8
    ModuleNameTemplate(
        int16_qconfig | None_qconfig_template,
        freeze=True,
    ),
    MatmulDtypeTemplate(  # gemm single int8 input
        input_dtypes=[qint8, qint8],
        # prefix=["head", xxx],  # 名字和 torch.nn.Module.named_module() 返回的一致
    ),
    ConvDtypeTemplate(  # gemm int8 input
        input_dtype=qint8,
        weight_dtype=qint8,  # 成块的 weight 在这里配置
        # prefix=["head", xxx],  # 名字和 torch.nn.Module.named_module() 返回的一致
    ),
]

mix_qconifg = dict()
for k, _ in int16_qconfig.items():
    mix_qconifg[k] = QConfig(
        activation=FakeQuantize.with_args(
            observer=observer_v2.MixObserver,
            dtype=qint16,
        ),
        weight=FakeQuantize.with_args(
            observer=observer_v2.MinMaxObserver,
            dtype=qint8,
            qscheme=torch.per_channel_symmetric,
            ch_axis=0,
        ),
    )
cali_qconfig_setter = QconfigSetter(
    reference_qconfig=get_qconfig(observer=(observer_v2.MSEObserver)),
    templates=q_templates,
    enable_optimize=True,
    save_dir=ckpt_dir,
    custom_qconfig_mapping=mix_qconifg,
)

qat_qconfig_setter = QconfigSetter(
    reference_qconfig=get_qconfig(observer=(observer_v2.MinMaxObserver)),
    templates=q_templates,
    enable_optimize=True,
    save_dir=ckpt_dir,
    custom_qconfig_mapping=None,
)

from data import (  # noqa: E402
    calibration_example_data_loader,
    val_example_data_loader,
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
            ignore_extra=True,
            verbose=True,
            allow_miss=True,
        ),
        dict(type="RepModel2Deploy"),
        dict(
            type="Float2Calibration",
            convert_mode=convert_mode,
            example_data_loader=calibration_example_data_loader,
            qconfig_setter=cali_qconfig_setter,
        ),
        reset_prev_frame_info,
        dict(type="FixWeightQScale"),
        dict(type="DisableSyncConverter"),
    ],
)

qat_train_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    converters=[
        dict(type="RepModel2Deploy"),
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            example_data_loader=calibration_example_data_loader,
            qconfig_setter=qat_qconfig_setter,
        ),
        reset_prev_frame_info,
        dict(type="FixWeightQScale"),
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

calibration_val_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="RepModel2Deploy"),
        dict(
            type="Float2Calibration",
            convert_mode=convert_mode,
            example_data_loader=val_example_data_loader,
            qconfig_setter=cali_qconfig_setter,
        ),
        reset_prev_frame_info,
    ],
)

qat_val_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(type="RepModel2Deploy"),
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            example_data_loader=val_example_data_loader,
            qconfig_setter=qat_qconfig_setter,
        ),
        reset_prev_frame_info,
    ],
)

float_predict_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    converters=[
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "float-checkpoint-best.pth.tar"
            ),
            allow_miss=True,
            ignore_extra=True,
            verbose=True,
        ),
        dict(type="RepModel2Deploy"),
    ],
)

calibration_predict_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    converters=[
        dict(type="RepModel2Deploy"),
        dict(
            type="Float2Calibration",
            convert_mode=convert_mode,
            example_data_loader=val_example_data_loader,
            qconfig_setter=cali_qconfig_setter,
        ),
        reset_prev_frame_info,
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
    converters=[
        dict(type="RepModel2Deploy"),
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            example_data_loader=val_example_data_loader,
            qconfig_setter=qat_qconfig_setter,
        ),
        reset_prev_frame_info,
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "qat-checkpoint-best.pth.tar"
            ),
            allow_miss=False,
            ignore_extra=True,
            verbose=True,
        ),
    ],
)

deploy_model_convert_pipeline = copy.deepcopy(
    qat_predict_model_convert_pipeline
)
deploy_model_convert_pipeline["converters"][1].pop("example_data_loader", None)
deploy_model_convert_pipeline["converters"][1][
    "example_inputs"
] = deploy_inputs
