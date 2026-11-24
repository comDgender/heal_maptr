import os

import torch
from common import ckpt_dir
from data import calibration_example_data_loader
from horizon_plugin_pytorch.dtype import qint8, qint16
from horizon_plugin_pytorch.quantization import get_qconfig, observer_v2
from horizon_plugin_pytorch.quantization.qconfig_setter import *  # noqa: F401,F403,E501
from horizon_plugin_pytorch.quantization.qconfig_setter.qconfig_setter import (
    QconfigSetter,
)
from horizon_plugin_pytorch.quantization.qconfig_setter.templates.equalize_inout_scale_template import (  # noqa: E501
    EqualizeInOutScaleTemplate,
)
from horizon_plugin_pytorch.quantization.qconfig_setter.templates.gemm_dtype_template import (  # noqa: E501
    ConvDtypeTemplate,
    MatmulDtypeTemplate,
)
from horizon_plugin_pytorch.quantization.qconfig_setter.templates.module_name_template import (  # noqa: E501
    ModuleNameTemplate,
)
from horizon_plugin_pytorch.quantization.qconfig_setter.templates.sensivity_template import (  # noqa: E501
    SensitivityTemplate,
)

sensitive_path1 = os.path.join(
    ckpt_dir, "analysis", "output_prob_L1_sensitive_ops.pt"
)
sensitive_path2 = os.path.join(
    ckpt_dir, "analysis", "output_pred_L1_sensitive_ops.pt"
)

is_calib = True
QINT16_MAX = 32767
QINT8_MAX = 128

set_custom = {
    "output_pred_L1_sensitive_ops": 120,
    "output_prob_L1_sensitive_ops": 80,
}
output_pred_L1_sensitive_ops = os.path.join(
    ckpt_dir, "analysis", "output_pred_L1_sensitive_ops.pt"
)
output_prob_L1_sensitive_ops = os.path.join(
    ckpt_dir, "analysis", "output_prob_L1_sensitive_ops.pt"
)

if not os.path.exists(output_pred_L1_sensitive_ops) or not os.path.exists(
    output_prob_L1_sensitive_ops
):
    print("WARNING: Sensitive operation analysis files not found!")
    feak_qconfig_setter = QconfigSetter(
        reference_qconfig=get_qconfig(observer=(observer_v2.MSEObserver)),
        templates=[
            ModuleNameTemplate(
                {"": {"output": qint8, "weight": qint8}},
                enable_propagate=True,
            ),
            MatmulDtypeTemplate(
                input_dtypes=[qint8, qint8],
            ),
            ConvDtypeTemplate(
                input_dtype=qint8,
                weight_dtype=qint8,
            ),
        ],
        enable_optimize=True,
        save_dir="./qconfig_setting",
        optimize_templates_kwargs={
            EqualizeInOutScaleTemplate: {"only_fuse_relu_after_conv": True},
        },
    )
    cali_qconfig_setter = feak_qconfig_setter
    qat_qconfig_setter = feak_qconfig_setter
    print("WARNING: Sensitive operation analysis files not found!")
else:
    cali_qconfig_setter = QconfigSetter(
        reference_qconfig=get_qconfig(
            observer=(
                observer_v2.MSEObserver
                # observer_v2.HistogramObserver
            )
        ),
        templates=[
            # 2.1. 基础配置部分
            ModuleNameTemplate(
                {"": {"output": qint8, "weight": qint8}},
                enable_propagate=True,  # 全局 output fp16，weight int8
            ),
            ModuleNameTemplate(
                {
                    "backbone.quant": {"dtype": qint8, "threshold": 1.0},
                    "side_backbone.quant": {"dtype": qint8, "threshold": 1.0},
                },  # quant int8，固定 scale 方式 1
                freeze=True,  # 可以选择强制生效，不会被其他模板修改
            ),
            MatmulDtypeTemplate(  # gemm single int8 input
                input_dtypes=[qint8, qint8],
            ),
            ConvDtypeTemplate(  # gemm int8 input
                input_dtype=qint8,
                weight_dtype=qint8,  # 成块的 weight 在这里配置
            ),
            # 2.2. 精度调优部分
            SensitivityTemplate(
                sensitive_table=output_pred_L1_sensitive_ops,
                topk_or_ratio=set_custom["output_pred_L1_sensitive_ops"],
            ),
            SensitivityTemplate(
                sensitive_table=output_prob_L1_sensitive_ops,
                topk_or_ratio=set_custom["output_prob_L1_sensitive_ops"],
            ),
            ModuleNameTemplate(
                {
                    "decoder.loc_cumsum_conv": {"output": qint16},
                    # x_a_his 需要和输出 output_0 dtype一致
                    "encoder.agent_encoder.x_a_his_quant": {"output": qint16},
                    "encoder.agent_encoder.x_a_emb.freqs.0": {
                        "dtype": {"input": qint16, "output": qint16}
                    },
                    "encoder.agent_encoder.x_a_emb.freqs.2": {
                        "dtype": {"input": qint16, "output": qint16}
                    },
                    "encoder.agent_encoder.x_a_emb._generated_mul_0": {
                        "output": qint16
                    },
                    "encoder.agent_encoder.x_a_emb._generated_mul_2": {
                        "output": qint16
                    },
                    "encoder.agent_encoder.r_t_quant.0": {"output": qint16},
                    "encoder.agent_encoder.r_t_quant.3": {"output": qint16},
                    "encoder.agent_encoder.x_a_quant.0": {"output": qint16},
                    "decoder.locs_propose_pos_cat": {"output": qint16},
                    "encoder.agent_encoder.mid_his_quant.0": {
                        "output": qint16,
                        "threshold": 146.0,
                    },
                    # "encoder.agent_encoder.mid_his_quant.0": {
                    #    "threshold": 146.0
                    # },
                    "encoder.agent_encoder.x_a_emb.to_out.2": {
                        "output": qint16,
                        "threshold": 146.0,
                    },
                    # "encoder.agent_encoder.x_a_emb.to_out.2": {
                    #    "threshold": 146.0
                    # },
                },
                freeze=True,  # 可以选择强制生效，不会被其他模板修改
                enable_propagate=True,
            ),
        ],
        enable_optimize=True,  # False, True, 或与旧模板一致的兼容配置（改成 ENUM？）
        save_dir="./qconfig_setting",  # qconfig 保存路径，有默认值，用户可以改
        optimize_templates_kwargs={
            EqualizeInOutScaleTemplate: {"only_fuse_relu_after_conv": True},
        },
    )

    qat_qconfig_setter = QconfigSetter(
        reference_qconfig=get_qconfig(observer=(observer_v2.MinMaxObserver)),
        templates=[
            # 2.1. 基础配置部分
            ModuleNameTemplate(
                {"": {"output": qint8, "weight": qint8}},
                enable_propagate=True,  # 全局 output fp16，weight int8
            ),
            ModuleNameTemplate(
                {
                    "backbone.quant": {"dtype": qint8, "threshold": 1.0},
                    "side_backbone.quant": {"dtype": qint8, "threshold": 1.0},
                },  # quant int8，固定 scale 方式 1
                freeze=True,  # 可以选择强制生效，不会被其他模板修改
            ),
            MatmulDtypeTemplate(  # gemm single int8 input
                input_dtypes=[qint8, qint8],
            ),
            ConvDtypeTemplate(  # gemm int8 input
                input_dtype=qint8,
                weight_dtype=qint8,  # 成块的 weight 在这里配置
            ),
            # 2.2. 精度调优部分
            SensitivityTemplate(
                sensitive_table=output_pred_L1_sensitive_ops,
                topk_or_ratio=set_custom["output_pred_L1_sensitive_ops"],
            ),
            SensitivityTemplate(
                sensitive_table=output_prob_L1_sensitive_ops,
                topk_or_ratio=set_custom["output_prob_L1_sensitive_ops"],
            ),
            ModuleNameTemplate(
                {
                    "decoder.loc_cumsum_conv": {"output": qint16},
                    # x_a_his 需要和输出 output_0 dtype一致
                    "encoder.agent_encoder.x_a_his_quant": {"output": qint16},
                    "encoder.agent_encoder.x_a_emb.freqs.0": {
                        "dtype": {"input": qint16, "output": qint16}
                    },
                    "encoder.agent_encoder.x_a_emb.freqs.2": {
                        "dtype": {"input": qint16, "output": qint16}
                    },
                    "encoder.agent_encoder.x_a_emb._generated_mul_0": {
                        "output": qint16
                    },
                    "encoder.agent_encoder.x_a_emb._generated_mul_2": {
                        "output": qint16
                    },
                    "encoder.agent_encoder.r_t_quant.0": {"output": qint16},
                    "encoder.agent_encoder.r_t_quant.3": {"output": qint16},
                    "encoder.agent_encoder.x_a_quant.0": {"output": qint16},
                    "decoder.locs_propose_pos_cat": {"output": qint16},
                    "encoder.agent_encoder.mid_his_quant.0": {
                        "output": qint16,
                        "threshold": 146.0,
                    },
                    # "encoder.agent_encoder.mid_his_quant.0": {
                    #    "threshold": 146.0
                    # },
                    "encoder.agent_encoder.x_a_emb.to_out.2": {
                        "output": qint16,
                        "threshold": 146.0,
                    },
                    # "encoder.agent_encoder.x_a_emb.to_out.2": {
                    #    "threshold": 146.0
                    # },
                },
                freeze=True,  # 可以选择强制生效，不会被其他模板修改
                enable_propagate=True,
            ),
        ],
        enable_optimize=True,  # False, True, 或与旧模板一致的兼容配置（改成 ENUM？）
        save_dir="./qconfig_setting",  # qconfig 保存路径，有默认值，用户可以改
        optimize_templates_kwargs={
            EqualizeInOutScaleTemplate: {"only_fuse_relu_after_conv": True},
        },
    )


class FixScaleConverter:
    def __init__(self, prefixes=None):
        if prefixes is not None:
            assert isinstance(prefixes, (list, tuple))
        self.prefixes = prefixes

    def _fix_fq_scale(self, model: torch.nn.Module):
        from horizon_plugin_pytorch.quantization.fake_quantize_base import (
            FakeQuantizeBase,
        )

        for _, m in model.named_modules():
            if isinstance(m, FakeQuantizeBase):
                if m.ch_axis < 0:
                    m.fix_params()

    def __call__(self, model: torch.nn.Module):
        if self.prefixes is None:
            self._fix_fq_scale(model)
        else:
            for name in self.prefixes:
                self._fix_fq_scale(model.get_submodule(name))
        return model


float2calibration = dict(
    type="Float2Calibration",
    convert_mode="jit-strip",
    example_data_loader=calibration_example_data_loader,
    qconfig_setter=cali_qconfig_setter,
)

float2qat = dict(
    type="Float2QAT",
    convert_mode="jit-strip",
    example_data_loader=calibration_example_data_loader,
    qconfig_setter=qat_qconfig_setter,
)
