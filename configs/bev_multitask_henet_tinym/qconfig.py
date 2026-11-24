# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

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

int16_add_ops = [
    f"task_decoders.dynamic_objects_det.layers.{layer}.add2"
    for layer in range(6, 35, 7)
] + [
    f"task_decoders.static_objects_det.layers.{layer}.add2"
    for layer in range(6, 35, 7)
]
float32_ops = (
    [
        f"task_decoders.dynamic_objects_det.layers.{layer}.cls_layers"
        for layer in range(6, 35, 7)
    ]
    + [
        f"task_decoders.dynamic_objects_det.layers.{layer}.quality_layers"
        for layer in range(6, 35, 7)
    ]
    + [
        f"task_decoders.static_objects_det.layers.{layer}.cls_layers"
        for layer in range(6, 35, 7)
    ]
    + [
        f"task_decoders.static_objects_det.layers.{layer}.quality_layers"
        for layer in range(6, 35, 7)
    ]
)

sensitive_ops = {
    "task_decoders.dynamic_objects_det.anchor_encoder.vel_fc.0": {
        "input": [qint16]
    },
    "task_decoders.static_objects_det.anchor_encoder.vel_fc.0": {
        "input": [qint16]
    },
    "task_decoders.static_objects_det.layers.36.attn.softmax.mul": {
        "output": qint16
    },
    "task_decoders.occ.occ_head.predicter.2": {
        "output": qint16,
        "weight": qint16,
    },
    "task_decoders.static_objects_det.anchor_encoder.pos_fc.0": {
        "input": [qint16],
        "weight": qint16,
    },
    "camera_net.backbone.stages.0.block.1.dwconv.0": {
        "weight": qint16,
        "output": qint16,
    },
    "task_decoders.dynamic_objects_det.anchor_encoder.pos_fc.0": {
        "input": [qint16],
        "weight": qint16,
    },
    "camera_net.backbone.stages.3.block.5.act.lut": {"output": qint16},
    "task_decoders.dynamic_objects_det.anchor_encoder.vel_fc.1": {
        "output": qint16
    },
    "camera_net.backbone.patch_embed.0.2": {"output": qint16},
    "task_decoders.static_objects_det.layers.41.add2": {"output": qint16},
    "camera_net.backbone.stages.3.block.2.pwconv2": {"weight": qint16},
    "camera_net.backbone.patch_embed.1.0": {"weight": qint16},
    "task_decoders.dynamic_objects_det.layers.41.add2": {"output": qint16},
    "task_decoders.static_objects_det.anchor_encoder.pos_fc.1": {
        "output": qint16
    },
    "task_decoders.dynamic_objects_det.anchor_encoder.pos_fc.1": {
        "output": qint16
    },
    "camera_net.backbone.stages.3.block.3.pwconv2": {"weight": qint16},
    "task_decoders.static_objects_det.layers.8.attn.softmax.mul": {
        "output": qint16
    },
    "task_decoders.occ.occ_head.predicter.1.softplus": {"output": qint16},
    "task_decoders.static_objects_det.layers.15.attn.softmax.mul": {
        "output": qint16
    },
    "camera_net.backbone.stages.3.block.4.pwconv2": {"weight": qint16},
    "camera_net.backbone.stages.3.block.0.pwconv2": {"weight": qint16},
    "task_decoders.dynamic_objects_det.layers.1.attn.softmax.mul": {
        "output": qint16
    },
    "camera_net.backbone.stages.3.block.5.add": {"output": qint16},
    "camera_net.backbone.patch_embed.0.0": {"weight": qint16},
    "camera_net.view_transformer.encoder.layers.0.ffn.add": {"output": qint16},
    "camera_net.backbone.stages.0.block.0.pwconv2": {"weight": qint16},
    "camera_net.backbone.stages.3.block.0.add": {"output": qint16},
    "camera_net.backbone.stages.3.block.2.act.lut": {"output": qint16},
    "task_decoders.dynamic_objects_det.layers.29.attn.softmax.mul": {
        "output": qint16
    },
    "camera_net.backbone.patch_embed.1.2": {"output": qint16},
    "task_decoders.static_objects_det.layers.22.attn.softmax.mul": {
        "output": qint16
    },
    "camera_net.backbone.stages.3.block.1.pwconv2": {"weight": qint16},
    "camera_net.backbone.stages.3.block.4.act.lut": {"output": qint16},
    "task_decoders.occ.occ_head.predicter.0": {
        "output": qint16,
        "weight": qint16,
    },
    "camera_net.backbone.stages.3.block.0.act.lut": {"output": qint16},
    "camera_net.backbone.stages.0.block.3.dwconv.0": {
        "weight": qint16,
        "output": qint16,
    },
    "task_decoders.dynamic_objects_det.layers.36.attn.softmax.mul": {
        "output": qint16
    },
    "camera_net.backbone.stages.2.block.3.pwconv2": {"weight": qint16},
    "camera_net.backbone.stages.3.block.3.act.lut": {"output": qint16},
    "camera_net.backbone.stages.3.block.2.add": {"output": qint16},
    "camera_net.backbone.stages.3.block.3.dwconv.0": {
        "weight": qint16,
        "output": qint16,
    },
    "task_decoders.static_objects_det.anchor_encoder.vel_fc.1": {
        "output": qint16
    },
    "camera_net.backbone.downsample_block.2.proj.0": {
        "output": qint16,
        "weight": qint16,
    },
    "camera_net.backbone.stages.3.block.4.add": {"output": qint16},
    "camera_net.backbone.stages.3.block.1.add": {"output": qint16},
    "camera_net.view_transformer.encoder.layers.0.ffn.ffn2": {
        "output": qint16,
        "weight": qint16,
    },
    "camera_net.backbone.stages.3.block.5.pwconv2": {"weight": qint16},
    "task_decoders.static_objects_det.layers.1.attn.softmax.mul": {
        "output": qint16
    },
    "task_decoders.static_objects_det.layers.3.feat_mul": {"output": qint16},
    "camera_net.backbone.stages.1.block.0.add": {"output": qint16},
    "camera_net.backbone.stages.2.block.0.pwconv2": {"weight": qint16},
    "task_decoders.dynamic_objects_det.fc_after": {"output": qint16},
    "camera_net.backbone.downsample_block.0.proj.0": {
        "weight": qint16,
        "output": qint16,
    },
    "camera_net.view_transformer.encoder.layers.0.ca.output_proj": {
        "output": qint16,
        "weight": qint16,
    },
    "camera_net.backbone.stages.1.block.1.add": {"output": qint16},
    "task_decoders.dynamic_objects_det.layers.15.attn.softmax.mul": {
        "output": qint16
    },
    "camera_net.view_transformer.encoder.layers.0.ca.add_res": {
        "output": qint16
    },
    (
        "camera_net.view_transformer.encoder.layers.0.ca."
        "deformable_attention.value_proj"
    ): {
        "output": qint16,
        "weight": qint16,
    },
    "camera_net.backbone.stages.2.block.2.pwconv2": {"weight": qint16},
    "task_decoders.static_objects_det.layers.29.attn.softmax.mul": {
        "output": qint16
    },
    "camera_net.backbone.stages.3.block.3.add": {"output": qint16},
    (
        "camera_net.view_transformer.encoder.layers.0.ca."
        "deformable_attention.mul_attention"
    ): {"output": qint16},
    "task_decoders.static_objects_det.layers.6.layers.8": {"output": qint16},
    "task_decoders.maptr.bev_decoders.0.decoder.layers.0.ffn.add": {
        "output": qint16
    },
    "camera_net.backbone.stages.0.block.0.add": {"output": qint16},
    "task_decoders.dynamic_objects_det.layers.6.layers.8": {"output": qint16},
    "camera_net.backbone.stages.2.block.1.pwconv2": {"weight": qint16},
    "camera_net.backbone.stages.3.block.1.dwconv.0": {
        "weight": qint16,
        "output": qint16,
    },
    "task_decoders.occ.occ_head.final_conv.0": {
        "output": qint16,
        "weight": qint16,
    },
    "task_decoders.discrete_map.decoder.layers.0.sa.softmax.mul": {
        "output": qint16
    },
    "task_decoders.static_objects_det.anchor_encoder.cat": {"output": qint16},
    "task_decoders.maptr.bev_decoders.0.decoder.layers.1.ffn.add": {
        "output": qint16
    },
    "camera_net.backbone.stages.0.block.0.dwconv.0": {
        "output": qint16,
        "weight": qint16,
    },
    "task_decoders.static_objects_det.anchor_encoder.pos_fc.4": {
        "output": qint16
    },
    "camera_net.backbone.downsample_block.1.proj.0": {
        "weight": qint16,
        "output": qint16,
    },
    "camera_net.backbone.stages.3.block.1.act.lut": {"output": qint16},
    "camera_net.backbone.stages.2.block.1.pwconv1.1.lut": {"output": qint16},
    "camera_net.backbone.stages.3.block.1.pwconv1": {
        "weight": qint16,
        "output": qint16,
    },
    "task_decoders.discrete_map.quant_object_query_embed": {"output": qint16},
    "task_decoders.static_objects_det.adapter.0.2": {"output": qint16},
    "task_decoders.dynamic_objects_det.anchor_encoder.size_fc.0": {
        "input": [qint16]
    },
    "task_decoders.occ.skip_add": {"output": qint16},
    "task_decoders.maptr.bev_decoders.0.reg_branches.0.0": {"output": qint16},
    "camera_net.backbone.stages.1.block.1.pwconv2": {"weight": qint16},
    "camera_net.backbone.stages.2.block.5.pwconv1.1.lut": {"output": qint16},
    "camera_net.backbone.stages.2.block.5.pwconv2": {"weight": qint16},
    "camera_net.backbone.stages.2.block.6.pwconv1.1.lut": {"output": qint16},
    "camera_net.backbone.stages.2.block.4.pwconv2": {"weight": qint16},
    "task_decoders.occ.adapter.0.0": {"weight": qint16},
    "camera_net.backbone.stages.0.block.0.pwconv1": {
        "output": qint16,
        "weight": qint16,
    },
    "camera_net.view_transformer.encoder.layers.0.ca_norm.bias_add": {
        "output": qint16
    },
    "camera_net.view_transformer.encoder.layers.0.ca.mul_pillarweight": {
        "output": qint16
    },
    "camera_net.backbone.stages.2.block.7.pwconv2": {"weight": qint16},
    "camera_net.backbone.stages.2.block.4.pwconv1.1.lut": {"output": qint16},
    "camera_net.backbone.stages.2.block.6.pwconv2": {"weight": qint16},
    (
        "camera_net.view_transformer.encoder.layers.0.ca."
        "deformable_attention.reduce_sum"
    ): {
        "output": qint16,
        "weight": qint16,
    },
    "task_decoders.dynamic_objects_det.layers.8.attn.softmax.mul": {
        "output": qint16
    },
    "camera_net.backbone.stages.3.block.4.pwconv1": {
        "output": qint16,
        "weight": qint16,
    },
    "task_decoders.static_objects_det.layers.5.bias_add": {"output": qint16},
    "task_decoders.dynamic_objects_det.anchor_encoder.pos_fc.4": {
        "output": qint16
    },
    "task_decoders.maptr.bev_decoders.0.decoder.layers.0.ffn.ffn2": {
        "output": qint16
    },
    "task_decoders.dynamic_objects_det.anchor_encoder.pos_fc.2.bias_add": {
        "output": qint16
    },
    "camera_net.backbone.stages.1.block.1.dwconv.0": {
        "weight": qint16,
        "output": qint16,
    },
    "camera_net.backbone.stages.1.block.0.pwconv2": {"weight": qint16},
    "task_decoders.static_objects_det.layers.3.feat_sum": {"output": qint16},
    "task_decoders.static_objects_det.layers.15.attn.matmul": {
        "output": qint16
    },
    "camera_net.backbone.stages.0.block.2.act.lut": {"output": qint16},
    "task_decoders.static_objects_det.anchor_encoder.yaw_fc.1": {
        "output": qint16
    },
    "task_decoders.static_objects_det.anchor_encoder.pos_fc.5.bias_add": {
        "output": qint16
    },
    "camera_net.backbone.stages.2.block.7.pwconv1.1.lut": {"output": qint16},
    "task_decoders.discrete_map.decoder.layers.4.sa.softmax.mul": {
        "output": qint16
    },
    "task_decoders.static_objects_det.anchor_encoder.size_fc.2.bias_add": {
        "output": qint16
    },
    "camera_net.backbone.stages.1.block.2.act.lut": {"output": qint16},
    "camera_net.view_transformer.encoder.layers.0.ffn_norm.bias_add": {
        "output": qint16
    },
    "task_decoders.dynamic_objects_det.layers.0.attn.softmax.mul": {
        "output": qint16
    },
    "task_decoders.static_objects_det.layers.13.layers.8": {"output": qint16},
    "camera_net.view_transformer.encoder.layers.0.ca.query_reduce_sum": {
        "output": qint16,
        "weight": qint16,
    },
    "task_decoders.static_objects_det.anchor_encoder.pos_fc.2.bias_add": {
        "output": qint16
    },
    "camera_net.backbone.stages.2.block.0.add": {"output": qint16},
    "camera_net.backbone.stages.3.block.5.pwconv1": {
        "output": qint16,
        "weight": qint16,
    },
    "camera_net.backbone.stages.1.block.2.add": {"output": qint16},
    "task_decoders.dynamic_objects_det.layers.29.attn.matmul": {
        "output": qint16
    },
    "task_decoders.static_objects_det.layers.1.add": {"output": qint16},
    "task_decoders.dynamic_objects_det.layers.22.attn.softmax.mul": {
        "output": qint16
    },
    "camera_net.backbone.stages.2.block.0.pwconv1.1.lut": {"output": qint16},
    "task_decoders.maptr.bev_decoders.0.decoder.layers.2.ffn.add": {
        "output": qint16
    },
    "task_decoders.static_objects_det.layers.1.query_cat": {"output": qint16},
    "task_decoders.dynamic_objects_det.layers.6.layers.9.bias_add": {
        "output": qint16
    },
    "camera_net.backbone.stages.3.block.2.dwconv.0": {
        "weight": qint16,
        "output": qint16,
    },
    "camera_net.backbone.stages.3.block.2.pwconv1": {
        "weight": qint16,
        "output": qint16,
    },
    "camera_net.backbone.stages.2.block.5.dwconv.0.0": {
        "weight": qint16,
        "output": qint16,
    },
    "task_decoders.maptr.adapter.0.0": {"weight": qint16},
    "task_decoders.static_objects_det.layers.3.residual_op": {
        "output": qint16
    },
    "camera_net.backbone.stages.0.block.0.act.lut": {"output": qint16},
    "task_decoders.occ.adapter.0.2": {"output": qint16},
    "camera_net.backbone.stages.3.block.4.dwconv.0": {
        "output": qint16,
        "weight": qint16,
    },
    "camera_net.backbone.stages.0.block.2.pwconv2": {"weight": qint16},
    "camera_net.backbone.stages.0.block.1.pwconv1": {
        "output": qint16,
        "weight": qint16,
    },
    "task_decoders.dynamic_objects_det.layers.8.attn.matmul": {
        "output": qint16
    },
    (
        "camera_net.view_transformer.encoder.layers.0.ca."
        "deformable_attention.stack_sampling"
    ): {"output": qint16},
    "task_decoders.static_objects_det.layers.6.add1": {"output": qint16},
    "task_decoders.maptr.skip_add": {"output": qint16},
    "task_decoders.static_objects_det.layers.8.attn.matmul": {
        "output": qint16
    },
    "task_decoders.static_objects_det.anchor_encoder.pos_fc.10": {
        "output": qint16
    },
    "task_decoders.static_objects_det.layers.4.short_add": {"output": qint16},
    "camera_net.backbone.stages.3.block.5.dwconv.0": {
        "output": qint16,
        "weight": qint16,
    },
    "camera_net.backbone.stages.0.block.3.pwconv2": {"weight": qint16},
    "camera_net.backbone.stages.2.block.2.add": {"output": qint16},
    "camera_net.backbone.stages.0.block.1.pwconv2": {"weight": qint16},
    "task_decoders.static_objects_det.layers.6.layers.9.bias_add": {
        "output": qint16
    },
    "task_decoders.static_objects_det.fc_after": {"output": qint16},
    "camera_net.view_transformer.encoder.quant_cams_embeds": {
        "output": qint16
    },
    "camera_net.backbone.stages.2.block.4.add": {"output": qint16},
    "task_decoders.dynamic_objects_det.anchor_encoder.cat": {"output": qint16},
    "camera_net.backbone.stages.2.block.5.add": {"output": qint16},
    "camera_net.backbone.stages.0.block.3.pwconv1": {
        "weight": qint16,
        "output": qint16,
    },
    "camera_net.backbone.stages.1.block.2.pwconv2": {"weight": qint16},
    "task_decoders.maptr.bev_decoders.0.decoder.layers.1.ffn.ffn2": {
        "output": qint16
    },
    "task_decoders.dynamic_objects_det.anchor_encoder.pos_fc.5.bias_add": {
        "output": qint16
    },
    "task_decoders.static_objects_det.layers.3.feat_cat": {"output": qint16},
    "camera_net.backbone.stages.3.block.3.pwconv1": {
        "weight": qint16,
        "output": qint16,
    },
    "task_decoders.discrete_map.decoder.layers.5.sa.softmax.mul": {
        "output": qint16
    },
    "task_decoders.dynamic_objects_det.anchor_encoder.size_fc.1": {
        "output": qint16
    },
    "task_decoders.static_objects_det.layers.13.add1": {"output": qint16},
    "task_decoders.dynamic_objects_det.anchor_encoder.pos_fc.11.bias_add": {
        "output": qint16
    },
    "camera_net.backbone.stages.3.block.0.pwconv1": {
        "weight": qint16,
        "output": qint16,
    },
    "camera_net.backbone.stages.2.block.4.dwconv.0.0": {
        "weight": qint16,
        "output": qint16,
    },
    "camera_net.backbone.stages.2.block.2.pwconv1.1.lut": {"output": qint16},
    "camera_net.backbone.stages.2.block.7.add": {"output": qint16},
    "camera_net.backbone.stages.0.block.2.dwconv.0": {
        "weight": qint16,
        "output": qint16,
    },
    "task_decoders.static_objects_det.anchor_encoder.yaw_fc.11.bias_add": {
        "output": qint16
    },
    "task_decoders.static_objects_det.mat_quant_stub": {"output": qint16},
    "camera_net.view_transformer.encoder.add_level_embeds.0": {
        "output": qint16
    },
    "camera_net.view_transformer.encoder.cat_fl": {"output": qint16},
    "camera_net.neck.conv_extract.0.0": {"output": qint16, "weight": qint16},
    "camera_net.view_transformer.encoder.quant_level_embeds": {
        "output": qint16
    },
    "task_decoders.static_objects_det.skip_add": {"output": qint16},
    "task_decoders.discrete_map.decoder.layers.0.ca.reduce_sum": {
        "output": qint16,
        "weight": qint16,
    },
    "camera_net.backbone.stages.2.block.1.add": {"output": qint16},
    "task_decoders.static_objects_det.layers.10.feat_mul": {"output": qint16},
    "camera_net.backbone.stages.2.block.3.pwconv1.1.lut": {"output": qint16},
    "camera_net.view_transformer.encoder.addcams_embeds.0": {"output": qint16},
    "task_decoders.dynamic_objects_det.layers.22.attn.matmul": {
        "output": qint16
    },
    "task_decoders.static_objects_det.anchor_encoder.pos_fc.11.bias_add": {
        "output": qint16
    },
    "task_decoders.dynamic_objects_det.layers.13.layers.8": {"output": qint16},
    "task_decoders.static_objects_det.layers.3.output_proj": {
        "output": qint16
    },
    "task_decoders.static_objects_det.layers.6.layers.4.bias_add": {
        "output": qint16
    },
    "camera_net.backbone.stages.0.block.2.pwconv1": {
        "weight": qint16,
        "output": qint16,
    },
    "task_decoders.static_objects_det.layers.6.layers.2": {"weight": qint16},
    "camera_net.view_transformer.encoder.layers.0.sa_norm.bias_add": {
        "output": qint16
    },
    "task_decoders.static_objects_det.layers.2.bias_add": {"output": qint16},
    "camera_net.backbone.stages.2.block.3.dwconv.0.0": {"weight": qint16},
    "camera_net.backbone.stages.1.block.1.act.lut": {"output": qint16},
    "camera_net.backbone.stages.0.block.1.add": {"output": qint16},
    "task_decoders.static_objects_det.layers.6.layers.3": {"output": qint16},
    "camera_net.backbone.stages.1.block.2.dwconv.0": {
        "output": qint16,
        "weight": qint16,
    },
    "camera_net.backbone.stages.2.block.2.dwconv.0.0": {"weight": qint16},
    "camera_net.backbone.stage_norm.3": {"output": qint16},
    "task_decoders.discrete_map.decoder.layers.2.sa.softmax.mul": {
        "output": qint16
    },
    "camera_net.backbone.stages.2.block.0.dwconv.0.0": {
        "weight": qint16,
        "output": qint16,
    },
    "task_decoders.static_objects_det.layers.20.layers.8": {"output": qint16},
    "camera_net.backbone.stages.1.block.0.act.lut": {"output": qint16},
    "task_decoders.static_objects_det.layers.15.add": {"output": qint16},
    "task_decoders.dynamic_objects_det.layers.1.query_cat": {"output": qint16},
    "task_decoders.dynamic_objects_det.layers.35.attn.softmax.mul": {
        "output": qint16
    },
    "task_decoders.static_objects_det.layers.4.pre_norm.bias_add": {
        "output": qint16
    },
    "task_decoders.static_objects_det.layers.4.layers.1": {"output": qint16},
    "task_decoders.static_objects_det.layers.29.attn.matmul": {
        "output": qint16
    },
    "task_decoders.maptr.bev_decoders.0.reg_branches.0.5": {"output": qint16},
    "camera_net.backbone.stages.2.block.0.pwconv1.0": {"weight": qint16},
    "task_decoders.dynamic_objects_det.anchor_encoder.pos_fc.7": {
        "output": qint16
    },
    "task_decoders.static_objects_det.anchor_encoder.vel_fc.11.bias_add": {
        "output": qint16
    },
    "task_decoders.occ.adapter.1.0": {"weight": qint16},
    "task_decoders.static_objects_det.layers.17.residual_op": {
        "output": qint16
    },
    "task_decoders.static_objects_det.layers.3.weight_softmax.mul": {
        "output": qint16
    },
    "task_decoders.maptr.bev_decoders.0.decoder.layers.1.ca_norm.bias_add": {
        "output": qint16
    },
    "task_decoders.static_objects_det.layers.0.add": {"output": qint16},
    "task_decoders.static_objects_det.anchor_encoder.yaw_fc.0": {
        "input": [qint16]
    },
    "task_decoders.maptr.bev_decoders.0.decoder.layers.2.ca_norm.bias_add": {
        "output": qint16
    },
    "task_decoders.maptr.bev_decoders.0.reg_branches.0.3": {"output": qint16},
    "camera_net.backbone.stages.1.block.0.pwconv1": {
        "weight": qint16,
        "output": qint16,
    },
    "task_decoders.static_objects_det.layers.27.layers.8": {"output": qint16},
    "task_decoders.static_objects_det.layers.10.feat_sum": {"output": qint16},
    "task_decoders.static_objects_det.layers.13.layers.9.bias_add": {
        "output": qint16
    },
    "task_decoders.discrete_map.adapter.0.0": {"weight": qint16},
    "camera_net.backbone.stages.3.block.0.dwconv.0": {
        "weight": qint16,
        "output": qint16,
    },
    "task_decoders.static_objects_det.anchor_encoder.vel_fc.2.bias_add": {
        "output": qint16
    },
    "camera_net.backbone.stages.2.block.7.dwconv.0.0": {"weight": qint16},
    "camera_net.backbone.stages.2.block.1.pwconv1.0": {
        "weight": qint16,
        "output": qint16,
    },
    "task_decoders.static_objects_det.anchor_encoder.yaw_fc.2.bias_add": {
        "output": qint16
    },
    "task_decoders.static_objects_det.layers.10.output_proj": {
        "output": qint16
    },
    "task_decoders.static_objects_det.anchor_encoder.size_fc.0": {
        "input": [qint16]
    },
    "camera_net.backbone.stages.0.block.3.act.lut": {"output": qint16},
    "task_decoders.static_objects_det.adapter.0.0": {"weight": qint16},
    "task_decoders.dynamic_objects_det.anchor_encoder.pos_fc.10": {
        "output": qint16
    },
    "task_decoders.maptr.bev_decoders.0.reg_branches.0.2": {"output": qint16},
    "task_decoders.static_objects_det.anchor_encoder.yaw_fc.8.bias_add": {
        "output": qint16
    },
    "task_decoders.static_objects_det.layers.13.layers.1": {"output": qint16},
    "task_decoders.discrete_map.decoder.layers.1.sa.softmax.mul": {
        "output": qint16
    },
    "camera_net.backbone.stages.2.block.2.pwconv1.0": {
        "weight": qint16,
        "output": qint16,
    },
    "task_decoders.static_objects_det.anchor_encoder.vel_fc.3": {
        "weight": qint16
    },
    "camera_net.backbone.stages.2.block.3.add": {"output": qint16},
    "task_decoders.static_objects_det.layers.24.feat_sum": {"output": qint16},
    "camera_net.backbone.stages.1.block.1.pwconv1": {
        "weight": qint16,
        "output": qint16,
    },
    "task_decoders.static_objects_det.layers.10.residual_op": {
        "output": qint16
    },
    "task_decoders.static_objects_det.layers.17.feat_sum": {"output": qint16},
    "task_decoders.dynamic_objects_det.layers.3.feat_mul": {"output": qint16},
    "task_decoders.maptr.bev_decoders.0.decoder.layers.4.ffn.add": {
        "output": qint16
    },
    "task_decoders.dynamic_objects_det.anchor_encoder.pos_fc.8.bias_add": {
        "output": qint16
    },
    "task_decoders.static_objects_det.anchor_encoder.pos_fc.7": {
        "output": qint16
    },
    "task_decoders.static_objects_det.layers.27.add1": {"output": qint16},
    "task_decoders.discrete_map.decoder.layers.0.ca_norm.bias_add": {
        "output": qint16
    },
    "task_decoders.static_objects_det.layers.23.bias_add": {"output": qint16},
    "task_decoders.maptr.bev_decoders.0.decoder.layers.3.ffn.add": {
        "output": qint16
    },
    "camera_net.backbone.stages.2.block.4.pwconv1.0": {
        "weight": qint16,
        "output": qint16,
    },
    "task_decoders.static_objects_det.layers.6.layers.1": {"output": qint16},
    "task_decoders.static_objects_det.layers.22.attn.matmul": {
        "output": qint16
    },
    "task_decoders.static_objects_det.layers.38.residual_op": {
        "output": qint16
    },
    "task_decoders.dynamic_objects_det.layers.1.add": {"output": qint16},
    "task_decoders.maptr.bev_decoders.0.decoder.layers.0.ca.softmax.mul": {
        "output": qint16
    },
    "task_decoders.static_objects_det.layers.1.attn.attn_matmul": {
        "output": qint16
    },
    "task_decoders.static_objects_det.anchor_encoder.size_fc.10": {
        "output": qint16
    },
    "task_decoders.static_objects_det.layers.26.bias_add": {"output": qint16},
    "task_decoders.dynamic_objects_det.layers.15.attn.matmul": {
        "output": qint16
    },
    "camera_net.view_transformer.encoder.layers.0.ffn.ffn1": {
        "weight": qint16
    },
    "camera_net.backbone.stages.0.block.1.act.lut": {"output": qint16},
    "camera_net.backbone.stages.2.block.6.add": {"output": qint16},
    "task_decoders.dynamic_objects_det.layers.6.layers.4.bias_add": {
        "output": qint16
    },
    "task_decoders.maptr.bev_decoders.0.decoder.layers.2.sa_norm.bias_add": {
        "output": qint16
    },
    "task_decoders.static_objects_det.layers.17.weights_fc": {
        "output": qint16
    },
    "camera_net.backbone.stages.1.block.2.pwconv1": {
        "weight": qint16,
        "output": qint16,
    },
    "task_decoders.discrete_map.decoder.layers.0.ca.add_res": {
        "output": qint16
    },
    "task_decoders.dynamic_objects_det.layers.10.residual_op": {
        "output": qint16
    },
    "task_decoders.maptr.bev_decoders.0.decoder.layers.1.sa_norm.bias_add": {
        "output": qint16
    },
    "task_decoders.static_objects_det.layers.8.add": {"output": qint16},
    "task_decoders.static_objects_det.layers.7.query_cat": {"output": qint16},
    "camera_net.backbone.stages.2.block.6.dwconv.0.0": {
        "weight": qint16,
        "output": qint16,
    },
    "camera_net.backbone.stages.1.block.0.dwconv.0": {"weight": qint16},
    "task_decoders.static_objects_det.layers.10.feat_cat": {"output": qint16},
    "camera_net.view_transformer.quant_bev_query": {"output": qint16},
    "camera_net.backbone.stages.0.block.2.add": {"output": qint16},
    "task_decoders.dynamic_objects_det.anchor_encoder.vel_fc.10": {
        "output": qint16
    },
    "camera_net.backbone.stages.2.block.5.pwconv1.0": {
        "weight": qint16,
        "output": qint16,
    },
    "task_decoders.discrete_map.decoder.layers.3.sa.softmax.mul": {
        "output": qint16
    },
    "task_decoders.static_objects_det.anchor_encoder.size_fc.11.bias_add": {
        "output": qint16
    },
    "task_decoders.static_objects_det.anchor_encoder.yaw_fc.4": {
        "output": qint16
    },
    "task_decoders.maptr.bev_decoders.0.reg_branches.0.6": {"weight": qint16},
    "task_decoders.maptr.bev_decoders.0.decoder.layers.2.ffn.ffn2": {
        "output": qint16
    },
    "camera_net.backbone.stages.2.block.7.pwconv1.0": {
        "weight": qint16,
        "output": qint16,
    },
    "task_decoders.maptr.bev_decoders.0.decoder.layers.0.ca.mul_attention": {
        "output": qint16
    },
    "camera_net.backbone.stages.2.block.1.dwconv.0.0": {
        "output": qint16,
        "weight": qint16,
    },
    "task_decoders.maptr.bev_decoders.0.decoder.layers.1.ca.add_res": {
        "output": qint16
    },
    "camera_net.backbone.stages.2.block.3.pwconv1.0": {
        "weight": qint16,
        "output": qint16,
    },
    "task_decoders.discrete_map.reference_points": {"weight": qint16},
    "task_decoders.maptr.bev_decoders.0.decoder.layers.0.ca.add_res": {
        "output": qint16
    },
    "task_decoders.discrete_map.decoder.layers.0.ca.output_proj": {
        "output": qint16
    },
    "task_decoders.static_objects_det.anchor_encoder.size_fc.1": {
        "output": qint16
    },
    "task_decoders.maptr.bev_decoders.0.decoder.layers.0.ca_norm.bias_add": {
        "output": qint16
    },
    "task_decoders.static_objects_det.layers.38.feat_mul": {"output": qint16},
    "task_decoders.static_objects_det.layers.24.residual_op": {
        "output": qint16
    },
    "task_decoders.maptr.bev_decoders.0.decoder.layers.2.ca.add_res": {
        "output": qint16
    },
    "task_decoders.discrete_map.decoder.layers.0.ca.mul_attention": {
        "output": qint16
    },
    "task_decoders.maptr.adapter.0.2": {"output": qint16},
    "task_decoders.static_objects_det.layers.11.pre_norm.bias_add": {
        "output": qint16
    },
    "task_decoders.dynamic_objects_det.anchor_encoder.vel_fc.2.bias_add": {
        "output": qint16
    },
    "task_decoders.static_objects_det.layers.17.feat_cat": {"output": qint16},
    "task_decoders.static_objects_det.layers.6.layers.10": {"weight": qint16},
    "task_decoders.static_objects_det.layers.6.layers.6": {"output": qint16},
    "task_decoders.static_objects_det.layers.0.attn.softmax.mul": {
        "output": qint16
    },
    "camera_net.backbone.stages.0.block.3.add": {"output": qint16},
    "task_decoders.static_objects_det.layers.41.add1": {"output": qint16},
    "task_decoders.static_objects_det.layers.20.layers.0": {"weight": qint16},
    "task_decoders.static_objects_det.layers.17.feat_mul": {"output": qint16},
    "task_decoders.static_objects_det.layers.13.layers.6": {"output": qint16},
    "task_decoders.dynamic_objects_det.layers.4.short_add": {"output": qint16},
    "task_decoders.discrete_map.decoder.layers.1.ca.reduce_sum": {
        "output": qint16
    },
    "task_decoders.static_objects_det.layers.31.feat_mul": {"output": qint16},
    "task_decoders.static_objects_det.anchor_encoder.pos_fc.8.bias_add": {
        "output": qint16
    },
    "task_decoders.maptr.bev_decoders.0.decoder.layers.3.ca_norm.bias_add": {
        "output": qint16
    },
    "task_decoders.static_objects_det.layers.38.feat_sum": {"output": qint16},
    "task_decoders.dynamic_objects_det.layers.6.layers.1": {"output": qint16},
    "camera_net.backbone.stages.2.block.6.pwconv1.0": {"output": qint16},
    "task_decoders.discrete_map.decoder.layers.5.ffn.add": {"output": qint16},
    "task_decoders.static_objects_det.layers.19.bias_add": {"output": qint16},
    "task_decoders.dynamic_objects_det.layers.6.layers.10": {"weight": qint16},
    "task_decoders.static_objects_det.layers.14.query_cat": {"output": qint16},
    "task_decoders.dynamic_objects_det.layers.6.layers.3": {"output": qint16},
    "task_decoders.static_objects_det.anchor_encoder.vel_fc.10": {
        "output": qint16
    },
    "task_decoders.discrete_map.decoder.layers.0.add_self_attn": {
        "output": qint16
    },
    "task_decoders.discrete_map.decoder.layers.5.sa_norm.bias_add": {
        "output": qint16
    },
    "task_decoders.dynamic_objects_det.layers.0.add": {"output": qint16},
    "task_decoders.discrete_map.decoder.layers.3.ca.add_res": {
        "output": qint16
    },
    "task_decoders.static_objects_det.layers.36.attn.matmul": {
        "output": qint16
    },
    "task_decoders.discrete_map.decoder.layers.5.ca.add_res": {
        "output": qint16
    },
    "camera_net.view_transformer.encoder.layers.0.sa.add_res": {
        "output": qint16
    },
    "task_decoders.static_objects_det.layers.24.feat_mul": {"output": qint16},
    "task_decoders.dynamic_objects_det.adapter.0.2": {"output": qint16},
    "task_decoders.maptr.bev_decoders.0.decoder.layers.4.ffn.ffn2": {
        "output": qint16
    },
    "camera_net.view_transformer.encoder.layers.0.sa.output_proj": {
        "output": qint16
    },
    "task_decoders.static_objects_det.layers.3.weights_fc": {"output": qint16},
    "task_decoders.discrete_map.decoder.layers.4.ca_norm.bias_add": {
        "output": qint16
    },
    "task_decoders.static_objects_det.anchor_encoder.yaw_fc.7": {
        "output": qint16
    },
    "task_decoders.dynamic_objects_det.layers.6.add1": {"output": qint16},
    "task_decoders.dynamic_objects_det.layers.7.attn.softmax.mul": {
        "output": qint16
    },
    "task_decoders.static_objects_det.anchor_encoder.yaw_fc.10": {
        "output": qint16
    },
    "task_decoders.static_objects_det.layers.8.query_cat": {"output": qint16},
    "task_decoders.discrete_map.decoder.layers.3.sa_norm.bias_add": {
        "output": qint16
    },
    "task_decoders.dynamic_objects_det.layers.6.layers.6": {"output": qint16},
    "task_decoders.dynamic_objects_det.anchor_encoder.pos_fc.3": {
        "weight": qint16
    },
    "task_decoders.static_objects_det.layers.20.layers.3": {"output": qint16},
    "task_decoders.static_objects_det.anchor_encoder.pos_fc.3": {
        "weight": qint16
    },
    "task_decoders.discrete_map.decoder.layers.5.ca_norm.bias_add": {
        "output": qint16
    },
    "task_decoders.static_objects_det.layers.10.weight_softmax.mul": {
        "output": qint16
    },
    "task_decoders.discrete_map.cls_branches.5.0": {"output": qint16},
    "task_decoders.maptr.bev_decoders.0.decoder.layers.0.ffn.ffn1_act": {
        "output": qint16
    },
    "task_decoders.dynamic_objects_det.layers.3.residual_op": {
        "output": qint16
    },
    "task_decoders.static_objects_det.layers.21.attn.softmax.mul": {
        "output": qint16
    },
    "task_decoders.maptr.bev_decoders.0.decoder.layers.0.ffn_norm.bias_add": {
        "output": qint16
    },
    "task_decoders.discrete_map.decoder.layers.4.ffn.add": {"output": qint16},
    "task_decoders.maptr.bev_decoders.0.decoder.layers.0.ca.reduce_sum": {
        "output": qint16
    },
    "task_decoders.static_objects_det.layers.7.add": {"output": qint16},
    "task_decoders.static_objects_det.layers.15.query_cat": {"output": qint16},
    "task_decoders.static_objects_det.layers.41.quality_layers.5.bias_add": {
        "output": qint16
    },
}

q_template = [
    ModuleNameTemplate({"": qint8}),
    MatmulDtypeTemplate(
        input_dtypes=[qint8, qint8],
    ),
    ConvDtypeTemplate(
        input_dtype=qint8,
        weight_dtype=qint8,
    ),
    ModuleNameTemplate(
        {
            m: {"dtype": qint8, "threshold": 1.0}
            for m in ["camera_net.backbone.quant"]
        },
    ),
    ModuleNameTemplate(
        {
            m: {"input": [qint16, qint16], "output": qint16}
            for m in int16_add_ops
        },
    ),
    ModuleNameTemplate(
        {
            "task_decoders.dynamic_objects_det.instance_bank.anchor_quant_stub": {  # noqa: E501
                "dtype": qint16,
                "threshold": 60,
            }
        },
        freeze=True,
    ),
    ModuleNameTemplate(
        {
            "task_decoders.static_objects_det.instance_bank.anchor_quant_stub": {  # noqa: E501
                "dtype": qint16,
                "threshold": 60,
            }
        },
        freeze=True,
    ),
    ModuleNameTemplate(
        {m: None for m in float32_ops},
        freeze=True,
    ),
    ModuleNameTemplate(sensitive_ops),
]

cali_qconfig_setter = QconfigSetter(
    reference_qconfig=get_qconfig(observer=(observer_v2.HistogramObserver)),
    templates=q_template,
    enable_optimize=True,
    save_dir=os.path.join(ckpt_dir, "./qconfig_setting"),
    custom_qconfig_mapping=None,
)


float2calibration = dict(
    type="Float2Calibration",
    convert_mode=convert_mode,
    example_inputs=example_inputs,
    qconfig_setter=cali_qconfig_setter,
)


calibration_train_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "float_stage3-checkpoint-best.pth.tar"
            ),
            allow_miss=True,
            ignore_extra=True,
            verbose=True,
        ),
        dict(type="RepModel2Deploy"),
        float2calibration,
    ],
)

calibration_predict_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(type="RepModel2Deploy"),
        float2calibration,
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

# HBIR export: fake-quant graph (Float2QAT val) + calibration scales.
# No QAT stage — same cali_qconfig_setter as PTQ.
from model import deploy_inputs  # noqa: E402

deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(type="RepModel2Deploy"),
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            example_inputs=deploy_inputs,
            state="val",
            qconfig_setter=cali_qconfig_setter,
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
