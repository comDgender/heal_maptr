"""Model definitions for SparseBEV HENet-tiny NuScenes detection."""

import copy
import os

import torch
from common import (
    CLASSES,
    anchor_file,
    ckpt_dir,
    dense_depth_loss_weight,
    drop_out,
    embed_dims,
    num_anchors,
    num_classes,
    num_decoder,
    num_depth_layers,
    num_groups,
    num_levels,
    num_single_frame_decoder,
    temp_anchors,
)

henet_backbone = dict(
    type="HENet",
    in_channels=3,
    block_nums=[4, 3, 8, 6],
    embed_dims=[64, 128, 192, 384],
    attention_block_num=[0, 0, 0, 0],
    mlp_ratios=[2, 2, 2, 3],
    mlp_ratio_attn=2,
    act_layer=["nn.GELU", "nn.GELU", "nn.GELU", "nn.GELU"],
    use_layer_scale=[True, True, True, True],
    layer_scale_init_value=1e-5,
    num_classes=1000,
    include_top=False,
    extra_act=[False, False, False, False],
    final_expand_channel=0,
    feature_mix_channel=1024,
    block_cls=["GroupDWCB", "GroupDWCB", "AltDWCB", "DWCB"],
    down_cls=["S2DDown", "S2DDown", "S2DDown", "None"],
    patch_embed="origin",
)

sparse_head = dict(
    type="SparseBEVHead",
    enable_dn=True,
    level_index=[2],
    cls_threshold_to_reg=0.05,
    instance_bank=dict(
        type="MemoryBank",
        num_anchor=num_anchors,
        embed_dims=embed_dims,
        num_memory_instances=num_anchors,
        anchor=anchor_file,
        num_temp_instances=temp_anchors,
        confidence_decay=0.6,
    ),
    anchor_encoder=dict(
        type="SparseBEVEncoder",
        pos_embed_dims=128,
        size_embed_dims=32,
        yaw_embed_dims=32,
        vel_embed_dims=64,
        vel_dims=3,
    ),
    num_single_frame_decoder=num_single_frame_decoder,
    operation_order=[
        "deformable",
        "ffn",
        "norm",
        "refine",
    ]
    * num_single_frame_decoder
    + [
        "temp_interaction",
        "interaction",
        "norm",
        "deformable",
        "ffn",
        "norm",
        "refine",
    ]
    * (num_decoder - num_single_frame_decoder),
    ffn=dict(
        type="AsymmetricFFN",
        in_channels=embed_dims * 2,
        pre_norm=True,
        embed_dims=embed_dims,
        feedforward_channels=embed_dims * 4,
        num_fcs=2,
        ffn_drop=drop_out,
    ),
    deformable_model=dict(
        type="DeformableFeatureAggregation",
        embed_dims=embed_dims,
        num_groups=num_groups,
        num_levels=num_levels,
        num_cams=6,
        attn_drop=0.15,
        use_camera_embed=True,
        residual_mode="cat",
        kps_generator=dict(
            type="SparseBEVKeyPointsGenerator",
            num_pts=8,
        ),
    ),
    refine_layer=dict(
        type="SparseBEVRefinementModule",
        embed_dims=embed_dims,
        num_cls=num_classes,
        refine_yaw=True,
    ),
    target=dict(
        type="SparseBEVTarget",
        num_dn_groups=5,
        num_temp_dn_groups=3,
        dn_noise_scale=[2.0] * 3 + [0.5] * 7,
        max_dn_gt=32,
        add_neg_dn=True,
        cls_weight=2.0,
        box_weight=0.25,
        reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,
        cls_wise_reg_weights={
            CLASSES.index("traffic_cone"): [
                2.0,
                2.0,
                2.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                1.0,
                1.0,
            ],
        },
    ),
    cls_allow_reverse=[CLASSES.index("barrier")],
    loss_cls=dict(
        type="FocalLoss",
        loss_name="cls",
        num_classes=num_classes + 1,
        gamma=2.0,
        alpha=0.25,
        loss_weight=2.0,
    ),
    loss_reg=dict(type="L1Loss", loss_weight=0.25),
    loss_cns=dict(type="CrossEntropyLoss", use_sigmoid=True),
    loss_yns=dict(type="GaussianFocalLoss"),
    decoder=dict(type="SparseBEVDecoder"),
    reg_weights=[2.0] * 3 + [1.0] * 7,
)

model = dict(
    type="SparseBEV",
    compiler_model=False,
    backbone=henet_backbone,
    neck=dict(
        type="MMFPN",
        in_strides=[2, 4, 8, 16, 32],
        in_channels=[64, 64, 128, 192, 384],
        fix_out_channel=embed_dims,
        out_strides=[4, 8, 16, 32],
    ),
    depth_branch=dict(
        type="DenseDepthNet",
        embed_dims=embed_dims,
        num_depth_layers=num_depth_layers,
        loss_weight=dense_depth_loss_weight,
        points_key="lidar.points",
    ),
    head=sparse_head,
)

deploy_model = copy.deepcopy(model)
deploy_model["compiler_model"] = True
deploy_model.pop("depth_branch", None)
deploy_model["head"]["deformable_model"]["attn_drop"] = 0.0
deploy_model["head"].pop("target", None)
deploy_model["head"].pop("loss_cls", None)
deploy_model["head"].pop("loss_reg", None)
deploy_model["head"].pop("loss_cns", None)
deploy_model["head"].pop("loss_yns", None)
deploy_model["head"].pop("decoder", None)

test_model = copy.deepcopy(model)
# DN (denoising) is training-only; the calibration val-state quantized graph
# traced by `calibration_val_model_convert_pipeline` (Float2Calibration) must
# use the same topology as the val model (enable_dn=False), otherwise the
# traced eval graph / checkpoint keys mismatch the training model.
test_model["head"]["enable_dn"] = False


def get_deploy_input():
    return {
        "img": torch.randn((6, 3, 256, 704)),
        "projection_mat": torch.randn((6, 4, 4)),
        "cached_anchor": torch.randn((1, num_anchors, 11)),
        "cached_feature": torch.randn((1, num_anchors, embed_dims)),
    }


def get_eval_trace_input():
    return {
        "img": torch.randn((6, 3, 256, 704)),
        "projection_mat": torch.randn((6, 4, 4)),
        "cached_anchor": torch.randn((1, num_anchors, 11)),
        "cached_feature": torch.randn((1, num_anchors, embed_dims)),
        "cached_confidence": torch.randn((1, num_anchors)),
        "mask": torch.ones((1)).bool(),
        "timestamp": torch.randn((1)),
        "lidar2global": torch.randn((1, 4, 4)),
        "lidar2img": torch.randn((6, 4, 4)),
    }


def get_train_trace_input():
    return {
        "img": torch.randn((6, 3, 256, 704)),
        "timestamp": torch.randn((1)),
        "lidar2global": torch.randn((1, 4, 4)),
        "lidar2img": torch.randn((6, 4, 4)),
        "lidar_bboxes_labels": torch.randn((1, 20, 10)),
        "instance_ids": torch.randn((1, 20)),
        "camera_intrinsic": torch.randn((6, 3, 3)),
        "points": torch.rand((1, 20000, 3)),
    }


deploy_inputs = get_deploy_input()
eval_trace_inputs = get_eval_trace_input()
train_trace_inputs = get_train_trace_input()

first_frame_input = {
    "cached_anchor": torch.zeros((1, num_anchors, 11)),
    "cached_feature": torch.zeros((1, num_anchors, embed_dims)),
    "cached_confidence": torch.zeros((1, num_anchors)),
    "mask": torch.zeros((1)).bool(),
}

hbir_infer_model = dict(
    type="SparseBEVIrInfer",
    first_frame_input=first_frame_input,
    projection_mat_key="lidar2img",
    global_mat_key="lidar2global",
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized_infer.bc"),
    ),
    decoder=dict(type="SparseBEVDecoder"),
    use_memory_bank=True,
    confidence_decay=0.6,
    num_temp_instances=temp_anchors,
    num_memory_instances=num_anchors,
)

# On-board BPU runtime after compile (``compile/model.hbm``).
# ``host`` is the BPU server address; override with your board's IP.
hbm_infer_model = copy.deepcopy(hbir_infer_model)
hbm_infer_model["use_nv12"] = True
hbm_infer_model["ir_model"] = dict(
    type="HbmModule",
    model_path=os.path.join(ckpt_dir, "compile", "model.hbm"),
    host="10.103.43.99",
    return_tensor=True,
)

output_names = ["classification", "prediction", "quality", "feature"]
