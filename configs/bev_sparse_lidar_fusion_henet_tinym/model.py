"""Model definitions for SparseBEV lidar fusion HENet-tiny NuScenes."""

import copy
import os

import numpy as np
import torch
from common import (
    CLASSES,
    anchor_file,
    ckpt_dir,
    dense_depth_loss_weight,
    drop_out,
    embed_dims,
    lidar_level_idx,
    max_num_points,
    max_voxels,
    num_anchors,
    num_classes,
    num_decoder,
    num_depth_layers,
    num_groups,
    num_levels,
    num_single_frame_decoder,
    point_cloud_range,
    temp_anchors,
    voxel_size,
)


def get_feature_map_size(point_cloud_range, voxel_size):
    point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
    voxel_size = np.array(voxel_size, dtype=np.float32)
    grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
    return np.round(grid_size).astype(np.int64)


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

lidar_network = dict(
    type="CenterPointDetector",
    feature_map_shape=get_feature_map_size(point_cloud_range, voxel_size),
    pre_process=dict(
        type="CenterPointPreProcess",
        pc_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels_num=max_voxels,
        max_points_in_voxel=max_num_points,
        norm_range=[-51.2, -51.2, -5.0, 0.0, 51.2, 51.2, 3.0, 255.0],
        norm_dims=[0, 1, 2, 3],
    ),
    reader=dict(
        type="PillarFeatureNet",
        num_input_features=5,
        num_filters=(64,),
        with_distance=False,
        pool_size=(max_num_points, 1),
        voxel_size=voxel_size,
        pc_range=point_cloud_range,
        bn_kwargs={},
        quantize=True,
        use_4dim=True,
        use_conv=True,
        hw_reverse=True,
    ),
    scatter=dict(
        type="PointPillarScatter",
        num_input_features=64,
        use_horizon_pillar_scatter=True,
        quantize=True,
    ),
    backbone=dict(
        type="HENet",
        in_channels=64,
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
    ),
    neck=dict(
        type="MMFPN",
        in_strides=[2, 4, 8, 16, 32],
        in_channels=[64, 64, 128, 192, 384],
        fix_out_channel=embed_dims,
        out_strides=[4, 8, 16, 32],
    ),
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
        "lidar_deformable",
        "ffn",
        "norm",
        "refine",
    ]
    * num_single_frame_decoder
    + [
        "temp_interaction",
        "interaction",
        "norm",
        "lidar_deformable",
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
        type="DeformableFeatureAggregationLiF",
        fuse_module=dict(
            type="InstanceFuseModule",
            input_channels=[256, 256],
            fuse_channel=256,
        ),
        embed_dims=embed_dims,
        num_groups=num_groups,
        num_levels=num_levels,
        num_cams=6,
        attn_drop=0.15,
        use_camera_embed=True,
        residual_mode="cat",
        point_cloud_range=point_cloud_range,
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
    type="SparseBevFusion",
    compiler_model=False,
    lidar_net=lidar_network,
    lidar_level_idx=lidar_level_idx,
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
deploy_model["head"]["enable_dn"] = False
deploy_model["head"]["deformable_model"]["attn_drop"] = 0.0
deploy_model["head"].pop("target", None)
deploy_model["head"].pop("loss_cls", None)
deploy_model["head"].pop("loss_reg", None)
deploy_model["head"].pop("loss_cns", None)
deploy_model["head"].pop("loss_yns", None)
deploy_model["head"].pop("decoder", None)
deploy_model["head"]["operation_order"] = [
    "deformable",
    "ffn",
    "norm",
    "refine",
] * num_single_frame_decoder + [
    "temp_interaction",
    "interaction",
    "norm",
    "lidar_deformable",
    "ffn",
    "norm",
    "refine",
] * (
    num_decoder - num_single_frame_decoder
)
deploy_model["lidar_net"] = copy.deepcopy(lidar_network)
deploy_model["lidar_net"].pop("pre_process")

test_model = copy.deepcopy(model)


def get_deploy_input():
    return {
        "img": torch.randn((6, 3, 256, 704)),
        "projection_mat": torch.randn((6, 4, 4)),
        "cached_anchor": torch.randn((1, num_anchors, 11)),
        "cached_feature": torch.randn((1, num_anchors, embed_dims)),
        "features": torch.randn((1, 5, 20, 40000), dtype=torch.float32),
        "coors": torch.zeros([40000, 4]).int(),
    }


def get_eval_trace_input():
    return {
        "img": torch.randn((6, 3, 256, 704)),
        "lidar2global": torch.randn((1, 4, 4)),
        "lidar2img": torch.randn((6, 4, 4)),
        "points": [torch.randn((100, 5))],
    }


deploy_inputs = get_deploy_input()
eval_trace_inputs = get_eval_trace_input()

first_frame_input = {
    "cached_anchor": torch.zeros((1, num_anchors, 11)),
    "cached_feature": torch.zeros((1, num_anchors, embed_dims)),
    "cached_confidence": torch.zeros((1, num_anchors)),
    "mask": torch.zeros((1)).bool(),
}

hbir_infer_model = dict(
    type="SparseBEVFusionIrInfer",
    first_frame_input=first_frame_input,
    projection_mat_key="lidar2img",
    global_mat_key="lidar2global",
    lidar_preprocess=model["lidar_net"]["pre_process"],
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
