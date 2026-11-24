# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import copy
import os
from collections import OrderedDict

import torch
from common import (
    _dim_,
    _num_levels_,
    _pos_dim_,
    anchor_file,
    aux_seg_cfg,
    bev_h_,
    bev_size,
    bev_w_,
    camera_input,
    ckpt_dir,
    discrete_map_bbox_size,
    discrete_map_code_weights,
    discrete_map_num_classes,
    discrete_map_num_query,
    dynamic_objects_num_classes,
    fixed_ptsnum_per_gt_line,
    fixed_ptsnum_per_pred_line,
    get_feature_map_size,
    head_embed_dims,
    hidden_dim,
    input_size,
    lidar_input,
    map_bev_h_,
    map_bev_w_,
    map_point_cloud_range,
    max_camoverlap_num,
    max_num_points,
    max_voxels,
    num_classes_occ,
    num_decoder,
    num_dn_groups,
    num_map_classes,
    num_single_frame_decoder,
    num_temp_dn_groups,
    num_vec_one2many,
    num_vec_one2one,
    point_cloud_range,
    sparse_drop_out,
    sparse_embed_dims,
    sparse_num_groups,
    sparse_num_levels,
    static_objects_class_names,
    static_objects_num_classes,
    use_adapter,
    use_discrete_map_head,
    use_dynamic_det_head,
    use_lidar2img,
    use_maptr_head,
    use_occ_head,
    use_static_objects_det_head,
    voxel_size,
    with_track,
)


def dynamic_objects_process_func(meta, is_compile):
    if not is_compile:
        tmp_meta = copy.deepcopy(meta)
        img = tmp_meta["img"]
        timestamp = tmp_meta["timestamp"]
        if "lidar2ego" in tmp_meta:
            lidar2ego = tmp_meta["lidar2ego"]
        else:
            lidar2ego = None
        lidar2global = tmp_meta["lidar2global"]
        tmp_meta = tmp_meta["seq_meta"][0]
        tmp_meta.update(
            {
                "img": img,
                "timestamp": timestamp,
                "lidar2global": lidar2global,
            }
        )
        if lidar2ego is not None:
            tmp_meta["lidar2ego"] = lidar2ego
        if "gt_moving_objects_info" in tmp_meta:
            gt_moving_objects_info = tmp_meta["gt_moving_objects_info"]
            tmp_meta.update(gt_moving_objects_info)
        return tmp_meta
    else:
        return meta


def static_objects_process_func(meta, is_compile):
    if not is_compile:
        tmp_meta = copy.deepcopy(meta)
        img = tmp_meta["img"]
        timestamp = tmp_meta["timestamp"]
        if "lidar2ego" in tmp_meta:
            lidar2ego = tmp_meta["lidar2ego"]
        else:
            lidar2ego = None
        lidar2global = tmp_meta["lidar2global"]
        tmp_meta = tmp_meta["seq_meta"][0]
        tmp_meta.update(
            {
                "img": img,
                "timestamp": timestamp,
                "lidar2global": lidar2global,
            }
        )
        if lidar2ego is not None:
            tmp_meta["lidar2ego"] = lidar2ego
        if "gt_static_objects_info" in tmp_meta:
            gt_static_objects_info = tmp_meta["gt_static_objects_info"]
            tmp_meta.update(gt_static_objects_info)
        return tmp_meta
    else:
        return meta


def discrete_map_process_func(meta):
    tmp_meta = copy.deepcopy(meta)
    if "gt_discrete_map_info" in tmp_meta["seq_meta"][0]:
        discrete_map_info = tmp_meta["seq_meta"][0]["gt_discrete_map_info"]
        tmp_meta["seq_meta"][0] = discrete_map_info
    return tmp_meta


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
        bn_kwargs=None,
        quantize=True,
        use_4dim=True,
        use_conv=True,
        hw_reverse=True,
    ),
    backbone=dict(
        type="PointPillarScatter",
        num_input_features=64,
        use_horizon_pillar_scatter=True,
        quantize=True,
    ),
    neck=dict(
        type="SECONDNeck",
        in_feature_channel=64,
        down_layer_nums=[3, 5, 5],
        down_layer_strides=[2, 2, 2],
        down_layer_channels=[64, 128, 256],
        up_layer_strides=[0.5, 1, 2],
        up_layer_channels=[128, 128, 128],
        bn_kwargs=None,
        quantize=True,
        use_relu6=False,
    ),
)

camera_network = dict(
    type="BevFormer",
    out_indices=(-1,),
    backbone=dict(
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
    ),
    neck=dict(
        type="FPN",
        in_strides=[32],
        in_channels=[384],
        out_strides=[32],
        out_channels=[_dim_],
        bn_kwargs=dict(eps=1e-5, momentum=0.1),
    ),
    view_transformer=dict(
        type="SingleBevFormerViewTransformer",
        bev_h=bev_h_,
        bev_w=bev_w_,
        pc_range=point_cloud_range,
        num_points_in_pillar=2,
        embed_dims=_dim_,
        queue_length=1,
        in_indices=(-1,),
        single_bev=True,
        use_lidar2img=use_lidar2img,
        max_camoverlap_num=max_camoverlap_num,
        virtual_bev_h=64,
        virtual_bev_w=80,
        positional_encoding=dict(
            type="LearnedPositionalEncoding",
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        encoder=dict(
            type="SingleBEVFormerEncoder",
            num_layers=1,
            return_intermediate=False,
            bev_h=bev_h_,
            bev_w=bev_w_,
            embed_dims=_dim_,
            encoder_layer=dict(
                type="SingleFrameBEVFormerEncoderLayer",
                embed_dims=_dim_,
                selfattention=dict(
                    type="HorizonMultiScaleDeformableAttention",
                    embed_dims=_dim_,
                    num_points=2,
                    num_levels=1,
                    grid_align_num=4,
                    batch_first=True,
                    feats_size=[[bev_w_, bev_h_]],
                ),
                crossattention=dict(
                    type="HorizonSpatialCrossAttention",
                    max_camoverlap_num=max_camoverlap_num,
                    bev_h=bev_h_,
                    bev_w=bev_w_,
                    deformable_attention=dict(
                        type="HorizonMultiScaleDeformableAttention3D",
                        embed_dims=_dim_,
                        num_points=2,
                        num_levels=_num_levels_,
                        grid_align_num=64,
                        feats_size=[[30, 16]],
                    ),
                    embed_dims=_dim_,
                ),
                dropout=0.1,
            ),
        ),
    ),
)

dynamic_bev_head = dict(
    type="DenseBEVSparseHead",
    filter_empty_gt=True,
    meta_process_func=dynamic_objects_process_func,
    with_track=with_track,
    use_adapter=use_adapter,
    in_dim=_dim_,
    gt_key="lidar_bboxes_labels",
    enable_dn=True,
    level_index=[0],
    num_views=1,
    cls_threshold_to_reg=0.05,
    bev_size=bev_size,
    instance_bank=dict(
        type="MemoryBank",
        num_anchor=384,
        embed_dims=_dim_,
        num_memory_instances=384,
        anchor=anchor_file,
        num_temp_instances=128 if with_track else 0,
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
        in_channels=sparse_embed_dims * 2,
        pre_norm=True,
        embed_dims=sparse_embed_dims,
        feedforward_channels=sparse_embed_dims * 4,
        num_fcs=2,
        ffn_drop=sparse_drop_out,
    ),
    deformable_model=dict(
        type="DeformableFeatureAggregation",
        use_int16_matmul=False,
        embed_dims=sparse_embed_dims,
        num_groups=sparse_num_groups,
        num_levels=sparse_num_levels,
        num_cams=1,
        attn_drop=0.15,
        use_camera_embed=True,
        sample_from_bev=True,
        residual_mode="cat",
        kps_generator=dict(
            type="SparseBEVKeyPointsGenerator",
            num_pts=8,
        ),
    ),
    refine_layer=dict(
        type="SparseBEVRefinementModule",
        embed_dims=sparse_embed_dims,
        num_cls=dynamic_objects_num_classes,
        refine_yaw=True,
    ),
    target=dict(
        type="SparseBEVTarget",
        num_dn_groups=num_dn_groups,
        num_temp_dn_groups=num_temp_dn_groups,
        dn_noise_scale=[2.0] * 3 + [0.5] * 7,
        max_dn_gt=32,
        add_neg_dn=True,
        cls_weight=2.0,
        box_weight=0.25,
        reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,
    ),
    loss_cls=dict(
        type="FocalLoss",
        loss_name="cls",
        num_classes=dynamic_objects_num_classes + 1,
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

static_objects_det_head = dict(
    type="DenseBEVSparseHead",
    with_track=with_track,
    filter_empty_gt=True,
    meta_process_func=static_objects_process_func,
    use_adapter=use_adapter,
    in_dim=_dim_,
    gt_key="lidar_bboxes_labels",
    enable_dn=True,
    level_index=[0],
    num_views=1,
    cls_threshold_to_reg=0.05,
    bev_size=bev_size,
    instance_bank=dict(
        type="MemoryBank",
        num_anchor=200,
        embed_dims=_dim_,
        num_memory_instances=200,
        anchor=anchor_file,
        num_temp_instances=80 if with_track else 0,
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
        in_channels=sparse_embed_dims * 2,
        pre_norm=True,
        embed_dims=sparse_embed_dims,
        feedforward_channels=sparse_embed_dims * 4,
        num_fcs=2,
        ffn_drop=sparse_drop_out,
    ),
    deformable_model=dict(
        type="DeformableFeatureAggregation",
        use_int16_matmul=False,
        embed_dims=sparse_embed_dims,
        num_groups=sparse_num_groups,
        num_levels=sparse_num_levels,
        num_cams=1,
        attn_drop=0.15,
        use_camera_embed=True,
        sample_from_bev=True,
        residual_mode="cat",
        kps_generator=dict(
            type="SparseBEVKeyPointsGenerator",
            num_pts=8,
        ),
    ),
    refine_layer=dict(
        type="SparseBEVRefinementModule",
        embed_dims=sparse_embed_dims,
        num_cls=static_objects_num_classes,
        refine_yaw=True,
    ),
    target=dict(
        type="SparseBEVTarget",
        num_dn_groups=num_dn_groups,
        num_temp_dn_groups=num_temp_dn_groups,
        dn_noise_scale=[2.0] * 3 + [0.5] * 7,
        max_dn_gt=32,
        add_neg_dn=True,
        cls_weight=2.0,
        box_weight=0.25,
        reg_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 4,
        cls_wise_reg_weights={
            static_objects_class_names.index("traffic_cone"): [
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
    cls_allow_reverse=[static_objects_class_names.index("barrier")],
    loss_cls=dict(
        type="FocalLoss",
        loss_name="cls",
        num_classes=static_objects_num_classes + 1,
        gamma=2.0,
        alpha=0.25,
        loss_weight=2.0,
    ),
    loss_reg=dict(type="L1Loss", loss_weight=0.25),
    loss_cns=dict(type="CrossEntropyLoss", use_sigmoid=True),
    loss_yns=dict(type="GaussianFocalLoss"),
    decoder=dict(type="SparseBEVDecoder", num_output=120),
    reg_weights=[2.0] * 3 + [1.0] * 7,
)

occ_head = dict(
    type="BevformerOccDetDecoder",
    use_adapter=use_adapter,
    in_dim=_dim_,
    use_mask=True,
    lidar_input=lidar_input,
    camera_input=camera_input,
    num_classes=num_classes_occ,
    roi_resizer=dict(
        type="RoiResize",
        in_strides=[1],
        roi_resize_cfgs=[
            dict(
                in_stride=1,
                roi_box=(14, 14, 114, 114),
            )
        ],
    ),
    occ_head=dict(
        type="BEVOCCHead2D",
        in_dim=_dim_,
        out_dim=128,
        Dz=16,
        num_classes=num_classes_occ,
        use_predicter=True,
        use_upsample=True,
    ),
    loss_occ=dict(
        type="CrossEntropyLoss",
        use_sigmoid=False,
        ignore_index=255,
        loss_weight=1.0,
    ),
)

maptr_head = dict(
    type="MaptrDetDecoder",
    use_adapter=use_adapter,
    in_dim=_dim_,
    sd_map_fusion=False,
    roi_resizer=dict(
        type="RoiResize",
        in_strides=[1],
        roi_resize_cfgs=[
            dict(
                in_stride=1,
                roi_box=(39, 14, 89, 114),
            )
        ],
    ),
    osm_encoder=dict(
        type="ConvDown",
        in_dim=1,
        mid_dim=hidden_dim // 2,
        out_dim=hidden_dim,
        quant_input=True,
    ),
    bev_fusion=dict(
        type="MapFusion",
        input_dim=_dim_,
        embed_dims=hidden_dim,
        bev_h=map_bev_h_,
        bev_w=map_bev_w_,
        bev_down=dict(
            type="ConvDown",
            in_dim=_dim_,
            mid_dim=_dim_,
            out_dim=_dim_,
            quant_input=False,
        ),
        fusion_up=dict(
            type="ConvUp",
            in_dim=2 * hidden_dim,
            mid_dim=_dim_,
            out_dim=_dim_,
        ),
    ),
    bev_decoders=[
        dict(
            type="MapInstanceDetectorHead",
            in_channels=_dim_,
            num_cam=6,
            bev_h=map_bev_h_,
            bev_w=map_bev_w_,
            embed_dims=head_embed_dims,
            num_vec_one2one=num_vec_one2one,
            num_vec_one2many=num_vec_one2many,
            k_one2many=6,
            num_pts_per_vec=fixed_ptsnum_per_pred_line,
            num_pts_per_gt_vec=fixed_ptsnum_per_gt_line,
            transform_method="minmax",
            gt_shift_pts_pattern="v2",
            code_size=2,
            num_classes=num_map_classes,
            aux_seg=aux_seg_cfg,
            decoder=dict(
                type="MapInstanceDecoder",
                num_layers=6,
                return_intermediate=True,
                decoder_layer=dict(
                    type="DetrTransformerDecoderLayer",
                    embed_dims=head_embed_dims,
                    crossattention=dict(
                        type="HorizonMultiPointDeformableAttention",
                        embed_dims=head_embed_dims,
                        num_levels=1,
                        grid_align_num=2,
                        num_points=fixed_ptsnum_per_pred_line,
                        feats_size=[[map_bev_w_, map_bev_h_]],
                    ),
                    dropout=0.1,
                ),
            ),
            criterion=dict(
                type="MapTRCriterion",
                dir_interval=1,
                num_classes=num_map_classes,
                code_weights=[1.0, 1.0, 1.0, 1.0],
                sync_cls_avg_factor=True,
                pc_range=map_point_cloud_range,
                num_pts_per_vec=fixed_ptsnum_per_pred_line,
                num_pts_per_gt_vec=fixed_ptsnum_per_gt_line,
                gt_shift_pts_pattern="v2",
                aux_seg=aux_seg_cfg,
                assigner=dict(
                    type="MapTRAssigner",
                    cls_cost=dict(type="FocalLossCost", weight=8.0),
                    pts_cost=dict(
                        type="OrderedPtsL1Cost", weight=3.75, beta=0.01
                    ),
                    pc_range=map_point_cloud_range,
                ),
                loss_cls=dict(
                    type="FocalLoss",
                    loss_name="cls",
                    num_classes=num_map_classes + 1,
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=4.0,
                    reduction="mean",
                ),
                loss_pts=dict(type="PtsL1Loss", loss_weight=3.75, beta=0.01),
                loss_dir=dict(type="PtsDirCosLoss", loss_weight=0.0075),
                loss_seg=dict(
                    type="SimpleLoss", pos_weight=6.0, loss_weight=1.0
                ),
            ),
            post_process=dict(
                type="MapTRPostProcess",
                post_center_range=[-25, -45, -25, -45, 25, 45, 25, 45],
                pc_range=map_point_cloud_range,
                max_num=50,
                num_classes=num_map_classes,
            ),
        ),
    ],
)

discrete_map_head = dict(
    type="BEVFormerDet2DDecoder",
    use_adapter=use_adapter,
    meta_process_func=discrete_map_process_func,
    bev_h=bev_h_,
    bev_w=bev_w_,
    num_query=discrete_map_num_query,
    embed_dims=_dim_,
    pc_range=point_cloud_range,
    code_size=discrete_map_bbox_size,
    num_classes=discrete_map_num_classes,
    decoder=dict(
        type="Detection2DTransformerDecoder",
        num_layers=6,
        return_intermediate=True,
        decoder_layer=dict(
            type="DetrTransformerDecoderLayer",
            crossattention=dict(
                type="HorizonMultiScaleDeformableAttention",
                embed_dims=_dim_,
                num_levels=1,
                batch_first=False,
                grid_align_num=10,
                feats_size=[[bev_w_, bev_h_]],
            ),
            dropout=0.1,
        ),
    ),
    criterion=dict(
        type="BevFormer2DCriterion",
        code_weights=discrete_map_code_weights,
        num_classes=discrete_map_num_classes,
        assigner=dict(
            type="BevFormerHungarianAssigner2D",
            cls_cost=dict(type="FocalLossCost", weight=2.0),
            reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
        ),
        loss_cls=dict(
            type="FocalLoss",
            loss_name="cls",
            num_classes=discrete_map_num_classes + 1,
            alpha=0.25,
            gamma=2.0,
            loss_weight=2.0,
            reduction="mean",
        ),
        loss_bbox=dict(
            type="L1Loss",
            loss_weight=0.25,
        ),
        pc_range=point_cloud_range,
        bbox_key="lidar_bboxes_labels",
    ),
    post_process=dict(
        type="BevFormer2DProcess",
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        pc_range=point_cloud_range,
        max_num=int(discrete_map_num_query * 0.6),
        num_classes=discrete_map_num_classes,
    ),
)

bev_decoders = OrderedDict()
if use_dynamic_det_head:
    bev_decoders["dynamic_objects_det"] = dynamic_bev_head
if use_static_objects_det_head:
    bev_decoders["static_objects_det"] = static_objects_det_head
if use_occ_head:
    bev_decoders["occ"] = occ_head
if use_discrete_map_head:
    bev_decoders["discrete_map"] = discrete_map_head
if use_maptr_head:
    bev_decoders["maptr"] = maptr_head

# model settings
model = dict(
    type="BEVFusionMultitask",
    lidar_network=lidar_network if lidar_input else None,
    camera_network=camera_network if camera_input else None,
    task_decoders=bev_decoders,
    fuse_module=dict(type="BevFuseModule", input_c=384 + _dim_, fuse_c=_dim_),
    bev_h=bev_h_,
    bev_w=bev_w_,
    loss_weight_dict={
        "dynamic_objects_det": 4,
        "occ": 96,
        "maptr": 1.1,
        "discrete_map": 2,
        "static_objects_det": 2,
    },
)


test_model = copy.deepcopy(model)


# --- Stage models (3-stage float training) ---
# Stage LR / epoch / step params live in common.py (shared with train.py).

model_stage1 = copy.deepcopy(model)
model_stage1["task_decoders"] = OrderedDict(
    dynamic_objects_det=dynamic_bev_head,
    static_objects_det=static_objects_det_head,
    occ=occ_head,
    discrete_map=discrete_map_head,
)
model_stage1["loss_weight_dict"] = {
    "dynamic_objects_det": 4,
    "occ": 90,
    "discrete_map": 2,
    "static_objects_det": 2,
}

# stage2 maptr, only train head
model_stage2 = copy.deepcopy(model)
model_stage2["task_decoders"] = OrderedDict(
    maptr=maptr_head,
)
model_stage2["loss_weight_dict"] = {
    "maptr": 1,
}

# stage3 full tasks
# Stage LR / epoch / step params live in common.py (shared with train.py).

model_stage3 = copy.deepcopy(model)
model_stage3["task_decoders"] = OrderedDict(
    dynamic_objects_det=dynamic_bev_head,
    static_objects_det=static_objects_det_head,
    occ=occ_head,
    discrete_map=discrete_map_head,
    maptr=maptr_head,
)
model_stage3["loss_weight_dict"] = {
    "dynamic_objects_det": 4,
    "occ": 96,
    "maptr": 1.1,
    "discrete_map": 2,
    "static_objects_det": 2,
}

# =====================================================================
# Deploy / HBIR export (tensor-only inputs, no seq_meta)
# =====================================================================
_num_cam = 6
_virtual_bev_h = 64
_virtual_bev_w = 80
_num_points_in_pillar = 2
_virtual_len = _virtual_bev_h * _virtual_bev_w
_dyn_mem = dynamic_bev_head["instance_bank"]["num_memory_instances"]
_sta_mem = static_objects_det_head["instance_bank"]["num_memory_instances"]

deploy_model = copy.deepcopy(model)
deploy_model["camera_network"]["view_transformer"]["is_compile"] = True

# DenseBEVSparseHead: compile path + MemoryBank keys for IrInfer.
for _det_name, _task_prefix in (
    ("dynamic_objects_det", "dynamic_objects"),
    ("static_objects_det", "static_objects"),
):
    if _det_name not in deploy_model["task_decoders"]:
        continue
    _det = deploy_model["task_decoders"][_det_name]
    _det["is_compile"] = True
    _det["enable_dn"] = False
    _det["instance_bank"]["task_name"] = _task_prefix
    _det["deformable_model"]["attn_drop"] = 0.0
    for _k in (
        "target",
        "loss_cls",
        "loss_reg",
        "loss_cns",
        "loss_yns",
        "decoder",
    ):
        _det.pop(_k, None)

if "occ" in deploy_model["task_decoders"]:
    deploy_model["task_decoders"]["occ"]["is_compile"] = True
    deploy_model["task_decoders"]["occ"].pop("loss_occ", None)

if "discrete_map" in deploy_model["task_decoders"]:
    _dm = deploy_model["task_decoders"]["discrete_map"]
    _dm["is_compile"] = True
    _dm["return_bev_embed"] = True
    _dm.pop("criterion", None)
    _dm.pop("post_process", None)

if "maptr" in deploy_model["task_decoders"]:
    _mh = deploy_model["task_decoders"]["maptr"]["bev_decoders"][0]
    _mh["is_deploy"] = True
    _mh["aux_seg"] = dict(
        use_aux_seg=False,
        bev_seg=False,
        pv_seg=False,
    )
    _mh.pop("criterion", None)
    _mh.pop("post_process", None)

deploy_inputs = {
    "img": torch.randn((_num_cam, 3, input_size[0], input_size[1])),
    "projection_mat": torch.randn((1, 4, 4)),
    "queries_rebatch_grid": torch.randn(
        (_num_cam, _virtual_bev_h, _virtual_bev_w, 2)
    ),
    "restore_bev_grid": torch.randn(
        (1, max_camoverlap_num * bev_h_, bev_w_, 2)
    ),
    "reference_points_rebatch": torch.randn(
        (_num_cam, _virtual_len, _num_points_in_pillar, 2)
    ),
    "bev_pillar_counts": torch.randn((1, bev_h_ * bev_w_, 1)),
    "dynamic_objects_cached_anchor": torch.randn((1, _dyn_mem, 11)),
    "dynamic_objects_cached_feature": torch.randn((1, _dyn_mem, _dim_)),
    "static_objects_cached_anchor": torch.randn((1, _sta_mem, 11)),
    "static_objects_cached_feature": torch.randn((1, _sta_mem, _dim_)),
}

# Flat export names (tree_flatten order of deploy_model outputs). Aligns with
# BevMultitaskIrInfer HBM remapping.
output_names = [
    "dynamic_objects_det.classification",
    "dynamic_objects_det.prediction",
    "dynamic_objects_det.quality",
    "dynamic_objects_det.feature",
    "static_objects_det.classification",
    "static_objects_det.prediction",
    "static_objects_det.quality",
    "static_objects_det.feature",
    "occ",
    "bev_embed",
    "discrete_map.classification",
    "discrete_map.ref_out",
    "discrete_map.prediction",
    "om.classification",
    "om.prediction",
]

first_frame_input = {
    "dynamic_objects_cached_anchor": torch.zeros((1, _dyn_mem, 11)),
    "dynamic_objects_cached_feature": torch.zeros((1, _dyn_mem, _dim_)),
    "dynamic_objects_cached_confidence": torch.zeros((1, _dyn_mem)),
    "static_objects_cached_anchor": torch.zeros((1, _sta_mem, 11)),
    "static_objects_cached_feature": torch.zeros((1, _sta_mem, _dim_)),
    "static_objects_cached_confidence": torch.zeros((1, _sta_mem)),
    "mask": torch.zeros((1,)).bool(),
}

hbir_infer_model = dict(
    type="BevMultitaskIrInfer",
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized_infer.bc"),
    ),
    deploy_model=deploy_model,
    first_frame_input=first_frame_input,
    global_mat_key="lidar2global",
    use_memory_bank=True,
    dynamic_objects_confidence_decay=0.6,
    dynamic_objects_num_temp_instances=128,
    dynamic_objects_num_memory_instances=_dyn_mem,
    static_objects_confidence_decay=0.6,
    static_objects_num_temp_instances=80,
    static_objects_num_memory_instances=_sta_mem,
    use_nv12=False,
    dynamic_objects_det_decoder=dict(type="SparseBEVDecoder"),
    static_objects_det_decoder=dict(type="SparseBEVDecoder"),
    maptr_post_process=dict(
        type="MapTRPostProcess",
        post_center_range=[-25, -45, -25, -45, 25, 45, 25, 45],
        pc_range=map_point_cloud_range,
        max_num=50,
        num_classes=num_map_classes,
    ),
    discrete_map_post_process=dict(
        type="BevFormer2DProcess",
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        pc_range=point_cloud_range,
        max_num=int(discrete_map_num_query * 0.6),
        num_classes=discrete_map_num_classes,
    ),
)

hbm_infer_model = copy.deepcopy(hbir_infer_model)
hbm_infer_model["use_nv12"] = True
hbm_infer_model["ir_model"] = dict(
    type="HbmModule",
    model_path=os.path.join(ckpt_dir, "compile", "model.hbm"),
    host="10.103.43.100",
    return_tensor=True,
)
