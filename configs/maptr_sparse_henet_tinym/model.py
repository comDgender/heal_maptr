"""Model definitions for sparse MapTR HENet-tiny NuScenes.

Structure overview::

    SparseMapTR
    ├── backbone: HENet-tiny (ImageNet pretrained)
    ├── neck: FPN -> 256-dim multi-scale features
    └── bev_decoders[0]: SparseMapPerceptionDecoder
            ├── decoder: SparseMapHead (6-layer sparse query stack)
            ├── criterion: MapTR losses (float train only)
            ├── post_process: MapTRPostProcess (eval only)
            └── depth_branch: dense depth aux loss (float train only)

Variant models:
- ``model``: full training graph with aux losses.
- ``test_model``: eval graph, aux/criterion removed.
- ``deploy_model``: export graph, returns raw cls/reg tensors.
- ``hbir_infer_model``: HBIR runtime + Python post-process.
- ``hbm_infer_model``: on-board BPU runtime (``model.hbm``) + post-process.
"""

import copy
import os

import torch
from common import (
    _dim_,
    anchor_file,
    aux_seg_cfg,
    ckpt_dir,
    drop_out,
    feat_indices,
    fixed_ptsnum_per_gt_line,
    fixed_ptsnum_per_pred_line,
    num_anchor,
    num_decoder,
    num_depth_layers,
    num_groups,
    num_levels,
    num_map_classes,
    num_single_frame_decoder,
    point_cloud_range,
    post_center_range,
)

# Sparse query head: 100 map anchors x 20 points, 6 decoder layers.
# Reuses sparsebev blocks (InstanceBankOE, DeformableFeatureAggregationOEv2).
sparse_head = dict(
    type="SparseMapHead",
    feat_indices=feat_indices,
    num_anchor=num_anchor,
    num_pts_per_vec=fixed_ptsnum_per_pred_line,
    num_views=6,
    projection_mat_key="lidar2img",
    anchor_encoder=dict(
        type="SparsePoint3DEncoder",
        embed_dims=_dim_,
        input_dim=fixed_ptsnum_per_pred_line * 2,
    ),
    instance_bank=dict(
        type="InstanceBankOE",
        num_anchor=num_anchor,
        embed_dims=_dim_,
        anchor=anchor_file,
        num_temp_instances=0,
        confidence_decay=0.6,
        anchor_grad=True,
        feat_grad=True,
    ),
    instance_interaction=dict(
        type="MultiheadAttention",
        embed_dim=_dim_,
        num_heads=num_groups,
        batch_first=True,
        attn_drop=drop_out,
        proj_drop=drop_out,
    ),
    norm_layer=dict(
        type=torch.nn.LayerNorm,
        normalized_shape=_dim_,
    ),
    ffn=dict(
        type="AsymmetricFFNOE",
        in_channels=_dim_ * 2,
        pre_norm=True,
        embed_dims=_dim_,
        num_fcs=2,
        ffn_drop=0.1,
        feedforward_channels=_dim_ * 4,
    ),
    deformable_model=dict(
        type="DeformableFeatureAggregationOEv2",
        embed_dims=_dim_,
        num_groups=num_groups,
        num_levels=num_levels,
        num_cams=6,
        attn_drop=0.15,
        residual_mode="cat",
        use_camera_embed=True,
        grid_align_num=4,
        kps_generator=dict(
            type="SparsePoint3DKeyPointsGenerator",
            embed_dims=_dim_,
            num_sample=fixed_ptsnum_per_pred_line,
            num_learnable_pts=2,
            fix_height=(0, 0.5, -0.5),
            ground_height=-1.84023,
        ),
    ),
    refine_layer=dict(
        type="SparsePoint3DRefinementModule",
        embed_dims=_dim_,
        num_sample=fixed_ptsnum_per_pred_line,
        coords_dim=2,
        num_cls=num_map_classes,
        with_cls_branch=True,
    ),
    num_decoder=num_decoder,
    num_single_frame_decoder=num_single_frame_decoder,
    operation_order=[
        "interaction",
        "norm",
        "deformable",
        "ffn",
        "norm",
        "refine",
    ]
    * num_single_frame_decoder
    + [
        "interaction",
        "norm",
        "deformable",
        "ffn",
        "norm",
        "refine",
    ]
    * (num_decoder - num_single_frame_decoder),
)

model = dict(
    type="SparseMapTR",
    out_indices=(1, 2, 3, 4),
    # Image encoder + FPN; no BEV encoder in sparse MapTR.
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
        stage_out_norm=False,
    ),
    neck=dict(
        type="FPN",
        in_strides=[4, 8, 16, 32],
        in_channels=[64, 128, 192, 384],
        out_strides=[4, 8, 16, 32],
        out_channels=[_dim_, _dim_, _dim_, _dim_],
        bn_kwargs=dict(eps=1e-5, momentum=0.1),
    ),
    bev_decoders=[
        dict(
            type="SparseMapPerceptionDecoder",
            embed_dims=_dim_,
            num_cam=6,
            num_vec_one2one=num_anchor,
            num_vec_one2many=0,
            k_one2many=0,
            num_pts_per_vec=fixed_ptsnum_per_pred_line,
            transform_method="minmax",
            is_deploy=False,
            decoder=sparse_head,
            # PV seg + dense depth aux tasks, active during float training.
            aux_seg=aux_seg_cfg,
            depth_branch=dict(
                type="DenseDepthNetOE",
                embed_dims=_dim_,
                num_depth_layers=num_depth_layers,
                loss_weight=0.2,
                points_key="lidar.points",
            ),
            criterion=dict(
                type="MapTRCriterion",
                dir_interval=1,
                num_classes=num_map_classes,
                code_weights=[1.0, 1.0, 1.0, 1.0],
                sync_cls_avg_factor=True,
                pc_range=point_cloud_range,
                num_pts_per_vec=fixed_ptsnum_per_pred_line,
                num_pts_per_gt_vec=fixed_ptsnum_per_gt_line,
                gt_shift_pts_pattern="sparsedrive",
                aux_seg=aux_seg_cfg,
                pred_absolute_points=True,
                assigner=dict(
                    type="MapTRAssigner",
                    cls_cost=dict(type="FocalLossCost", weight=1.0),
                    pts_cost=dict(
                        type="OrderedPtsL1Cost", weight=0.5, beta=0.01
                    ),
                    pc_range=point_cloud_range,
                    pred_absolute_points=True,
                ),
                loss_cls=dict(
                    type="FocalLoss",
                    loss_name="cls",
                    num_classes=num_map_classes + 1,
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=1.0,
                    reduction="mean",
                ),
                loss_pts=dict(type="PtsL1Loss", loss_weight=0.5, beta=0.01),
                loss_dir=dict(type="PtsDirCosLoss", loss_weight=0),
                loss_pv_seg=dict(
                    type="SimpleLoss", pos_weight=1.0, loss_weight=2.0
                ),
            ),
            post_process=dict(
                type="MapTRPostProcess",
                pc_range=point_cloud_range,
                post_center_range=post_center_range,
                max_num=num_anchor,
                num_classes=num_map_classes,
                pred_absolute_points=True,
            ),
        ),
    ],
)

# Eval: drop aux branches and loss, keep post_process for metrics.
test_model = copy.deepcopy(model)
test_model["bev_decoders"][0]["aux_seg"] = dict(
    use_aux_seg=False,
    bev_seg=False,
    pv_seg=False,
    dense_depth=False,
)
test_model["bev_decoders"][0].pop("criterion")

# Export: is_deploy=True returns raw head outputs for HBIR compilation.
deploy_model = copy.deepcopy(model)
deploy_model["bev_decoders"][0]["is_deploy"] = True
deploy_model["bev_decoders"][0]["aux_seg"] = dict(
    use_aux_seg=False,
    bev_seg=False,
    pv_seg=False,
    dense_depth=False,
)
deploy_model["bev_decoders"][0].pop("criterion")
deploy_model["bev_decoders"][0].pop("post_process")

from qconfig import float_predict_model_convert_pipeline  # noqa: E402

# BPU infer: run quantized.bc then apply MapTRPostProcess in Python.
hbir_infer_model = dict(
    type="SparseMapIrInfer",
    test_model=copy.deepcopy(test_model),
    projection_mat_key="lidar2img",
    model_convert_pipeline=float_predict_model_convert_pipeline,
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized_infer.bc"),
    ),
)

# On-board BPU eval after compile (``compile/model.hbm``).
hbm_infer_model = copy.deepcopy(hbir_infer_model)
hbm_infer_model["use_nv12"] = True
hbm_infer_model["ir_model"] = dict(
    type="HbmModule",
    model_path=os.path.join(ckpt_dir, "compile/model.hbm"),
    host="10.103.43.99",
    return_tensor=True,
)

# Single-frame demo infer with score threshold filtering.
single_hbir_infer_model = copy.deepcopy(hbir_infer_model)
single_hbir_infer_model["test_model"]["bev_decoders"][0]["post_process"] = (
    dict(
        type="MapTRPostProcess",
        pc_range=point_cloud_range,
        post_center_range=post_center_range,
        max_num=num_anchor,
        num_classes=num_map_classes,
        pred_absolute_points=True,
        score_threshold=0.2,
    )
)
