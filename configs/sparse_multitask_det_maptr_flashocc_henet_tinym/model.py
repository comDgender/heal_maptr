"""Model definitions for sparse multitask det + MapTR + FlashOcc."""

import copy
import os
from collections import OrderedDict

import torch
from common import (
    CLASSES,
    ckpt_dir,
    depth_grid_quant_scale,
    det_anchor_file,
    drop_out,
    embed_dims,
    enable_det_head,
    enable_occ_head,
    enable_om_head,
    fixed_ptsnum_per_gt_line,
    fixed_ptsnum_per_pred_line,
    grid_quant_scale,
    loss_weights,
    map_anchor_file,
    map_aux_seg_cfg,
    map_classes,
    map_point_cloud_range,
    map_post_center_range,
    num_anchors,
    num_classes,
    num_classes_occ,
    num_decoder,
    num_depth_layers,
    num_groups,
    num_levels,
    num_map_classes,
    num_single_frame_decoder,
    occ_bev_size,
    occ_depth,
    occ_grid_size,
    occ_num_points,
    om_num_anchor,
    temp_anchors,
    vt_input_hw,
)

det_head = dict(
    type="SparseBEVOEHead",
    enable_dn=True,
    level_index=[2],
    cls_threshold_to_reg=0.05,
    instance_bank=dict(
        type="MemoryBankOE",
        num_anchor=num_anchors,
        embed_dims=embed_dims,
        num_memory_instances=num_anchors,
        anchor=det_anchor_file,
        num_temp_instances=temp_anchors,
        confidence_decay=0.6,
    ),
    anchor_encoder=dict(
        type="SparseBEVOEEncoder",
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
        type="AsymmetricFFNOE",
        in_channels=embed_dims * 2,
        pre_norm=True,
        embed_dims=embed_dims,
        feedforward_channels=embed_dims * 4,
        num_fcs=2,
        ffn_drop=drop_out,
    ),
    deformable_model=dict(
        type="DeformableFeatureAggregationOE",
        embed_dims=embed_dims,
        num_groups=num_groups,
        num_levels=num_levels,
        num_cams=6,
        attn_drop=0.15,
        use_camera_embed=True,
        residual_mode="cat",
        kps_generator=dict(
            type="SparseBEVOEKeyPointsGenerator",
            num_pts=8,
        ),
        use_int16_matmul=False,
    ),
    refine_layer=dict(
        type="SparseBEVOERefinementModule",
        embed_dims=embed_dims,
        num_cls=num_classes,
        refine_yaw=True,
    ),
    target=dict(
        type="SparseBEVOETarget",
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
    decoder=dict(type="SparseBEVOEDecoder"),
    reg_weights=[2.0] * 3 + [1.0] * 7,
)

occ_head = dict(
    type="FlashOccHead",
    bev_feat_index=-1,
    bev_upscale=2,
    view_transformer=dict(
        type="LSSTransformer",
        in_channels=256,
        feat_channels=64,
        z_range=(-1.0, 5.4),
        depth=occ_depth,
        num_points=occ_num_points,
        bev_size=occ_bev_size,
        grid_size=occ_grid_size,
        num_views=6,
        grid_quant_scale=grid_quant_scale,
        depth_grid_quant_scale=depth_grid_quant_scale,
        homo_key="lidar2img",
    ),
    bev_encoder=dict(
        type="BevEncoder",
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
            quant_input=False,
        ),
        neck=dict(
            type="BiFPN",
            in_strides=[2, 4, 8, 16, 32],
            out_strides=[2, 4, 8, 16, 32],
            stride2channels=dict({2: 64, 4: 64, 8: 128, 16: 192, 32: 384}),
            out_channels=48,
            num_outs=5,
            stack=3,
            start_level=0,
            end_level=-1,
            fpn_name="bifpn_sum",
            upsample_type="function",
            use_fx=False,
        ),
    ),
    bev_decoder=dict(
        type="FlashOccDetDecoder",
        use_mask=True,
        num_classes=num_classes_occ,
        occ_head=dict(
            type="BEVOCCHead2D",
            in_dim=48,
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
    ),
)

om_head = dict(
    type="SparseMapPerceptionDecoder",
    embed_dims=embed_dims,
    num_cam=6,
    num_vec_one2one=om_num_anchor,
    num_vec_one2many=0,
    k_one2many=0,
    num_pts_per_vec=fixed_ptsnum_per_pred_line,
    transform_method="minmax",
    is_deploy=False,
    decoder=dict(
        type="SparseOMOEHead",
        feat_indices=[2],
        num_anchor=om_num_anchor,
        num_pts_per_vec=fixed_ptsnum_per_pred_line,
        num_views=6,
        projection_mat_key="lidar2img",
        anchor_encoder=dict(
            type="SparseOEPoint3DEncoder",
            embed_dims=embed_dims,
            input_dim=fixed_ptsnum_per_pred_line * 2,
        ),
        instance_bank=dict(
            type="InstanceBankOE",
            num_anchor=om_num_anchor,
            embed_dims=embed_dims,
            anchor=map_anchor_file,
            num_temp_instances=0,
            confidence_decay=0.6,
            anchor_grad=True,
            feat_grad=True,
        ),
        instance_interaction=dict(
            type="MultiheadAttention",
            embed_dim=embed_dims,
            num_heads=num_groups,
            batch_first=True,
            attn_drop=drop_out,
            proj_drop=drop_out,
        ),
        norm_layer=dict(
            type=torch.nn.LayerNorm,
            normalized_shape=embed_dims,
        ),
        ffn=dict(
            type="AsymmetricFFNOE",
            in_channels=embed_dims * 2,
            pre_norm=True,
            embed_dims=embed_dims,
            num_fcs=2,
            ffn_drop=0.1,
            feedforward_channels=embed_dims * 4,
        ),
        deformable_model=dict(
            type="DeformableFeatureAggregationOEv2",
            embed_dims=embed_dims,
            num_groups=num_groups,
            num_levels=num_levels,
            num_cams=6,
            attn_drop=0.15,
            residual_mode="cat",
            use_camera_embed=True,
            grid_align_num=4,
            kps_generator=dict(
                type="SparsePoint3DKeyPointsGenerator",
                embed_dims=embed_dims,
                num_sample=fixed_ptsnum_per_pred_line,
                num_learnable_pts=2,
                fix_height=(0, 0.5, -0.5),
                ground_height=-1.84023,
            ),
        ),
        refine_layer=dict(
            type="SparsePoint3DRefinementModule",
            embed_dims=embed_dims,
            num_sample=fixed_ptsnum_per_pred_line,
            coords_dim=2,
            num_cls=len(map_classes),
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
    ),
    aux_seg=map_aux_seg_cfg,
    criterion=dict(
        type="MapTRCriterion",
        dir_interval=1,
        num_classes=num_map_classes,
        code_weights=[1.0, 1.0, 1.0, 1.0],
        sync_cls_avg_factor=True,
        pc_range=map_point_cloud_range,
        num_pts_per_vec=fixed_ptsnum_per_pred_line,
        num_pts_per_gt_vec=fixed_ptsnum_per_gt_line,
        gt_shift_pts_pattern="sparsedrive",
        aux_seg=map_aux_seg_cfg,
        pred_absolute_points=True,
        assigner=dict(
            type="MapTRAssigner",
            cls_cost=dict(type="FocalLossCost", weight=1.0),
            pts_cost=dict(type="OrderedPtsL1Cost", weight=0.5, beta=0.01),
            pc_range=map_point_cloud_range,
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
        loss_pv_seg=dict(type="SimpleLoss", pos_weight=1.0, loss_weight=2.0),
    ),
    post_process=dict(
        type="MapTRPostProcess",
        pc_range=map_point_cloud_range,
        post_center_range=map_post_center_range,
        max_num=om_num_anchor,
        num_classes=num_map_classes,
        pred_absolute_points=True,
    ),
)

task_heads = OrderedDict()
if enable_det_head:
    task_heads["det"] = det_head
if enable_om_head:
    task_heads["om"] = om_head
if enable_occ_head:
    task_heads["occ"] = occ_head

model = dict(
    type="SparseMultitaskOE",
    compiler_model=False,
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
        type="MMFPN",
        in_strides=[2, 4, 8, 16, 32],
        in_channels=[64, 64, 128, 192, 384],
        fix_out_channel=embed_dims,
        out_strides=[4, 8, 16, 32],
    ),
    depth_branch=dict(
        type="DenseDepthNetOE",
        embed_dims=embed_dims,
        num_depth_layers=num_depth_layers,
        loss_weight=0.2,
        points_key="lidar.points",
    ),
    head=task_heads,
    loss_weights=loss_weights,
    split_neck=True,
)

test_model = copy.deepcopy(model)
if "om" in test_model["head"]:
    test_model["head"]["om"]["aux_seg"] = dict(
        use_aux_seg=False,
        bev_seg=False,
        pv_seg=False,
        dense_depth=False,
    )
    test_model["head"]["om"].pop("criterion", None)

deploy_model = copy.deepcopy(model)
deploy_model["compiler_model"] = True
deploy_model.pop("depth_branch", None)
if "det" in deploy_model["head"]:
    deploy_model["head"]["det"]["enable_dn"] = False
    deploy_model["head"]["det"].pop("target", None)
    deploy_model["head"]["det"].pop("loss_cls", None)
    deploy_model["head"]["det"].pop("loss_reg", None)
    deploy_model["head"]["det"].pop("loss_cns", None)
    deploy_model["head"]["det"].pop("loss_yns", None)
    deploy_model["head"]["det"].pop("decoder", None)
if "om" in deploy_model["head"]:
    deploy_model["head"]["om"]["is_deploy"] = True
    deploy_model["head"]["om"]["aux_seg"] = dict(
        use_aux_seg=False,
        bev_seg=False,
        pv_seg=False,
        dense_depth=False,
    )
    deploy_model["head"]["om"].pop("criterion", None)
    # Keep ``post_process`` on deploy_model. With ``is_deploy=True`` the head's
    # ``forward`` short-circuits to raw tensors (never calls _post_process), so
    # MapTRPostProcess is never traced into the compiled IR — it is dead
    # weight for compilation. But SparseMultitaskIrInfer.forward calls
    # ``deploy_model.head["om"].post_process`` directly (after get_outputs)
    # to decode om results for IR/HBM inference; popping it here crashes
    # that path with ``'NoneType' object is not callable``. See sparse_maptr
    # config, which pops post_process off deploy_model but re-attaches it
    # for IR-infer.
    # deploy_model["head"]["om"].pop("post_process", None)
if "occ" in deploy_model["head"]:
    deploy_model["head"]["occ"]["bev_decoder"]["is_compile"] = True

first_frame_input = {
    "cached_anchor": torch.zeros((1, num_anchors, 11)),
    "cached_feature": torch.zeros((1, num_anchors, embed_dims)),
    "cached_confidence": torch.zeros((1, num_anchors)),
    "mask": torch.zeros((1)).bool(),
}

from qconfig import float_stage1_predict_model_convert_pipeline  # noqa: E402

hbir_infer_model = dict(
    type="SparseMultitaskOEIrInfer",
    first_frame_input=first_frame_input,
    projection_mat_key="lidar2img",
    global_mat_key="lidar2global",
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized_infer.bc"),
    ),
    deploy_model=deploy_model,
    det_decoder=dict(type="SparseBEVOEDecoder"),
    use_memory_bank=True,
    confidence_decay=0.6,
    num_temp_instances=temp_anchors,
    num_memory_instances=num_anchors,
    vt_input_hw=vt_input_hw,
    model_convert_pipeline=float_stage1_predict_model_convert_pipeline,
)

# On-board BPU runtime after compile (``compile/model.hbm``).
# Same as hbir_infer_model but with a real HbmModule (compiled HBM) and
# use_nv12=True; model_convert_pipeline is dropped — the HBM is already
# compiled, so the QAT convert pipeline must not re-pipe ir_model.
# ``host`` is the BPU server address; override with your board's IP.
hbm_infer_model = copy.deepcopy(hbir_infer_model)
hbm_infer_model.pop("model_convert_pipeline", None)
hbm_infer_model["use_nv12"] = True
hbm_infer_model["ir_model"] = dict(
    type="HbmModule",
    model_path=os.path.join(ckpt_dir, "compile", "model.hbm"),
    host="10.103.43.99",
    return_tensor=True,
)

output_names = []
if enable_det_head:
    output_names.extend(
        [
            "det.classification",
            "det.prediction",
            "det.quality",
            "det.feature",
        ]
    )
if enable_om_head:
    output_names.extend(["om.classification", "om.prediction"])
if enable_occ_head:
    output_names.append("occ")

# Stage-2 model: map + occ only (backbone frozen). det is not trained in
# stage2; its weights are merged back from stage1 after training (see
# k8s merge_checkpoints step).
model_stage2 = copy.deepcopy(model)
model_stage2["head"] = {"om": om_head, "occ": occ_head}
model_stage2["loss_weights"] = dict(om=0.9, occ=9.0)

test_model_stage2 = copy.deepcopy(model_stage2)
if "om" in test_model_stage2["head"]:
    test_model_stage2["head"]["om"]["aux_seg"] = dict(
        use_aux_seg=False,
        bev_seg=False,
        pv_seg=False,
        dense_depth=False,
    )
    test_model_stage2["head"]["om"].pop("criterion", None)
