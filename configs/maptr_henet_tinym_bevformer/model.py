"""Model definitions for MapTR HENet-tiny BEVFormer NuScenes."""

import copy
import os

from common import (
    _num_levels_,
    aux_seg_cfg,
    bev_embed_dims,
    bev_h_,
    bev_sparse_rate,
    bev_w_,
    ckpt_dir,
    fixed_ptsnum_per_gt_line,
    fixed_ptsnum_per_pred_line,
    head_embed_dims,
    hidden_dim,
    k_one2many,
    max_camoverlap_num,
    num_map_classes,
    num_vec_one2many,
    num_vec_one2one,
    point_cloud_range,
    post_center_range,
    queue_length,
    single_bev,
    test_queue_length,
    use_lidar2img,
)

model = dict(
    type="MapTR",
    out_indices=(-1,),
    sd_map_fusion=True,
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
        stage_out_norm=True,
    ),
    neck=dict(
        type="FPN",
        in_strides=[32],
        in_channels=[384],
        out_strides=[32],
        out_channels=[bev_embed_dims],
        bn_kwargs=dict(eps=1e-5, momentum=0.1),
    ),
    view_transformer=dict(
        type="SingleFrameBevFormerViewTransformer",
        bev_h=bev_h_,
        bev_w=bev_w_,
        pc_range=point_cloud_range,
        num_points_in_pillar=4,
        embed_dims=bev_embed_dims,
        queue_length=queue_length,
        in_indices=(-1,),
        single_bev=single_bev,
        use_lidar2img=use_lidar2img,
        max_camoverlap_num=max_camoverlap_num,
        virtual_bev_h=int(bev_sparse_rate * bev_h_),
        virtual_bev_w=bev_w_,
        positional_encoding=dict(
            type="LearnedPositionalEncoding",
            num_feats=bev_embed_dims // 2,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        encoder=dict(
            type="SingleFrameBEVFormerEncoder",
            num_layers=1,
            return_intermediate=False,
            bev_h=bev_h_,
            bev_w=bev_w_,
            embed_dims=bev_embed_dims,
            encoder_layer=dict(
                type="SingleFrameBEVFormerEncoderLayer",
                embed_dims=bev_embed_dims,
                selfattention=dict(
                    type="HorizonMultiScaleDeformableAttention",
                    embed_dims=bev_embed_dims,
                    num_levels=1,
                    grid_align_num=10,
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
                        embed_dims=bev_embed_dims,
                        num_points=8,
                        num_levels=_num_levels_,
                        grid_align_num=20,
                        feats_size=[[25, 15]],
                    ),
                    embed_dims=bev_embed_dims,
                ),
                dropout=0.1,
            ),
        ),
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
        input_dim=bev_embed_dims,
        embed_dims=hidden_dim,
        bev_h=bev_h_,
        bev_w=bev_w_,
        bev_down=dict(
            type="ConvDown",
            in_dim=bev_embed_dims,
            mid_dim=bev_embed_dims,
            out_dim=bev_embed_dims,
            quant_input=False,
        ),
        fusion_up=dict(
            type="ConvUp",
            in_dim=2 * hidden_dim,
            mid_dim=bev_embed_dims,
            out_dim=bev_embed_dims,
        ),
    ),
    bev_decoders=[
        dict(
            type="MapInstanceDetectorHead",
            in_channels=bev_embed_dims,
            num_cam=6,
            bev_h=bev_h_,
            bev_w=bev_w_,
            embed_dims=head_embed_dims,
            num_vec_one2one=num_vec_one2one,
            num_vec_one2many=num_vec_one2many,
            k_one2many=k_one2many,
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
                        feats_size=[[bev_w_, bev_h_]],
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
                pc_range=point_cloud_range,
                num_pts_per_vec=fixed_ptsnum_per_pred_line,
                num_pts_per_gt_vec=fixed_ptsnum_per_gt_line,
                gt_shift_pts_pattern="v2",
                aux_seg=aux_seg_cfg,
                assigner=dict(
                    type="MapTRAssigner",
                    cls_cost=dict(type="FocalLossCost", weight=4.0),
                    pts_cost=dict(
                        type="OrderedPtsL1Cost", weight=2.5, beta=0.01
                    ),
                    pc_range=point_cloud_range,
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
                loss_pts=dict(type="PtsL1Loss", loss_weight=2.5, beta=0.01),
                loss_dir=dict(type="PtsDirCosLoss", loss_weight=0.005),
                loss_seg=dict(
                    type="SimpleLoss", pos_weight=4.0, loss_weight=1.0
                ),
                loss_pv_seg=dict(
                    type="SimpleLoss", pos_weight=1.0, loss_weight=2.0
                ),
            ),
            post_process=dict(
                type="MapTRPostProcess",
                post_center_range=post_center_range,
                pc_range=point_cloud_range,
                max_num=50,
                num_classes=num_map_classes,
            ),
        ),
    ],
)

calib_model = copy.deepcopy(model)
calib_model["bev_decoders"][0]["num_vec"] = num_vec_one2one + num_vec_one2many

test_model = copy.deepcopy(model)
test_model["view_transformer"]["queue_length"] = test_queue_length
test_model["bev_decoders"][0]["queue_length"] = test_queue_length
test_model["bev_decoders"][0]["aux_seg"] = dict(
    use_aux_seg=False,
    bev_seg=False,
    pv_seg=False,
)
test_model["bev_decoders"][0].pop("criterion")

deploy_model = copy.deepcopy(model)
deploy_model["view_transformer"]["queue_length"] = 1
deploy_model["view_transformer"]["is_compile"] = True
deploy_model["bev_decoders"][0]["queue_length"] = 1
deploy_model["bev_decoders"][0]["is_deploy"] = True
deploy_model["bev_decoders"][0]["aux_seg"] = dict(
    use_aux_seg=False,
    bev_seg=False,
    pv_seg=False,
)
deploy_model["bev_decoders"][0].pop("criterion")
deploy_model["bev_decoders"][0].pop("post_process")

from qconfig import calibration_predict_model_convert_pipeline  # noqa: E402

hbir_infer_model = dict(
    type="MapTRIrInfer",
    sd_map_fusion=True,
    test_model=copy.deepcopy(test_model),
    model_convert_pipeline=calibration_predict_model_convert_pipeline,
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized_infer.bc"),
    ),
)

single_hbir_infer_model = copy.deepcopy(hbir_infer_model)
single_hbir_infer_model["test_model"]["bev_decoders"][0]["post_process"] = (
    dict(
        type="MapTRPostProcess",
        post_center_range=post_center_range,
        pc_range=point_cloud_range,
        max_num=50,
        num_classes=num_map_classes,
        score_threshold=0.2,
    )
)

hbm_infer_model = copy.deepcopy(single_hbir_infer_model)
hbm_infer_model["is_nv12_type"] = True
hbm_infer_model["ir_model"] = dict(
    type="HbmModule",
    model_path=os.path.join(ckpt_dir, "compile/model.hbm"),
    host="10.103.43.99",
    return_tensor=True,
)
