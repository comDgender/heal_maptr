"""Model definitions for BevFormer tiny ResNet50 NuScenes detection."""

import copy
import os

from common import (
    _dim_,
    _num_levels_,
    _pos_dim_,
    bev_h_,
    bev_w_,
    ckpt_dir,
    num_classes,
    point_cloud_range,
)

model = dict(
    type="BevFormer",
    out_indices=(-1,),
    backbone=dict(
        type="ResNet50",
        num_classes=1000,
        bn_kwargs={},
        include_top=False,
    ),
    neck=dict(
        type="FPN",
        in_strides=[32],
        in_channels=[2048],
        out_strides=[32],
        out_channels=[_dim_],
        bn_kwargs=dict(eps=1e-5, momentum=0.1),
    ),
    view_transformer=dict(
        type="BevFormerViewTransformer",
        bev_h=bev_h_,
        bev_w=bev_w_,
        pc_range=point_cloud_range,
        num_points_in_pillar=4,
        embed_dims=_dim_,
        queue_length=3,
        in_indices=(-1,),
        max_camoverlap_num=2,
        virtual_bev_h=20,
        virtual_bev_w=32,
        use_lidar2img=False,
        positional_encoding=dict(
            type="LearnedPositionalEncoding",
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
        ),
        encoder=dict(
            type="BEVFormerEncoder",
            num_layers=3,
            return_intermediate=False,
            bev_h=bev_h_,
            bev_w=bev_w_,
            embed_dims=_dim_,
            encoder_layer=dict(
                type="BEVFormerEncoderLayer",
                selfattention=dict(
                    type="HorizonTemporalSelfAttention",
                    embed_dims=_dim_,
                    num_levels=1,
                    grid_align_num=100,
                    reduce_align_num=8,
                    feats_size=[[bev_w_, bev_h_]],
                ),
                crossattention=dict(
                    type="HorizonSpatialCrossAttention",
                    max_camoverlap_num=2,
                    bev_h=bev_h_,
                    bev_w=bev_w_,
                    deformable_attention=dict(
                        type="HorizonMultiScaleDeformableAttention3D",
                        embed_dims=_dim_,
                        num_points=8,
                        num_levels=_num_levels_,
                        grid_align_num=20,
                        feats_size=[[25, 15]],
                    ),
                    embed_dims=_dim_,
                ),
                dropout=0.1,
            ),
        ),
    ),
    bev_decoders=[
        dict(
            type="BEVFormerDetDecoder",
            bev_h=bev_h_,
            bev_w=bev_w_,
            num_query=900,
            embed_dims=_dim_,
            pc_range=point_cloud_range,
            decoder=dict(
                type="DetectionTransformerDecoder",
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
                type="BevFormerCriterion",
                bbox_key="ego_bboxes_labels",
                assigner=dict(
                    type="BevFormerHungarianAssigner3D",
                    cls_cost=dict(type="FocalLossCost", weight=2.0),
                    reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
                ),
                loss_cls=dict(
                    type="FocalLoss",
                    loss_name="cls",
                    num_classes=num_classes + 1,
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
            ),
            post_process=dict(
                type="BevFormerProcess",
                post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                pc_range=point_cloud_range,
                max_num=300,
                num_classes=10,
            ),
        ),
    ],
)
test_model = copy.deepcopy(model)

test_model["view_transformer"]["queue_length"] = 1

deploy_model = copy.deepcopy(model)

deploy_model["view_transformer"]["queue_length"] = 1
deploy_model["view_transformer"]["is_compile"] = True
# deploy_model["is_compile"] = True
deploy_model["bev_decoders"][0]["is_compile"] = True
deploy_model["bev_decoders"][0].pop("criterion")
deploy_model["bev_decoders"][0].pop("post_process")


hbir_deploy_model = copy.deepcopy(test_model)
hbir_deploy_model["view_transformer"]["queue_length"] = 1

from qconfig import qat_predict_model_convert_pipeline  # noqa: E402

hbir_infer_model = dict(
    type="BevFormerIrInfer",
    deploy_model=hbir_deploy_model,
    model_convert_pipeline=qat_predict_model_convert_pipeline,
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized_infer.bc"),
    ),
)

single_hbir_infer_model = copy.deepcopy(hbir_infer_model)
single_hbir_infer_model["deploy_model"]["bev_decoders"][0]["post_process"][
    "score_threshold"
] = 0.1

hbm_infer_model = copy.deepcopy(single_hbir_infer_model)
hbm_infer_model["is_nv12_type"] = True
hbm_infer_model["ir_model"] = dict(
    type="HbmModule",
    model_path=os.path.join(ckpt_dir, "compile/model.hbm"),
    host="10.103.43.99",
    return_tensor=True,
)
