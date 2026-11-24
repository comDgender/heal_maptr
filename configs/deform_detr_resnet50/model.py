import os

import torch.nn as nn
from common import (
    as_two_stage,
    ckpt_dir,
    num_classes,
    num_queries,
    with_box_refine,
)

model = dict(
    type="DeformableDETR",
    backbone=dict(
        type="ResNet50",
        bn_kwargs={},
        num_classes=num_classes,
        include_top=False,
    ),
    neck=dict(
        type="ChannelMapperNeck",
        in_channels=[512, 1024, 2048],
        out_indices=[2, 3, 4],
        out_channel=256,
        kernel_size=1,
        extra_convs=1,
        norm_layer=dict(
            type=nn.BatchNorm2d,
            num_features=256,
        ),
    ),
    position_embedding=dict(
        type="PositionEmbeddingSine",
        num_pos_feats=128,
        temperature=10000,
        normalize=True,
        offset=-0.5,
    ),
    transformer=dict(
        type="DeformableDetrTransformer",
        encoder=dict(
            type="DeformableDetrTransformerEncoder",
            embed_dim=256,
            num_heads=8,
            feedforward_dim=1024,
            attn_dropout=0.1,
            ffn_dropout=0.1,
            num_layers=6,
            post_norm=False,
            num_feature_levels=4,
        ),
        decoder=dict(
            type="DeformableDetrTransformerDecoder",
            embed_dim=256,
            num_heads=8,
            feedforward_dim=1024,
            attn_dropout=0.1,
            ffn_dropout=0.1,
            num_layers=6,
            return_intermediate=True,
            num_feature_levels=4,
            as_two_stage=as_two_stage,
            with_box_refine=with_box_refine,
        ),
        num_feature_levels=4,
        two_stage_num_proposals=num_queries,
        as_two_stage=as_two_stage,
    ),
    embed_dim=256,
    num_classes=num_classes,
    num_queries=num_queries,
    aux_loss=True,
    post_process=dict(
        type="DeformDetrPostProcess", select_box_nums_for_evaluation=300
    ),
    criterion=dict(
        type="DeformableCriterion",
        num_classes=num_classes,
        matcher=dict(
            type="HungarianMatcher",
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            use_focal=True,
            alpha=0.25,
            gamma=2.0,
        ),
        weight_dict={
            "loss_class": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
        },
        loss_class_type="focal_loss",
        alpha=0.25,
        gamma=2.0,
    ),
    as_two_stage=as_two_stage,
    with_box_refine=with_box_refine,
)

deploy_model = dict(
    type="DeformableDETR",
    backbone=dict(
        type="ResNet50",
        bn_kwargs={},
        num_classes=num_classes,
        include_top=False,
    ),
    neck=dict(
        type="ChannelMapperNeck",
        in_channels=[512, 1024, 2048],
        out_indices=[2, 3, 4],
        out_channel=256,
        kernel_size=1,
        extra_convs=1,
        norm_layer=nn.BatchNorm2d(256),
    ),
    position_embedding=dict(
        type="PositionEmbeddingSine",
        num_pos_feats=128,
        temperature=10000,
        normalize=True,
        offset=-0.5,
    ),
    transformer=dict(
        type="DeformableDetrTransformer",
        encoder=dict(
            type="DeformableDetrTransformerEncoder",
            embed_dim=256,
            num_heads=8,
            feedforward_dim=1024,
            attn_dropout=0.1,
            ffn_dropout=0.1,
            num_layers=6,
            post_norm=False,
            num_feature_levels=4,
        ),
        decoder=dict(
            type="DeformableDetrTransformerDecoder",
            embed_dim=256,
            num_heads=8,
            feedforward_dim=1024,
            attn_dropout=0.1,
            ffn_dropout=0.1,
            num_layers=6,
            return_intermediate=True,
            num_feature_levels=4,
            as_two_stage=as_two_stage,
            with_box_refine=with_box_refine,
        ),
        num_feature_levels=4,
        two_stage_num_proposals=num_queries,
        as_two_stage=as_two_stage,
    ),
    embed_dim=256,
    num_classes=num_classes,
    num_queries=num_queries,
    as_two_stage=as_two_stage,
    with_box_refine=with_box_refine,
)

hbir_model = dict(
    type="DeformDetrIrInfer",
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized_infer.bc"),
    ),
    post_process=dict(
        type="DeformDetrPostProcess",
    ),
)

hbm_model = dict(
    type="DeformDetrIrInfer",
    use_nv12=True,
    ir_model=dict(
        type="HbmModule",
        model_path=os.path.join(ckpt_dir, "compile/model.hbm"),
        host="10.103.43.99",
    ),
    post_process=dict(
        type="DeformDetrPostProcess",
    ),
)
