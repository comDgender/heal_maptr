"""Model definitions for BevFormer HENet camera-only det+occ pretrain."""

import copy

from common import (
    _dim_,
    _num_levels_,
    _pos_dim_,
    bev_h_,
    bev_w_,
    camera_input,
    lidar_input,
    num_classes,
    num_classes_occ,
    point_cloud_range,
    use_bev_head,
    use_lidar2img,
    use_occ_head,
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
        max_camoverlap_num=2,
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
                type="SingleBEVFormerEncoderLayer",
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
                    max_camoverlap_num=2,
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

bev_head = dict(
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
                grid_align_num=4,
                feats_size=[[bev_w_, bev_h_]],
            ),
            dropout=0.1,
        ),
    ),
    criterion=dict(
        type="BevFormerCriterion",
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
        bbox_key="lidar_bboxes_labels",
    ),
    post_process=dict(
        type="BevFormerProcess",
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        pc_range=point_cloud_range,
        max_num=300,
        num_classes=10,
    ),
)

occ_head = dict(
    type="BevformerOccDetDecoder",
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
        in_dim=256,
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
        loss_weight=6.0,
    ),
)

bev_decoders = []
if use_bev_head:
    bev_decoders.append(bev_head)
if use_occ_head:
    bev_decoders.append(occ_head)

model = dict(
    type="BevFusion",
    lidar_network=None,
    camera_network=camera_network,
    bev_decoders=bev_decoders,
    fuse_module=None,
    bev_h=bev_h_,
    bev_w=bev_w_,
)

test_model = copy.deepcopy(model)
