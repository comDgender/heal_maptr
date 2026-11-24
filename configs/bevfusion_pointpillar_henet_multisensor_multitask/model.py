"""Model definitions for BevFusion PointPillars + HENet multisensor multitask."""  # noqa: E501

import copy
import os

import numpy as np
import torch
from common import (
    _dim_,
    _num_levels_,
    _pos_dim_,
    bev_h_,
    bev_w_,
    camera_input,
    ckpt_dir,
    lidar_input,
    max_num_points,
    max_voxels,
    norm_cfg,
    num_classes,
    num_classes_occ,
    point_cloud_range,
    use_bev_head,
    use_lidar2img,
    use_occ_head,
    voxel_size,
)


def get_feature_map_size(point_cloud_range, voxel_size):
    point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
    voxel_size = np.array(voxel_size, dtype=np.float32)
    grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
    return np.round(grid_size).astype(np.int64)


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
        bn_kwargs=norm_cfg,
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
        bn_kwargs=norm_cfg,
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
    lidar_network=lidar_network if lidar_input else None,
    camera_network=camera_network if camera_input else None,
    bev_decoders=bev_decoders,
    fuse_module=dict(type="BevFuseModule", input_c=384 + _dim_, fuse_c=_dim_),
    bev_h=bev_h_,
    bev_w=bev_w_,
)

deploy_model = copy.deepcopy(model)
deploy_model["lidar_network"].pop("pre_process")
deploy_model["camera_network"]["view_transformer"]["is_compile"] = True
deploy_model["bev_decoders"][0]["is_compile"] = True
deploy_model["bev_decoders"][0].pop("criterion")
deploy_model["bev_decoders"][0].pop("post_process")
deploy_model["bev_decoders"][1]["is_compile"] = True
deploy_model["bev_decoders"][1].pop("loss_occ")

deploy_inputs = dict(
    features=torch.randn((1, 5, 20, 40000), dtype=torch.float32),
    coors=torch.zeros([40000, 4]).int(),
    img=torch.randn((6, 3, 512, 960)),
    queries_rebatch_grid=torch.randn((6, 64, 80, 2)),
    restore_bev_grid=torch.randn((1, 256, 128, 2)),
    reference_points_rebatch=torch.randn((6, 5120, 2, 2)),
    bev_pillar_counts=torch.randn((1, 16384, 1)),
)

example_inputs = deploy_inputs

hbir_deploy_model = copy.deepcopy(model)

hbir_infer_model = dict(
    type="BevFusionHbirInfer",
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized_infer.bc"),
    ),
    deploy_model=hbir_deploy_model,
)

# On-board BPU runtime after compile (``compile/model.hbm``).
# ``host`` is the BPU server address; override with your board's IP.
# BevFusionHbirInfer accepts use_nv12 (stored, though not used in
# process_input) and ir_model (HbmModule here instead of HbirModule).
hbm_infer_model = copy.deepcopy(hbir_infer_model)
hbm_infer_model["use_nv12"] = True
hbm_infer_model["ir_model"] = dict(
    type="HbmModule",
    model_path=os.path.join(ckpt_dir, "compile", "model.hbm"),
    host="10.103.43.99",
    return_tensor=True,
)

# Exported graph output names, in deploy_model (is_compile=True) output
# order: [bev_embed, outputs_classes, reference_out, outputs_coords, occ_preds]
# (det decoder 4-tuple + occ 1-tensor). BevFusionHbirInfer consumes outputs by
# order (list(outputs.values())), so names are for export readability, but
# passing them to hbir_exporter keeps the exported graph self-describing.
output_names = [
    "bev_embed",
    "outputs_classes",
    "reference_out",
    "outputs_coords",
    "occ_preds",
]

test_model = copy.deepcopy(model)
