"""CenterPoint PointPillars model definitions."""

import copy
import os

import numpy as np
import torch
from common import (
    ckpt_dir,
    common_heads,
    max_num_points,
    max_voxels,
    norm_cfg,
    point_cloud_range,
    tasks,
    voxel_size,
    with_velocity,
)

example_num_voxels = 40000


def build_example_inputs(num_voxels=None):
    n = int(example_num_voxels if num_voxels is None else num_voxels)
    return dict(
        features=torch.randn((1, 5, 20, n), dtype=torch.float32),
        coors=torch.zeros([n, 4]).int(),
    )


deploy_inputs = build_example_inputs()


def get_feature_map_size(point_cloud_range, voxel_size):
    point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
    voxel_size = np.array(voxel_size, dtype=np.float32)
    grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
    grid_size = np.round(grid_size).astype(np.int64)
    return grid_size


feature_map_shape = get_feature_map_size(point_cloud_range, voxel_size)

_postprocess_cfg = dict(
    type="CenterPointPostProcess",
    tasks=tasks,
    norm_bbox=True,
    bbox_coder=dict(
        type="CenterPointBBoxCoder",
        pc_range=point_cloud_range[:2],
        post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        max_num=100,
        score_threshold=0.1,
        out_size_factor=4,
        voxel_size=voxel_size[:2],
    ),
    max_pool_nms=False,
    score_threshold=0.1,
    post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    min_radius=[4, 12, 10, 1, 0.85, 0.175],
    out_size_factor=4,
    nms_type="rotate",
    pre_max_size=1000,
    post_max_size=83,
    nms_thr=0.2,
    box_size=9,
)

_pre_process_cfg = dict(
    type="CenterPointPreProcess",
    pc_range=point_cloud_range,
    voxel_size=voxel_size,
    max_voxels_num=max_voxels,
    max_points_in_voxel=max_num_points,
    norm_range=[-51.2, -51.2, -5.0, 0.0, 51.2, 51.2, 3.0, 255.0],
    norm_dims=[0, 1, 2, 3],
)

_reader_cfg = dict(
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
)

_backbone_cfg = dict(
    type="PointPillarScatter",
    num_input_features=64,
    use_horizon_pillar_scatter=True,
    quantize=True,
)

_neck_cfg = dict(
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
)

_head_cfg = dict(
    type="CenterPointHead",
    in_channels=sum([128, 128, 128]),
    tasks=tasks,
    share_conv_channels=64,
    share_conv_num=1,
    common_heads=common_heads,
    head_conv_channels=64,
    init_bias=-2.19,
    final_kernel=3,
)

model = dict(
    type="CenterPointDetector",
    feature_map_shape=feature_map_shape,
    pre_process=_pre_process_cfg,
    reader=_reader_cfg,
    backbone=_backbone_cfg,
    neck=_neck_cfg,
    head=_head_cfg,
    targets=dict(
        type="CenterPointLidarTarget",
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        tasks=tasks,
        dense_reg=1,
        max_objs=500,
        gaussian_overlap=0.1,
        min_radius=2,
        out_size_factor=4,
        norm_bbox=True,
        with_velocity=with_velocity,
    ),
    loss=dict(
        type="CenterPointLoss",
        loss_cls=dict(type="GaussianFocalLoss", loss_weight=1.0),
        loss_bbox=dict(
            type="L1Loss",
            reduction="mean",
            loss_weight=0.25,
        ),
        with_velocity=with_velocity,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
    ),
    postprocess=_postprocess_cfg,
)

deploy_model = dict(
    type="CenterPointDetector",
    feature_map_shape=feature_map_shape,
    is_deploy=True,
    reader=_reader_cfg,
    backbone=_backbone_cfg,
    neck=_neck_cfg,
    head=_head_cfg,
)

hbir_model = dict(
    type="CenterPointDetectorIrInfer",
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized_infer.bc"),
    ),
    # Deepcopy: smoke may shrink model.pre_process.max_voxels_num for OOM;
    # HBIR graph is frozen at export shape (1,5,20,40000) and must keep
    # the original preprocess size.
    pre_process=copy.deepcopy(_pre_process_cfg),
    feature_map_shape=feature_map_shape,
    tasks=tasks,
    postprocess=copy.deepcopy(_postprocess_cfg),
)

hbm_model = copy.deepcopy(hbir_model)
hbm_model["transpose_output"] = True
hbm_model["ir_model"] = dict(
    type="HbmModule",
    model_path=os.path.join(ckpt_dir, "compile/model.hbm"),
    host="10.103.43.99",
    return_tensor=True,
)

example_inputs = deploy_inputs
