import copy
import os

import numpy as np
import torch
from common import ckpt_dir, class_names

example_num_points = 150000


def build_example_inputs(num_points=None):
    n = int(example_num_points if num_points is None else num_points)
    return dict(points=[torch.randn(n, 4)])


example_inputs = build_example_inputs()

# PointPillars settings
norm_cfg = None

# Voxelization cfg
pc_range = [0, -39.68, -3, 69.12, 39.68, 1]
voxel_size = [0.16, 0.16, 4.0]
max_points_in_voxel = 100
max_voxels_num = 12000


def get_feature_map_size(point_cloud_range, voxel_size):
    point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
    voxel_size = np.array(voxel_size, dtype=np.float32)
    grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
    grid_size = np.round(grid_size).astype(np.int64)
    return grid_size


model = dict(
    type="PointPillarsDetector",
    feature_map_shape=get_feature_map_size(pc_range, voxel_size),
    pre_process=dict(
        type="PointPillarsPreProcess",
        pc_range=pc_range,
        voxel_size=voxel_size,
        max_voxels_num=max_voxels_num,
        max_points_in_voxel=max_points_in_voxel,
    ),
    reader=dict(
        type="PillarFeatureNet",
        num_input_features=4,
        num_filters=(64,),
        with_distance=False,
        pool_size=(1, max_points_in_voxel),
        voxel_size=voxel_size,
        pc_range=pc_range,
        bn_kwargs=norm_cfg,
        quantize=True,
        use_4dim=True,
        use_conv=True,
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
        up_layer_strides=[1, 2, 4],
        up_layer_channels=[128, 128, 128],
        bn_kwargs=norm_cfg,
        quantize=True,
    ),
    head=dict(
        type="PointPillarsHead",
        num_classes=len(class_names),
        in_channels=sum([128, 128, 128]),
        use_direction_classifier=True,
    ),
    anchor_generator=dict(
        type="Anchor3DGeneratorStride",
        anchor_sizes=[[1.6, 3.9, 1.56]],  # noqa B006
        anchor_strides=[[0.32, 0.32, 0.0]],  # noqa B006
        anchor_offsets=[[0.16, -39.52, -1.78]],  # noqa B006
        rotations=[[0, 1.57]],  # noqa B006
        class_names=class_names,
        match_thresholds=[0.6],
        unmatch_thresholds=[0.45],
    ),
    targets=dict(
        type="LidarTargetAssigner",
        box_coder=dict(
            type="GroundBox3dCoder",
            n_dim=7,
        ),
        class_names=class_names,
        positive_fraction=-1,
    ),
    loss=dict(
        type="PointPillarsLoss",
        num_classes=len(class_names),
        loss_cls=dict(
            type="FocalLoss",
            alpha=0.25,
            gamma=2.0,
            from_logits=False,
            reduction="none",
            loss_weight=1.0,
        ),
        loss_bbox=dict(
            type="SmoothL1Loss",
            beta=1 / 9.0,
            reduction="none",
            loss_weight=2.0,
        ),
        loss_dir=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            reduction="none",
            loss_weight=0.2,
        ),
    ),
    postprocess=dict(
        type="PointPillarsPostProcess",
        num_classes=len(class_names),
        box_coder=dict(
            type="GroundBox3dCoder",
            n_dim=7,
        ),
        use_direction_classifier=True,
        num_direction_bins=2,
        # test_cfg
        use_rotate_nms=False,
        nms_pre_max_size=1000,
        nms_post_max_size=300,
        nms_iou_threshold=0.5,
        score_threshold=0.4,
        post_center_limit_range=[0, -39.68, -5, 69.12, 39.68, 5],
        max_per_img=100,
    ),
)

deploy_model = dict(
    type="PointPillarsDetector",
    feature_map_shape=get_feature_map_size(pc_range, voxel_size),
    is_deploy=True,
    pre_process=dict(
        type="PointPillarsPreProcess",
        pc_range=pc_range,
        voxel_size=voxel_size,
        max_voxels_num=max_voxels_num,
        max_points_in_voxel=max_points_in_voxel,
    ),
    reader=dict(
        type="PillarFeatureNet",
        num_input_features=4,
        num_filters=(64,),
        with_distance=False,
        pool_size=(1, max_points_in_voxel),
        voxel_size=voxel_size,
        pc_range=pc_range,
        bn_kwargs=norm_cfg,
        quantize=True,
        use_4dim=True,
        use_conv=True,
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
        up_layer_strides=[1, 2, 4],
        up_layer_channels=[128, 128, 128],
        bn_kwargs=norm_cfg,
        quantize=True,
    ),
    head=dict(
        type="PointPillarsHead",
        num_classes=len(class_names),
        in_channels=sum([128, 128, 128]),
        use_direction_classifier=True,
    ),
)


hbir_model = dict(
    type="PointPillarsDetectorIrInfer",
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized_infer.bc"),
    ),
    anchor_generator=dict(
        type="Anchor3DGeneratorStride",
        anchor_sizes=[[1.6, 3.9, 1.56]],  # noqa B006
        anchor_strides=[[0.32, 0.32, 0.0]],  # noqa B006
        anchor_offsets=[[0.16, -39.52, -1.78]],  # noqa B006
        rotations=[[0, 1.57]],  # noqa B006
        class_names=class_names,
        match_thresholds=[0.6],
        unmatch_thresholds=[0.45],
    ),
    postprocess=dict(
        type="PointPillarsPostProcess",
        num_classes=len(class_names),
        box_coder=dict(
            type="GroundBox3dCoder",
            n_dim=7,
        ),
        use_direction_classifier=True,
        num_direction_bins=2,
        # test_cfg
        use_rotate_nms=False,
        nms_pre_max_size=1000,
        nms_post_max_size=300,
        nms_iou_threshold=0.5,
        score_threshold=0.4,
        post_center_limit_range=[0, -39.68, -5, 69.12, 39.68, 5],
        max_per_img=100,
    ),
)

hbm_model = dict(
    type="PointPillarsDetectorIrInfer",
    ir_model=dict(
        type="HbmModule",
        model_path=os.path.join(ckpt_dir, "compile/model.hbm"),
        host="10.103.43.99",
    ),
    hbm_infer=True,
    anchor_generator=dict(
        type="Anchor3DGeneratorStride",
        anchor_sizes=[[1.6, 3.9, 1.56]],  # noqa B006
        anchor_strides=[[0.32, 0.32, 0.0]],  # noqa B006
        anchor_offsets=[[0.16, -39.52, -1.78]],  # noqa B006
        rotations=[[0, 1.57]],  # noqa B006
        class_names=class_names,
        match_thresholds=[0.6],
        unmatch_thresholds=[0.45],
    ),
    postprocess=dict(
        type="PointPillarsPostProcess",
        num_classes=len(class_names),
        box_coder=dict(
            type="GroundBox3dCoder",
            n_dim=7,
        ),
        use_direction_classifier=True,
        num_direction_bins=2,
        # test_cfg
        use_rotate_nms=False,
        nms_pre_max_size=1000,
        nms_post_max_size=300,
        nms_iou_threshold=0.5,
        score_threshold=0.4,
        post_center_limit_range=[0, -39.68, -5, 69.12, 39.68, 5],
        max_per_img=100,
    ),
)

deploy_infer_model = copy.deepcopy(hbir_model)
deploy_infer_model["ir_model"] = deploy_model
