import copy
import math
import os

import torch
from common import (
    bev_size,
    ckpt_dir,
    data_shape,
    depth,
    grid_size,
    num_points,
    point_cloud_range,
    voxel_size,
    vt_input_hw,
)

W = int(grid_size[1])
H = int(grid_size[0])


def get_grid_quant_scale(grid_shape, view_shape):
    max_coord = max(*grid_shape, *view_shape)
    coord_bit_num = math.ceil(math.log(max_coord + 1, 2))
    coord_shift = 15 - coord_bit_num
    coord_shift = max(min(coord_shift, 8), 0)
    grid_quant_scale = 1.0 / (1 << coord_shift)
    return grid_quant_scale


view_shape = [data_shape[1] / 16, data_shape[2] / 16]
featview_shape = [view_shape[0] * 6, view_shape[1]]
grid_quant_scale = get_grid_quant_scale(grid_size, featview_shape)

depthview_shape = [6 * depth, view_shape[0] * view_shape[1]]
depth_quant_scale = get_grid_quant_scale(grid_size, depthview_shape)

tasks = [
    dict(
        name="bbox",
        num_class=10,
        class_names=[
            "car",
            "truck",
            "construction_vehicle",
            "bus",
            "trailer",
            "barrier",
            "motorcycle",
            "bicycle",
            "pedestrian",
            "traffic_cone",
        ],
    )
]

model = dict(
    type="Bev",
    bev_feat_index=0,
    bev_upscale=2,
    backbone=dict(
        type="ResNet50",
        num_classes=1000,
        bn_kwargs={},
        include_top=False,
    ),
    neck=dict(
        type="FPN",
        in_strides=[2, 4, 8, 16, 32],
        in_channels=[64, 256, 512, 1024, 2048],
        out_strides=[16, 32],
        out_channels=[1024, 2048],
        fix_out_channel=256,
        bn_kwargs=dict(eps=1e-5, momentum=0.1),
    ),
    view_transformer=dict(
        type="LSSTransformer",
        use_vtv2=True,
        cal_minmax=False,
        in_channels=256,
        feat_channels=256,
        z_range=(-5.0, 3.0, 0.125),
        depth=depth,
        num_points=num_points,
        bev_size=bev_size,
        grid_size=grid_size,
        num_views=6,
        homo_key="lidar2img",
        grid_quant_scale=grid_quant_scale,
        depth_grid_quant_scale=depth_quant_scale,
    ),
    bev_encoder=dict(
        type="BevEncoder",
        backbone=dict(
            type="ResNet50",
            num_classes=1000,
            bn_kwargs={},
            include_top=False,
            input_channels=256,
            quant_input=False,
        ),
        neck=dict(
            type="FPN",
            in_strides=[2, 4, 8, 16, 32],
            in_channels=[256, 256, 512, 1024, 2048],
            out_strides=[8, 16, 32],
            out_channels=[512, 1024, 2048],
            fix_out_channel=256,
            bn_kwargs=dict(eps=1e-5, momentum=0.1),
        ),
    ),
    bev_decoders=[
        dict(
            type="BevDetDecoderV2",
            name="bev_det",
            task_weight=1.0,
            gt_key="lidar_bboxes_labels",
            head=dict(
                type="CenterPointHead",
                in_channels=256,
                tasks=tasks,
                share_conv_channels=64,
                share_conv_num=1,
                common_heads=dict(
                    reg=(2, 2),
                    height=(1, 2),
                    dim=(3, 2),
                    rot=(2, 2),
                    vel=(2, 2),
                ),
                head_conv_channels=64,
                num_heatmap_convs=2,
                final_kernel=3,
            ),
            target=dict(
                type="CenterPointTargetForLidarBBOX",
                grid_size=[1024, 1024, 1],
                voxel_size=voxel_size,
                point_cloud_range=point_cloud_range,
                tasks=tasks,
                dense_reg=1,
                max_objs=500,
                gaussian_overlap=0.1,
                min_radius=2,
                out_size_factor=8,
                norm_bbox=True,
                with_velocity=True,
            ),
            loss=dict(
                type="CenterPointLoss",
                loss_cls=dict(type="GaussianFocalLoss", loss_weight=1.0),
                loss_bbox=dict(
                    type="L1Loss",
                    reduction="mean",
                    loss_weight=0.25,
                ),
                with_velocity=True,
                code_weights=[
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.2,
                    0.2,
                ],
            ),
            decoder=dict(
                type="CenterPointPostProcess",
                tasks=tasks,
                norm_bbox=True,
                bbox_coder=dict(
                    type="CenterPointBBoxCoder",
                    pc_range=point_cloud_range[:2],
                    post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                    max_num=500,
                    score_threshold=0.1,
                    out_size_factor=8,
                    voxel_size=voxel_size[:2],
                ),
                max_pool_nms=False,
                score_threshold=0.1,
                post_center_limit_range=[
                    -61.2,
                    -61.2,
                    -10.0,
                    61.2,
                    61.2,
                    10.0,
                ],
                min_radius=[4, 12, 10, 1, 0.85, 0.175],
                out_size_factor=4,
                nms_type="rotate",
                pre_max_size=1000,
                post_max_size=500,
                nms_thr=0.2,
                box_size=9,
            ),
        ),
    ],
)

deploy_model = dict(
    type="Bev",
    compile_model=True,
    bev_feat_index=0,
    bev_upscale=2,
    backbone=dict(
        type="ResNet50",
        num_classes=1000,
        bn_kwargs={},
        include_top=False,
    ),
    neck=dict(
        type="FPN",
        in_strides=[2, 4, 8, 16, 32],
        in_channels=[64, 256, 512, 1024, 2048],
        out_strides=[16, 32],
        out_channels=[1024, 2048],
        fix_out_channel=256,
        bn_kwargs=dict(eps=1e-5, momentum=0.1),
    ),
    view_transformer=dict(
        type="LSSTransformer",
        use_vtv2=True,
        cal_minmax=False,
        in_channels=256,
        feat_channels=256,
        z_range=(-5.0, 3.0, 0.125),
        depth=depth,
        num_points=num_points,
        bev_size=bev_size,
        grid_size=grid_size,
        num_views=6,
        homo_key="lidar2img",
        grid_quant_scale=grid_quant_scale,
        depth_grid_quant_scale=depth_quant_scale,
    ),
    bev_encoder=dict(
        type="BevEncoder",
        backbone=dict(
            type="ResNet50",
            num_classes=1000,
            bn_kwargs={},
            include_top=False,
            input_channels=256,
            quant_input=False,
        ),
        neck=dict(
            type="FPN",
            in_strides=[2, 4, 8, 16, 32],
            in_channels=[256, 256, 512, 1024, 2048],
            out_strides=[8, 16, 32],
            out_channels=[512, 1024, 2048],
            fix_out_channel=256,
            bn_kwargs=dict(eps=1e-5, momentum=0.1),
        ),
    ),
    bev_decoders=[
        dict(
            type="BevDetDecoderV2",
            name="bev_det",
            task_weight=1.0,
            gt_key="lidar_bboxes_labels",
            head=dict(
                type="CenterPointHead",
                in_channels=256,
                tasks=tasks,
                share_conv_channels=64,
                share_conv_num=1,
                common_heads=dict(
                    reg=(2, 2),
                    height=(1, 2),
                    dim=(3, 2),
                    rot=(2, 2),
                    vel=(2, 2),
                ),
                head_conv_channels=64,
                num_heatmap_convs=2,
                final_kernel=3,
            ),
        )
    ],
)


def get_input_point(num_pts):
    inputs = {"img": torch.randn((6,) + data_shape)}
    inputs["points0"] = torch.randn((num_pts, H, W, 2))
    inputs["points1"] = torch.randn((num_pts, H, W, 2))
    return inputs


deploy_inputs = get_input_point(num_points)


def get_example_inputs(num_pts):
    inputs = {"img": torch.randn((6,) + data_shape)}
    inputs["lidar2img"] = torch.randn((6, 4, 4))
    return inputs


example_inputs = get_example_inputs(num_points)

hbir_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    converters=[
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "float-checkpoint-best.pth.tar"
            ),
        ),
    ],
)

hbir_deploy_model = copy.deepcopy(deploy_model)
hbir_deploy_model["compile_model"] = False

_bev_det_decoder_infer = dict(
    type="BevDetDecoderInfer",
    name="bev_det",
    tasks=tasks,
    task_keys=["reg", "height", "dim", "rot", "vel", "heatmap"],
    decoder=copy.deepcopy(model["bev_decoders"][0]["decoder"]),
)

hbir_model = dict(
    type="BevIrInfer",
    deploy_model=hbir_deploy_model,
    vt_input_hw=vt_input_hw,
    model_convert_pipeline=hbir_model_convert_pipeline,
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized_infer.bc"),
    ),
    bev_decoder_infers=[_bev_det_decoder_infer],
)

hbm_model = copy.deepcopy(hbir_model)
hbm_model["use_nv12"] = True
hbm_model["ir_model"] = dict(
    type="HbmModule",
    model_path=os.path.join(ckpt_dir, "compile/model.hbm"),
    host="10.103.43.99",
)
