import os

from common import (
    bev_range,
    bn_kwargs,
    ckpt_dir,
    grid_quant_scales,
    num_classes,
    num_levels,
    num_query,
    query_align,
    vt_input_hw,
)
from qconfig import real_deploy_model_convert_pipeline

model = dict(
    type="Detr3d",
    backbone=dict(
        type="efficientnet",
        bn_kwargs=bn_kwargs,
        model_type="b3",
        num_classes=1000,
        include_top=False,
        activation="relu",
        use_se_block=False,
    ),
    neck=dict(
        type="BiFPN",
        in_strides=[2, 4, 8, 16, 32],
        out_strides=[8, 16, 32, 64, 128],
        stride2channels=dict({2: 24, 4: 32, 8: 48, 16: 136, 32: 384}),
        out_channels=256,
        num_outs=5,
        stack=3,
        start_level=2,
        end_level=-1,
        fpn_name="bifpn_sum",
        upsample_type="function",
        use_fx=True,
    ),
    head=dict(
        type="Detr3dHead",
        num_query=num_query,
        query_align=query_align,
        embed_dims=256,
        num_cls_fcs=2,
        num_reg_fcs=2,
        num_levels=num_levels,
        reg_out_channels=10,
        cls_out_channels=num_classes,
        bev_range=bev_range,
        transformer=dict(
            type="Detr3dTransformer",
            decoder=dict(
                type="Detr3dDecoder",
                num_layer=6,
                num_heads=8,
                embed_dims=256,
                dropout=0.1,
                num_levels=4,
                num_views=6,
                num_points=1,
            ),
            embed_dims=256,
            num_views=6,
            grid_quant_scales=grid_quant_scales,
        ),
        int8_output=False,
        dequant_output=True,
    ),
    target=dict(
        type="Detr3dTarget",
        bev_range=bev_range,
        cls_cost=dict(
            type="FocalLossCost",
            alpha=0.25,
            gamma=2.0,
            weight=2.0,
        ),
        reg_cost=dict(
            type="BBox3DL1Cost",
            weight=0.25,
        ),
    ),
    loss_cls=dict(
        type="FocalLoss",
        loss_name="cls",
        num_classes=num_classes + 1,
        loss_weight=2.0,
        alpha=0.25,
        gamma=2.0,
    ),
    loss_reg=dict(
        type="L1Loss",
        loss_weight=0.25,
    ),
    post_process=dict(
        type="Detr3dPostProcess", max_num=300, bev_range=bev_range
    ),
)

deploy_model = dict(
    type="Detr3d",
    compile_model=True,
    backbone=dict(
        type="efficientnet",
        bn_kwargs=bn_kwargs,
        model_type="b3",
        num_classes=1000,
        include_top=False,
        activation="relu",
        use_se_block=False,
    ),
    neck=dict(
        type="BiFPN",
        in_strides=[2, 4, 8, 16, 32],
        out_strides=[8, 16, 32, 64, 128],
        stride2channels=dict({2: 24, 4: 32, 8: 48, 16: 136, 32: 384}),
        out_channels=256,
        num_outs=5,
        stack=3,
        start_level=2,
        end_level=-1,
        fpn_name="bifpn_sum",
        upsample_type="function",
        use_fx=True,
    ),
    head=dict(
        type="Detr3dHead",
        num_query=num_query,
        query_align=query_align,
        embed_dims=256,
        num_cls_fcs=2,
        num_reg_fcs=2,
        num_levels=num_levels,
        reg_out_channels=10,
        cls_out_channels=num_classes,
        bev_range=bev_range,
        transformer=dict(
            type="Detr3dTransformer",
            decoder=dict(
                type="Detr3dDecoder",
                num_layer=6,
                num_heads=8,
                embed_dims=256,
                dropout=0.1,
                num_levels=4,
                num_views=6,
                num_points=1,
            ),
            embed_dims=256,
            num_views=6,
            grid_quant_scales=grid_quant_scales,
        ),
        int8_output=False,
        dequant_output=True,
    ),
)

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

hbir_deploy_model = dict(
    type="Detr3d",
    compile_model=True,
    backbone=dict(
        type="efficientnet",
        bn_kwargs=bn_kwargs,
        model_type="b3",
        num_classes=1000,
        include_top=False,
        activation="relu",
        use_se_block=False,
    ),
    neck=dict(
        type="BiFPN",
        in_strides=[2, 4, 8, 16, 32],
        out_strides=[8, 16, 32, 64, 128],
        stride2channels=dict({2: 24, 4: 32, 8: 48, 16: 136, 32: 384}),
        out_channels=256,
        num_outs=5,
        stack=3,
        start_level=2,
        end_level=-1,
        fpn_name="bifpn_sum",
        upsample_type="function",
        use_fx=True,
    ),
    head=dict(
        type="Detr3dHead",
        num_query=num_query,
        query_align=query_align,
        embed_dims=256,
        num_cls_fcs=2,
        num_reg_fcs=2,
        num_levels=num_levels,
        reg_out_channels=10,
        cls_out_channels=num_classes,
        bev_range=bev_range,
        transformer=dict(
            type="Detr3dTransformer",
            decoder=dict(
                type="Detr3dDecoder",
                num_layer=6,
                num_heads=8,
                embed_dims=256,
                dropout=0.1,
                num_levels=4,
                num_views=6,
                num_points=1,
            ),
            embed_dims=256,
            num_views=6,
            grid_quant_scales=grid_quant_scales,
        ),
        int8_output=False,
        dequant_output=True,
    ),
)

hbir_model = dict(
    type="Detr3dIrInfer",
    deploy_model=hbir_deploy_model,
    vt_input_hw=vt_input_hw,
    model_convert_pipeline=real_deploy_model_convert_pipeline,
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized_infer.bc"),
    ),
    post_process=dict(
        type="Detr3dPostProcess", max_num=300, bev_range=bev_range
    ),
    use_nv12=False,
)

hbm_model = dict(
    type="Detr3dIrInfer",
    deploy_model=hbir_deploy_model,
    vt_input_hw=vt_input_hw,
    model_convert_pipeline=real_deploy_model_convert_pipeline,
    ir_model=dict(
        type="HbmModule",
        host="10.103.43.99",
        model_path=os.path.join(ckpt_dir, "compile/model.hbm"),
    ),
    post_process=dict(
        type="Detr3dPostProcess", max_num=300, bev_range=bev_range
    ),
    use_nv12=True,
    keep_last_img=False,
)
