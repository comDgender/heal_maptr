import os

from common import bn_kwargs, ckpt_dir, num_classes

model = dict(
    type="FCOS",
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
        out_channels=120,
        num_outs=5,
        stack=6,
        start_level=2,
        end_level=-1,
        fpn_name="bifpn_sum",
        upsample_type="function",
        use_fx=True,
    ),
    head=dict(
        type="FCOSHead",
        num_classes=num_classes,
        in_strides=[8, 16, 32, 64, 128],
        out_strides=[8, 16, 32, 64, 128],
        stride2channels=dict({8: 120, 16: 120, 32: 120, 64: 120, 128: 120}),
        upscale_bbox_pred=False,
        feat_channels=120,
        stacked_convs=4,
        dequant_output=True,
        bbox_relu=False,
    ),
    targets=dict(
        type="DynamicFcosTarget",
        strides=[8, 16, 32, 64, 128],
        cls_out_channels=80,
        background_label=80,
        topK=10,
        loss_cls=dict(
            type="FocalLoss",
            loss_name="cls",
            num_classes=80 + 1,
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            reduction="none",
        ),
        loss_reg=dict(
            type="GIoULoss", loss_name="reg", loss_weight=3.0, reduction="none"
        ),
        bbox_relu=True,
    ),
    post_process=dict(
        type="FCOSDecoder",
        num_classes=80,
        strides=[8, 16, 32, 64, 128],
        nms_use_centerness=True,
        nms_sqrt=True,
        transforms=[dict(type="Resize")],
        inverse_transform_key=["scale_factor"],
        test_cfg=dict(
            score_thr=0.05,
            nms_pre=1000,
            nms=dict(name="nms", iou_threshold=0.6, max_per_img=100),
        ),
        upscale_bbox_pred=True,
        bbox_relu=True,
    ),
    loss_cls=dict(
        type="FocalLoss",
        loss_name="cls",
        num_classes=80 + 1,
        alpha=0.25,
        gamma=2.0,
        loss_weight=1.0,
    ),
    loss_centerness=dict(
        type="CrossEntropyLoss", loss_name="centerness", use_sigmoid=True
    ),
    loss_reg=dict(
        type="GIoULoss",
        loss_name="reg",
        loss_weight=1.0,
    ),
)

deploy_model = dict(
    type="FCOS",
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
        out_channels=120,
        num_outs=5,
        stack=6,
        start_level=2,
        end_level=-1,
        fpn_name="bifpn_sum",
        upsample_type="function",
        use_fx=True,
    ),
    head=dict(
        type="FCOSHead",
        num_classes=num_classes,
        in_strides=[8, 16, 32, 64, 128],
        out_strides=[8, 16, 32, 64, 128],
        stride2channels=dict({8: 120, 16: 120, 32: 120, 64: 120, 128: 120}),
        upscale_bbox_pred=False,
        feat_channels=120,
        stacked_convs=4,
        dequant_output=True,
        bbox_relu=False,
    ),
)

hbir_model = dict(
    type="FCOSIrInfer",
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized_infer.bc"),
    ),
    post_process=model["post_process"],
)

hbm_model = dict(
    type="FCOSIrInfer",
    is_nv12_type=True,
    ir_model=dict(
        type="HbmModule",
        model_path=os.path.join(ckpt_dir, "compile/model.hbm"),
        host="10.103.43.99",
    ),
    post_process=model["post_process"],
)
