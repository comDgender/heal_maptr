import os

import numpy as np
import torch
import torch.nn as nn
from common import (
    alpha,
    bn_kwargs,
    ckpt_dir,
    data_shape,
    loss_weights,
    num_classes,
    output_with_bn,
    train_scales,
)

model = dict(
    type="Segmentor",
    backbone=dict(
        type="MobileNetV1",
        num_classes=-1,
        bn_kwargs=bn_kwargs,
        alpha=alpha,
        dw_with_relu=True,
        include_top=False,
        flat_output=False,
    ),
    neck=dict(
        type="DwUnet",
        base_channels=int(32 * alpha),
        bn_kwargs=bn_kwargs,
        act_type=nn.ReLU,
        use_deconv=False,
        dw_with_act=True,
        output_scales=train_scales,
    ),
    head=dict(
        type="SegHead",
        num_classes=num_classes,
        in_strides=train_scales,
        out_strides=train_scales,
        stride2channels={
            stride: int(stride * 32 * alpha) for stride in train_scales
        },
        feat_channels=tuple(np.array(train_scales) * int(32 * alpha)),
        stacked_convs=0,
        argmax_output=False,
        dequant_output=True,
        int8_output=True,
        upscale=False,
        output_with_bn=output_with_bn,
        bn_kwargs=bn_kwargs,
    ),
    losses=dict(
        type="SoftmaxFocalLoss",
        loss_name="Focal",
        num_classes=num_classes,
        weight=loss_weights,
        reduction="mean",
    ),
)

deploy_model = dict(
    type="Segmentor",
    backbone=dict(
        type="MobileNetV1",
        num_classes=-1,
        bn_kwargs=bn_kwargs,
        alpha=alpha,
        dw_with_relu=True,
        include_top=False,
        flat_output=False,
    ),
    neck=dict(
        type="DwUnet",
        base_channels=int(32 * alpha),
        bn_kwargs=bn_kwargs,
        act_type=nn.ReLU,
        use_deconv=False,
        dw_with_act=True,
        output_scales=train_scales,
    ),
    head=dict(
        type="SegHead",
        num_classes=num_classes,
        in_strides=train_scales,
        out_strides=train_scales,
        stride2channels={
            stride: int(stride * 32 * alpha) for stride in train_scales
        },
        feat_channels=tuple(np.array(train_scales) * int(32 * alpha)),
        stacked_convs=0,
        argmax_output=True,
        dequant_output=False,
        int8_output=True,
        upscale=False,
        output_with_bn=output_with_bn,
        bn_kwargs=bn_kwargs,
        only_export_first=True,
    ),
    losses=None,
)

deploy_inputs = {"img": torch.randn((1,) + data_shape)}

hbir_model = dict(
    type="SegmentorIrInfer",
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized_infer.bc"),
    ),
)

hbm_model = dict(
    type="SegmentorIrInfer",
    use_nv12=True,
    ir_model=dict(
        type="HbmModule",
        model_path=os.path.join(ckpt_dir, "compile/model.hbm"),
        host="10.103.43.99",
    ),
)
