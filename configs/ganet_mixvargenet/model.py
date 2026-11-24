import os

from common import attn_ratio, bn_kwargs, ckpt_dir, hid_dim, radius

from heal.models.backbones.mixvargenet import MixVarGENetConfig

model = dict(
    type="GaNet",
    backbone=dict(
        type="MixVarGENet",
        net_config=[
            [
                MixVarGENetConfig(
                    in_channels=32,
                    out_channels=32,
                    head_op="mixvarge_f2",
                    stack_ops=[],
                    stride=1,
                    stack_factor=1,
                    fusion_strides=[],
                    extra_downsample_num=0,
                )
            ],
            [
                MixVarGENetConfig(
                    in_channels=32,
                    out_channels=32,
                    head_op="mixvarge_f4",
                    stack_ops=["mixvarge_f4", "mixvarge_f4"],
                    stride=2,
                    stack_factor=1,
                    fusion_strides=[],
                    extra_downsample_num=0,
                )
            ],
            [
                MixVarGENetConfig(
                    in_channels=32,
                    out_channels=64,
                    head_op="mixvarge_f4",
                    stack_ops=["mixvarge_f4", "mixvarge_f4"],
                    stride=2,
                    stack_factor=1,
                    fusion_strides=[],
                    extra_downsample_num=0,
                )
            ],
            [
                MixVarGENetConfig(
                    in_channels=64,
                    out_channels=96,
                    head_op="mixvarge_f2_gb16",
                    stack_ops=[
                        "mixvarge_f2_gb16",
                        "mixvarge_f2_gb16",
                        "mixvarge_f2_gb16",
                        "mixvarge_f2_gb16",
                        "mixvarge_f2_gb16",
                        "mixvarge_f2_gb16",
                    ],
                    stride=2,
                    stack_factor=1,
                    fusion_strides=[],
                    extra_downsample_num=0,
                )
            ],
            [
                MixVarGENetConfig(
                    in_channels=96,
                    out_channels=160,
                    head_op="mixvarge_f2_gb16",
                    stack_ops=["mixvarge_f2_gb16", "mixvarge_f2_gb16"],
                    stride=2,
                    stack_factor=1,
                    fusion_strides=[],
                    extra_downsample_num=0,
                )
            ],
        ],
        disable_quanti_input=False,
        input_channels=3,
        input_sequence_length=1,
        num_classes=1000,
        bn_kwargs=bn_kwargs,
        include_top=False,
        bias=True,
        output_list=[2, 3, 4],
    ),
    neck=dict(
        type="GaNetNeck",
        fpn_module=dict(
            type="FPN",
            in_strides=[8, 16, 32],
            in_channels=[64, 96, hid_dim],
            out_strides=[8, 16, 32],
            out_channels=[hid_dim, hid_dim, hid_dim],
        ),
        attn_in_channels=[160],
        attn_out_channels=[hid_dim],
        attn_ratios=[attn_ratio],
        pos_shape=(1, 10, 25),
    ),
    head=dict(
        type="GaNetHead",
        in_channel=hid_dim,
    ),
    targets=dict(
        type="GaNetTarget",
        hm_down_scale=8,
        radius=radius,
    ),
    post_process=dict(
        type="GaNetDecoder",
        root_thr=1,
        kpt_thr=0.4,
        cluster_thr=5,
        downscale=8,
    ),
    losses=dict(
        type="GaNetLoss",
        loss_kpts_cls=dict(
            type="LaneFastFocalLoss",
            loss_weight=1.0,
        ),
        loss_pts_offset_reg=dict(
            type="L1Loss",
            loss_weight=0.5,
        ),
        loss_int_offset_reg=dict(
            type="L1Loss",
            loss_weight=1.0,
        ),
    ),
)

deploy_model = dict(
    type="GaNet",
    backbone=model["backbone"],
    neck=model["neck"],
    head=model["head"],
    targets=None,
    post_process=None,
    losses=None,
)

hbir_model = dict(
    type="GaNetIrInfer",
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized_infer.bc"),
    ),
    post_process=dict(
        type="GaNetDecoder",
        root_thr=1,
        kpt_thr=0.4,
        cluster_thr=5,
        downscale=8,
    ),
)

hbm_model = dict(
    type="GaNetIrInfer",
    nv12_input=True,
    hbm_infer=True,
    ir_model=dict(
        type="HbmModule",
        model_path=os.path.join(ckpt_dir, "compile/model.hbm"),
        host="10.103.43.99",
    ),
    post_process=dict(
        type="GaNetDecoder",
        root_thr=1,
        kpt_thr=0.4,
        cluster_thr=5,
        downscale=8,
    ),
)
