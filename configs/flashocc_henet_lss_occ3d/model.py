"""Model structure config for FlashOcc HENet LSS Occ3D NuScenes."""

import copy
import os

import torch
from common import (
    bev_size,
    ckpt_dir,
    data_shape,
    depth,
    depth_quant_scale,
    grid_quant_scale,
    grid_size,
    num_classes,
    num_points,
    vt_input_hw,
)

model = dict(
    type="Bev",
    bev_feat_index=-1,
    bev_upscale=2,
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
        in_strides=[2, 4, 8, 16, 32],
        in_channels=[64, 64, 128, 192, 384],
        out_strides=[16, 32],
        out_channels=[256, 256],
        bn_kwargs=dict(eps=1e-5, momentum=0.1),
    ),
    view_transformer=dict(
        type="LSSTransformer",
        in_channels=256,
        feat_channels=64,
        z_range=(-1.0, 5.4),
        depth=depth,
        num_points=num_points,
        bev_size=bev_size,
        grid_size=grid_size,
        num_views=6,
        grid_quant_scale=grid_quant_scale,
        depth_grid_quant_scale=depth_quant_scale,
    ),
    bev_encoder=dict(
        type="BevEncoder",
        backbone=dict(
            type="HENet",
            in_channels=64,
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
            quant_input=False,
        ),
        neck=dict(
            type="BiFPN",
            in_strides=[2, 4, 8, 16, 32],
            out_strides=[2, 4, 8, 16, 32],
            stride2channels=dict({2: 64, 4: 64, 8: 128, 16: 192, 32: 384}),
            out_channels=48,
            num_outs=5,
            stack=3,
            start_level=0,
            end_level=-1,
            fpn_name="bifpn_sum",
            upsample_type="function",
            use_fx=True,
        ),
    ),
    bev_decoders=[
        dict(
            type="FlashOccDetDecoder",
            use_mask=True,
            num_classes=num_classes,
            occ_head=dict(
                type="BEVOCCHead2D",
                in_dim=48,
                out_dim=128,
                Dz=16,
                num_classes=num_classes,
                use_predicter=True,
                use_upsample=True,
            ),
            loss_occ=dict(
                type="CrossEntropyLoss",
                use_sigmoid=False,
                ignore_index=255,
                loss_weight=1.0,
            ),
        ),
    ],
)

deploy_model = copy.deepcopy(model)
deploy_model["compile_model"] = True
deploy_model["bev_decoders"][0]["is_compile"] = True


def get_example_inputs(num_pts):
    inputs = {"img": torch.randn((6,) + data_shape)}
    inputs["ego2img"] = torch.randn((6, 4, 4))
    return inputs


example_inputs = get_example_inputs(num_points)

deploy_inputs = {
    "img": torch.randn((6, 3, data_shape[1], data_shape[2])),
    "points0": torch.randn((num_points, 128, 128, 2)),
    "points1": torch.randn((num_points, 128, 128, 2)),
}

hbir_deploy_model = copy.deepcopy(deploy_model)
hbir_deploy_model["compile_model"] = False

float_load_checkpoint_pipeline = dict(
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

hbir_infer_model = dict(
    type="BevIrInfer",
    deploy_model=hbir_deploy_model,
    vt_input_hw=vt_input_hw,
    model_convert_pipeline=float_load_checkpoint_pipeline,
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized_infer.bc"),
    ),
    bev_decoder_infers=[
        dict(
            type="FlashOccDecoderInfer",
            name="occ_det",
        ),
    ],
)

hbm_infer_model = dict(
    type="BevIrInfer",
    deploy_model=hbir_deploy_model,
    vt_input_hw=vt_input_hw,
    use_nv12=True,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[],
    ),
    ir_model=dict(
        type="HbmModule",
        model_path=os.path.join(ckpt_dir, "compile/model.hbm"),
        host="10.35.xx.xx",
        return_tensor=True,
    ),
    bev_decoder_infers=[
        dict(
            type="FlashOccDecoderInfer",
            name="occ_det",
        ),
    ],
)
