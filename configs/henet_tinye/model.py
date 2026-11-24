import os

from common import ckpt_dir, mixup, num_classes

depth = [3, 3, 8, 6]
width = [48, 96, 192, 384]
attention_block_num = [0, 0, 0, 0]
mlp_ratios, mlp_ratio_attn = [2, 2, 2, 3], 2
act_layer = ["nn.GELU", "nn.GELU", "nn.GELU", "nn.GELU"]
use_layer_scale = [True, True, True, True]
extra_act = [False, False, False, False]
final_expand_channel, feature_mix_channel = 0, 1024
block_cls = ["DWCB", "GroupDWCB", "AltDWCB", "DWCB"]
down_cls = ["S2DDown", "S2DDown", "S2DDown", "None"]
patch_embed = "origin"
stage_out_norm = False


model = dict(
    type="Classifier",
    backbone=dict(
        type="HENet",
        in_channels=3,
        block_nums=depth,
        embed_dims=width,
        attention_block_num=attention_block_num,
        mlp_ratios=mlp_ratios,
        mlp_ratio_attn=mlp_ratio_attn,
        act_layer=act_layer,
        use_layer_scale=use_layer_scale,
        layer_scale_init_value=1e-5,
        num_classes=num_classes,
        include_top=True,
        extra_act=extra_act,
        final_expand_channel=final_expand_channel,
        feature_mix_channel=feature_mix_channel,
        block_cls=block_cls,
        down_cls=down_cls,
        patch_embed=patch_embed,
        stage_out_norm=stage_out_norm,
    ),
    losses=(
        dict(type="CEWithLabelSmooth")
        if not mixup
        else dict(type="SoftTargetCrossEntropy")
    ),
)
deploy_model = dict(
    type="Classifier",
    backbone=dict(
        type="HENet",
        in_channels=3,
        block_nums=depth,
        embed_dims=width,
        attention_block_num=attention_block_num,
        mlp_ratios=mlp_ratios,
        mlp_ratio_attn=mlp_ratio_attn,
        act_layer=act_layer,
        use_layer_scale=use_layer_scale,
        layer_scale_init_value=1e-5,
        num_classes=num_classes,
        include_top=True,
        extra_act=extra_act,
        final_expand_channel=final_expand_channel,
        feature_mix_channel=feature_mix_channel,
        block_cls=block_cls,
        down_cls=down_cls,
        patch_embed=patch_embed,
        stage_out_norm=stage_out_norm,
    ),
    losses=None,
)

hbir_model = dict(
    type="ClassifierIrInfer",
    ir_model=dict(
        type="HbirModule",
        model_path=os.path.join(ckpt_dir, "quantized_infer.bc"),
    ),
)

hbm_model = dict(
    type="ClassifierIrInfer",
    is_nv12_type=True,
    ir_model=dict(
        type="HbmModule",
        model_path=os.path.join(ckpt_dir, "compile/model.hbm"),
        host="10.103.43.99",
    ),
)
