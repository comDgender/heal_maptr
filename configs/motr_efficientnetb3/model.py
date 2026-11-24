import copy

from common import bn_kwargs, ckpt_dir, num_classes, num_queries
from common import train_batch_size_per_gpu as batch_size

# `criterion` lives only in `train_model` (not `model`/`test_model`):
# MotrCriterion.forward caches non-leaf loss tensors in `losses_dict`,
# which breaks deepcopy of the model during val (Validation callback).
model = dict(
    type="Motr",
    backbone=dict(
        type="efficientnet",
        bn_kwargs=bn_kwargs,
        model_type="b3",
        num_classes=1000,
        include_top=False,
        activation="relu",
        use_se_block=False,
    ),
    head=dict(
        type="MotrHead",
        transformer=dict(
            type="MotrDeformableTransformer",
            pos_embed=dict(
                type="PositionEmbeddingSine",
                num_pos_feats=128,
                normalize=True,
                temperature=20,
            ),
            d_model=256,
            num_queries=num_queries,
            dim_feedforward=1024,
            dropout=0.0,
            return_intermediate_dec=True,
            extra_track_attn=True,
            enc_n_points=1,
            dec_n_points=1,
        ),
        num_classes=num_classes,
        in_channels=[384],
        max_per_img=num_queries,
    ),
    post_process=dict(
        type="MotrPostProcess",
    ),
    track_embed=dict(
        type="QueryInteractionModule",
        dim_in=256,
        hidden_dim=1024,
    ),
    batch_size=batch_size,
)

test_model = copy.deepcopy(model)

# train_model adds criterion for float/qat trainers; calibrator uses
# the criterion-free `model`.
train_model = copy.deepcopy(model)
train_model["criterion"] = dict(
    type="MotrCriterion",
    num_classes=num_classes,
)

deploy_model = dict(
    type="Motr",
    backbone=model["backbone"],
    head=dict(
        type="MotrHead",
        transformer=dict(
            type="MotrDeformableTransformer",
            pos_embed=model["head"]["transformer"]["pos_embed"],
            d_model=256,
            num_queries=num_queries,
            dim_feedforward=1024,
            dropout=0.1,
            return_intermediate_dec=False,
            extra_track_attn=True,
            enc_n_points=1,
            dec_n_points=1,
        ),
        num_classes=num_classes,
        in_channels=[384],
        max_per_img=num_queries,
    ),
    # track_embed omitted: not in main-body trace graph, so leaving it
    # out avoids qconfig-template assigning it qint16 (qat ckpt has it
    # as qint8, downgraded by enable_optimize during combined trace).
    # qim is exported separately via deploy_qim.py.
    compile_motr=True,
)

hbir_model = dict(
    type="MotrIrInfer",
    ir_model=dict(
        type="HbirModule",
        model_path="%s/quantized_infer.bc" % ckpt_dir,
    ),
    qim_ir_model=dict(
        type="HbirModule",
        model_path="%s/qim/quantized_infer.bc" % ckpt_dir,
    ),
    post_process=dict(
        type="MotrPostProcess",
    ),
    LoadCheckpoint=dict(
        type="LoadCheckpoint",
        checkpoint_path="%s/qat-checkpoint-best.pth.tar" % ckpt_dir,
        allow_miss=True,
        ignore_extra=True,
    ),
    num_classes=num_classes,
)

hbm_model = dict(
    type="MotrIrInfer",
    ir_model=dict(
        type="HbmModule",
        model_path="%s/compile/model.hbm" % ckpt_dir,
        host="10.103.43.99",
    ),
    qim_ir_model=dict(
        type="HbmModule",
        model_path="%s/qim/compile/model.hbm" % ckpt_dir,
        host="10.103.43.99",
    ),
    post_process=dict(
        type="MotrPostProcess",
    ),
    LoadCheckpoint=dict(
        type="LoadCheckpoint",
        checkpoint_path="%s/qat-checkpoint-best.pth.tar" % ckpt_dir,
        allow_miss=True,
        ignore_extra=True,
    ),
    num_classes=num_classes,
    nv12_input=True,
    hbm_infer=True,
)
