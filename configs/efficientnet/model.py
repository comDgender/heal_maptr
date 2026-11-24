import os

import torch
from common import ckpt_dir, num_classes

model = dict(
    type="Classifier",
    backbone=dict(
        type="efficientnet",
        model_type="b0",
        num_classes=num_classes,
        activation="relu",
        use_se_block=False,
        bn_kwargs={},
    ),
    losses=dict(type="CEWithLabelSmooth"),
)
deploy_model = dict(
    type="Classifier",
    backbone=dict(
        type="efficientnet",
        model_type="b0",
        num_classes=num_classes,
        activation="relu",
        use_se_block=False,
        flat_output=False,
        bn_kwargs={},
    ),
    losses=None,
)
deploy_inputs = dict(img=torch.randn((1, 3, 224, 224)))

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
