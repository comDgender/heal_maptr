import copy
import os

import torch
from common import ckpt_dir, enable_vpu, march, task_name
from model import deploy_model, example_inputs
from qconfig import qat_qconfig_setter

deploy_inputs = dict(
    points=[
        torch.randn(150000, 4),
    ],
)

deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(
            type="Float2QAT",
            convert_mode="jit-strip",
            example_inputs=example_inputs,
            qconfig_setter=qat_qconfig_setter,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "qat-checkpoint-best.pth.tar"
            ),
            allow_miss=True,
            ignore_extra=True,
        ),
    ],
)


hbir_exporter = dict(
    type="HbirExporter",
    march=march,
    model=deploy_model,
    model_convert_pipeline=copy.deepcopy(deploy_model_convert_pipeline),
    example_inputs=deploy_inputs,
    save_path=ckpt_dir,
    model_name=task_name,
    input_names=list(deploy_inputs.keys()),
    enable_vpu=enable_vpu,
)

compile_dir = os.path.join(ckpt_dir, "compile")
hbir_compiler = dict(
    type="HbirCompiler",
    model_path=os.path.join(ckpt_dir, "qat.bc"),
    out_path=os.path.join(compile_dir, "model.hbm"),
    march=march,
    opt=2,
    jobs=64,
    max_time_per_fc=0.0,
    debug=True,
    input_source=["ddr"],
    enable_vpu=enable_vpu,
    cache_path=os.path.join(compile_dir, "cache"),
    # transpose_dim=dict(
    #     outputs={
    #         "global": [0, 2, 3, 1],
    #     }
    # ),
)
