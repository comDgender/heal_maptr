"""Deploy / compile configs."""

import copy
import os

from common import (
    ckpt_dir,
    convert_mode,
    enable_vpu,
    march,
    qat_mode,
    task_name,
)
from model import deploy_inputs, deploy_model
from qconfig import qat_qconfig_setter

deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            qconfig_setter=qat_qconfig_setter,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "qat-checkpoint-last.pth.tar"
            ),
        ),
    ],
)

compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=task_name,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["ddr", "ddr"],
    opt="O2",
    output_layout="NHWC",
    enable_vpu=enable_vpu,
    transpose_dim=dict(
        outputs={
            "global": [0, 2, 3, 1],
        }
    ),
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

hbir_compiler = dict(
    type="HbirCompiler",
    model_path=os.path.join(ckpt_dir, "qat.bc"),
    out_path=os.path.join(compile_dir, "model.hbm"),
    march=march,
    opt=2,
    jobs=64,
    max_time_per_fc=0.0,
    debug=True,
    input_source=["ddr", "ddr"],
    enable_vpu=enable_vpu,
    cache_path=os.path.join(compile_dir, "cache"),
    transpose_dim=dict(
        outputs={
            "global": [0, 2, 3, 1],
        }
    ),
)
