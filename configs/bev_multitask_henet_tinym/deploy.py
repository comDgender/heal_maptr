# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import os

from common import ckpt_dir, march, task_name
from model import deploy_inputs, deploy_model, output_names
from qconfig import deploy_model_convert_pipeline

hbir_exporter = dict(
    type="HbirExporter",
    march=march,
    model=deploy_model,
    model_convert_pipeline=deploy_model_convert_pipeline,
    example_inputs=deploy_inputs,
    save_path=ckpt_dir,
    model_name=task_name,
    input_names=list(deploy_inputs.keys()),
    output_names=output_names,
    enable_vpu=True,
)

compile_dir = os.path.join(ckpt_dir, "compile")

# img -> pyramid (NV12 via VPU); grids / caches / projection_mat -> ddr.
_input_source = ["pyramid"] + ["ddr"] * (len(deploy_inputs) - 1)

hbir_compiler = dict(
    type="HbirCompiler",
    model_path=os.path.join(ckpt_dir, "qat.bc"),
    out_path=os.path.join(compile_dir, "model.hbm"),
    march=march,
    opt=2,
    jobs=64,
    max_time_per_fc=0.0,
    preserve_quant_dequant=False,
    input_source=_input_source,
    debug=True,
    cache_path=os.path.join(compile_dir, "cache"),
    enable_vpu=True,
)

calops_cfg = dict(method="hook")
