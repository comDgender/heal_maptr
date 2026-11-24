"""Deploy / compile configuration for sparse multitask."""

import os

from common import ckpt_dir, deploy_inputs, march, task_name
from model import deploy_model, output_names
from qconfig import compile_cfg, export_model_convert_pipeline

hbir_exporter = dict(
    type="HbirExporter",
    march=march,
    model=deploy_model,
    model_convert_pipeline=export_model_convert_pipeline,
    example_inputs=deploy_inputs,
    save_path=ckpt_dir,
    model_name=task_name,
    input_names=list(deploy_inputs.keys()),
    output_names=output_names,
    enable_vpu=True,
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
    preserve_quant_dequant=False,
    input_source=compile_cfg["input_source"],
    debug=compile_cfg.get("debug", True),
    cache_path=os.path.join(compile_dir, "cache"),
)
