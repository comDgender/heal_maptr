"""Deploy / HBIR export configuration for FlashOcc HENet LSS NuScenes."""

import os

from common import ckpt_dir, march, task_name
from model import deploy_inputs, deploy_model
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
    enable_vpu=False,
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
    input_source=["pyramid", "ddr", "ddr"],
    cache_path="",
)
