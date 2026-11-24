"""Deploy / compile configs for BevFusion multisensor multitask."""

import os

from common import ckpt_dir, march, task_name
from model import deploy_inputs, deploy_model, output_names
from qconfig import real_deploy_model_convert_pipeline

hbir_exporter = dict(
    type="HbirExporter",
    model=deploy_model,
    model_convert_pipeline=real_deploy_model_convert_pipeline,
    example_inputs=deploy_inputs,
    save_path=ckpt_dir,
    model_name=task_name,
    march=march,
    input_names=list(deploy_inputs.keys()),
    output_names=output_names,
)

compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=task_name,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    # input_source per deploy_inputs key (features, coors, img,
    # queries_rebatch_grid, restore_bev_grid, reference_points_rebatch,
    # bev_pillar_counts): img -> pyramid (NV12 via VPU), the rest -> ddr.
    input_source="ddr, ddr, pyramid, ddr, ddr, ddr, ddr",
    opt="O2",
    enable_vpu=True,
)
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

calops_cfg = dict(method="hook")
