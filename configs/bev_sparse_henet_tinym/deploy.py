"""Deploy / HBIR export configuration for SparseBEV HENet-tiny NuScenes."""

import os

from common import ckpt_dir, march, task_name
from model import deploy_inputs, deploy_model, output_names
from qconfig import deploy_model_convert_pipeline

# Export hbir from the calibration product (bev_sparse has no qat stage):
# deploy_model_convert_pipeline builds a fake-quant graph from the float
# deploy_model (Float2QAT val + calibration qconfig setter) and loads the
# calibration checkpoint (fake-quant weights + observer scales). HbirExporter
# then converts it to a quantized hbir. Run:
#   python3 tools/export_hbir.py -c configs/bev_sparse_henet_tinym
# Produces qat.bc + quantized_infer.bc in ckpt_dir. output_names names the
# exported graph outputs so HbirModule (hbir_infer) returns them as dict
# keys (classification/prediction/quality/feature) — matching
# SparseBEVIrInfer.cache.
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
    input_source="pyramid, ddr, ddr, ddr",
    cache_path=os.path.join(compile_dir, "cache"),
)

calops_cfg = dict(method="hook")
