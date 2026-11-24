import copy
import os

import torch
from common import ckpt_dir, march, num_queries, task_name
from model import deploy_model
from qconfig import (  # noqa: F401
    qat_predict_model_convert_pipeline as _qat_pipeline,
)

deploy_inputs = dict(
    img=torch.randn((1, 3, 800, 1422)),
    query_pos=torch.randn((1, 256, 2, 128), dtype=torch.float),
    mask_query=torch.ones((1, 1, 1, num_queries), dtype=torch.float),
    ref_pts=torch.randn((1, 4, 2, 128), dtype=torch.float),
)

deploy_model_convert_pipeline = copy.deepcopy(_qat_pipeline)
deploy_model_convert_pipeline["converters"][0].pop("example_data_loader", None)
deploy_model_convert_pipeline["converters"][0][
    "example_inputs"
] = deploy_inputs

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
    input_source=["pyramid", "ddr", "ddr", "ddr"],
    cache_path="",
)

# QIM is exported & compiled separately (DDR inputs, no image). See
# deploy_qim.py — shared tools read top-level hbir_exporter/hbir_compiler,
# so qim lives in its own file.
