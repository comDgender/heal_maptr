import copy
import os

import torch
from common import ckpt_dir, march, num_queries
from qconfig import (  # noqa: F401
    qat_predict_model_convert_pipeline as _qat_pipeline,
)

# Own task_name so this subgraph does not collide with the main body: tools
# derive task as ``task_name or hbir_exporter.model_name`` and task names the
# perf csv dir, the board-side hbm dir and the board-side log dir. Without
# this, load_config_with_optional_base merges common.py's task_name in and
# both subgraphs would report as the main body.
task_name = "motr_efficientnetb3_qim"

# QIM sub-component, exported/compiled separately (DDR inputs, no image).
# Invoke: tools/export_hbir.py -c .../deploy_qim.py,
#         tools/compile_perf.py -c .../deploy_qim.py
# compile_qim=True routes forward through track_embed only.
qim_deploy_model = dict(
    type="Motr",
    # backbone/head intentionally omitted: not in qim-only trace graph,
    # so leaving them out avoids qconfig-template assigning them qint16
    # that diverges from the qat ckpt (where main-body attn_matmul was
    # downgraded to qint8 by enable_optimize during the combined trace).
    track_embed=dict(
        type="QueryInteractionModule",
        dim_in=256,
        hidden_dim=1024,
    ),
    compile_qim=True,
    num_queries=num_queries,
    queries_dim=256,
)

qim_deploy_inputs = dict(
    output_embedding=torch.randn((1, 1, num_queries, 256), dtype=torch.float),
    query_pos=torch.randn((1, 1, num_queries, 256), dtype=torch.float),
    mask_query=torch.ones((1, 1, 1, num_queries), dtype=torch.float),
)

qim_deploy_model_convert_pipeline = copy.deepcopy(_qat_pipeline)
qim_deploy_model_convert_pipeline["converters"][0].pop(
    "example_data_loader", None
)
qim_deploy_model_convert_pipeline["converters"][0][
    "example_inputs"
] = qim_deploy_inputs

hbir_exporter = dict(
    type="HbirExporter",
    march=march,
    model=qim_deploy_model,
    model_convert_pipeline=qim_deploy_model_convert_pipeline,
    example_inputs=qim_deploy_inputs,
    save_path=os.path.join(ckpt_dir, "qim"),
    model_name="qim",
    input_names=list(qim_deploy_inputs.keys()),
    enable_vpu=False,
    with_check=False,
)

qim_compile_dir = os.path.join(ckpt_dir, "qim", "compile")
hbir_compiler = dict(
    type="HbirCompiler",
    model_path=os.path.join(ckpt_dir, "qim", "qat.bc"),
    out_path=os.path.join(qim_compile_dir, "model.hbm"),
    march=march,
    opt=2,
    jobs=64,
    max_time_per_fc=0.0,
    preserve_quant_dequant=False,
    input_source=["ddr", "ddr", "ddr"],
    cache_path="",
)
