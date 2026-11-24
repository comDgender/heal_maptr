"""Quantization sensitivity analysis config."""

import copy
import os

from common import ckpt_dir
from data import val_data_loader
from model import model
from predict import float_predictor
from qconfig import quant_analysis_model_convert_pipeline

analysis_dataloader = copy.deepcopy(val_data_loader)
analysis_dataloader["num_workers"] = 0

# quant_analysis does per-layer L1 comparison on raw decoder outputs, so
# skip the head's post_process (BevFormerProcess decode/NMS) — it is
# non-differentiable and would distort the sensitivity analysis. This
# mirrors deploy, which also drops post_process. Flip this flag to False
# to keep post_process during analysis.
disable_post_process = True

quant_analysis_model = copy.deepcopy(model)
if disable_post_process:
    for _decoder in quant_analysis_model["bev_decoders"]:
        if "post_process" in _decoder:
            _decoder.pop("post_process")

quant_analysis_solver = dict(
    type="QuantAnalysis",
    model=quant_analysis_model,
    device_id=0,
    dataloader=analysis_dataloader,
    num_steps=100,
    baseline_model_convert_pipeline=float_predictor["model_convert_pipeline"],
    analysis_model_convert_pipeline=quant_analysis_model_convert_pipeline,
    analysis_model_type="fake_quant",
    out_dir=os.path.join(ckpt_dir, "quant_analysis"),
)
