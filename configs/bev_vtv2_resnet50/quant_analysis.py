"""Quantization sensitivity analysis config."""

import copy
import os

from common import ckpt_dir
from data import val_data_loader
from model import model
from predict import calibration_predictor, float_predictor

analysis_dataloader = copy.deepcopy(val_data_loader)
if isinstance(analysis_dataloader, dict):
    analysis_dataloader["num_workers"] = 0

# Sensitivity uses head tensors only; CenterPointPostProcess (NMS) removed.
quant_analysis_model = copy.deepcopy(model)
for _decoder in quant_analysis_model["bev_decoders"]:
    _decoder.pop("decoder", None)

quant_analysis_solver = dict(
    type="QuantAnalysis",
    model=quant_analysis_model,
    device_id=0,
    dataloader=analysis_dataloader,
    num_steps=100,
    baseline_model_convert_pipeline=float_predictor["model_convert_pipeline"],
    analysis_model_convert_pipeline=calibration_predictor[
        "model_convert_pipeline"
    ],
    analysis_model_type="fake_quant",
    out_dir=os.path.join(ckpt_dir, "quant_analysis"),
)
