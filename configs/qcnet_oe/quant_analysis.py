"""Quantization sensitivity analysis config."""

import copy
import os

from common import ckpt_dir
from data import val_data_loader
from model import cali_model
from predict import calibration_predictor, float_predictor

quant_data_loader = copy.deepcopy(val_data_loader)
quant_data_loader["batch_size"] = 1
quant_data_loader["num_workers"] = 0

quant_analysis_solver = dict(
    type="QuantAnalysis",
    model=cali_model,
    device_id=0,
    dataloader=quant_data_loader,
    num_steps=1000,
    baseline_model_convert_pipeline=float_predictor["model_convert_pipeline"],
    analysis_model_convert_pipeline=calibration_predictor[
        "model_convert_pipeline"
    ],
    analysis_model_type="fake_quant",
    out_dir=os.path.join(ckpt_dir, "analysis"),
)
