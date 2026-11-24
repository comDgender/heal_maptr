"""Quantization sensitivity analysis config."""

import copy
import os

from common import ckpt_dir
from data import val_data_loader
from model import model
from qconfig import (
    float_merged_predict_model_convert_pipeline,
    quant_analysis_model_convert_pipeline,
)

analysis_dataloader = copy.deepcopy(val_data_loader)
analysis_dataloader["num_workers"] = 0

quant_analysis_solver = dict(
    type="QuantAnalysis",
    model=copy.deepcopy(model),
    device_id=0,
    dataloader=analysis_dataloader,
    num_steps=100,
    baseline_model_convert_pipeline=(
        float_merged_predict_model_convert_pipeline
    ),
    analysis_model_convert_pipeline=quant_analysis_model_convert_pipeline,
    analysis_model_type="fake_quant",
    out_dir=os.path.join(ckpt_dir, "quant_analysis"),
)
