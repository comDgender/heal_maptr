# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import copy
import os

from common import ckpt_dir, convert_mode, qat_mode
from data import example_inputs, val_data_loader
from horizon_plugin_pytorch.quantization import get_qconfig, observer_v2
from horizon_plugin_pytorch.quantization.qconfig_setter import QconfigSetter
from model import model
from qconfig import q_template

# calibration ckpt → fake-quant via Float2Calibration (no fix_scale).
# Base templates only (no frozen); CASCADING matters: analysis sees how each
# op's dtype affects output sensitivity.
quant_analysis_qconfig_setter = QconfigSetter(
    reference_qconfig=get_qconfig(observer=(observer_v2.HistogramObserver)),
    templates=q_template,
    enable_optimize=True,
    save_dir=os.path.join(ckpt_dir, "./qconfig_setting_analysis"),
    custom_qconfig_mapping=None,
)

quant_analysis_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(type="RepModel2Deploy"),
        dict(
            type="Float2Calibration",
            convert_mode=convert_mode,
            example_inputs=example_inputs,
            qconfig_setter=quant_analysis_qconfig_setter,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "calibration-checkpoint-last.pth.tar"
            ),
            allow_miss=True,
            ignore_extra=True,
            verbose=True,
        ),
    ],
)

# float baseline pipeline for sensitivity (load float ckpt alone, no quant).
float_predict_model_convert_pipeline_for_analysis = dict(
    type="ModelConvertPipeline",
    converters=[
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "float_stage3-checkpoint-best.pth.tar"
            ),
            allow_miss=True,
            ignore_extra=True,
            verbose=True,
        ),
    ],
)

analysis_dataloader = copy.deepcopy(val_data_loader)
analysis_dataloader["num_workers"] = 0

# Sensitivity target: trace Cosine/L1/ATOL between float and fake-quant model
# outputs across each task head.
quant_analysis_solver = dict(
    type="QuantAnalysis",
    model=copy.deepcopy(model),
    device_id=0,
    dataloader=analysis_dataloader,
    num_steps=100,
    baseline_model_convert_pipeline=(
        float_predict_model_convert_pipeline_for_analysis
    ),
    analysis_model_convert_pipeline=quant_analysis_model_convert_pipeline,
    analysis_model_type="fake_quant",
    out_dir=os.path.join(ckpt_dir, "quant_analysis"),
)
