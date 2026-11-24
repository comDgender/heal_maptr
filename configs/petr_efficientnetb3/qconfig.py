import os

from common import ckpt_dir, convert_mode, qat_mode, training_step
from data import example_inputs
from horizon_plugin_pytorch.dtype import qint8, qint16
from horizon_plugin_pytorch.quantization import get_qconfig, observer_v2
from horizon_plugin_pytorch.quantization.qconfig_setter import (
    ConvDtypeTemplate,
    MatmulDtypeTemplate,
    ModuleNameTemplate,
    QconfigSetter,
)

attetion_matmul_names = [
    "head.transformer.decoder.decode_layers.0.self_attns.matmul",
    "head.transformer.decoder.decode_layers.1.self_attns.matmul",
    "head.transformer.decoder.decode_layers.2.self_attns.matmul",
    "head.transformer.decoder.decode_layers.3.self_attns.matmul",
    "head.transformer.decoder.decode_layers.4.self_attns.matmul",
    "head.transformer.decoder.decode_layers.5.self_attns.matmul",
    "head.transformer.decoder.decode_layers.0.corss_attn.matmul",
    "head.transformer.decoder.decode_layers.1.corss_attn.matmul",
    "head.transformer.decoder.decode_layers.2.corss_attn.matmul",
    "head.transformer.decoder.decode_layers.3.corss_attn.matmul",
    "head.transformer.decoder.decode_layers.4.corss_attn.matmul",
    "head.transformer.decoder.decode_layers.5.corss_attn.matmul",
]
attention_matmul_int16_templates = {}
for m in attetion_matmul_names:
    attention_matmul_int16_templates[m] = {
        "dtype": {
            "output": qint16,
        },
    }

qconfig_setter = QconfigSetter(
    reference_qconfig=get_qconfig(
        observer=(
            # Histogram over MSE: at 200 calib steps (800 frames) the histogram
            # uses the full sample set to place the clipping threshold, whereas
            # MSE fits per-batch and was the better fit at the old 40 steps.
            observer_v2.HistogramObserver
            if training_step == "calibration"
            else observer_v2.MinMaxObserver
        ),
        # fix_scale=True: QAT freezes activation scale at calib value.
        # Requires horizon-plugin-pytorch >= 3.4.1.
        fix_scale=training_step != "calibration",
    ),
    templates=[
        ModuleNameTemplate({"": qint8}, enable_propagate=True),
        ModuleNameTemplate(
            {"quant": {"dtype": qint8, "threshold": 1.0}},
        ),
        MatmulDtypeTemplate(
            input_dtypes=[qint8, qint8],
        ),
        ConvDtypeTemplate(
            input_dtype=qint8,
            weight_dtype=qint8,
        ),
        ModuleNameTemplate(
            attention_matmul_int16_templates,
        ),
    ],
    enable_optimize=True,
    save_dir="./qconfig_setting",
)

calibration_train_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    # qconfig_params removed: dead code under setter path (setter builds
    # kwargs from reference_qconfig, ignores hat_engine global qconfig).
    # The calibration observer is set by reference_qconfig above.
    converters=[
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "float-checkpoint-best.pth.tar"
            ),
            allow_miss=True,
            verbose=True,
        ),
        dict(
            type="Float2Calibration",
            convert_mode=convert_mode,
            qconfig_setter=qconfig_setter,
            example_inputs=example_inputs,
        ),
    ],
)

calibration_predict_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(
            type="Float2Calibration",
            convert_mode=convert_mode,
            qconfig_setter=qconfig_setter,
            example_inputs=example_inputs,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "calibration-checkpoint-best.pth.tar"
            ),
            ignore_extra=True,
            allow_miss=True,
            verbose=True,
        ),
    ],
)

qat_train_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    # activation scale freeze via fix_scale=True on setter reference_qconfig;
    # qconfig_params was dead code under the setter path.
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            qconfig_setter=qconfig_setter,
            example_inputs=example_inputs,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "calibration-checkpoint-best.pth.tar"
            ),
            # calib runs HistogramObserver, qat runs MinMaxObserver: the calib
            # ckpt carries extra histogram/histogram_edges buffers that the qat
            # graph has no slot for. Same flags as motr (qconfig.py:296-298).
            allow_miss=True,
            ignore_extra=True,
            verbose=True,
        ),
    ],
)

qat_predict_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            qconfig_setter=qconfig_setter,
            example_inputs=example_inputs,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "qat-checkpoint-best.pth.tar"
            ),
            ignore_extra=True,
            allow_miss=True,
            verbose=True,
        ),
    ],
)

real_deploy_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            qconfig_setter=qconfig_setter,
            example_inputs=example_inputs,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "qat-checkpoint-best.pth.tar"
            ),
            ignore_extra=True,
            allow_miss=True,
            verbose=True,
        ),
    ],
)
