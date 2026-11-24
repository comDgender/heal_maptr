"""Quantization config and model convert pipelines for MapTR BEVFormer.

Quantization flow (calibration-only, no QAT training):
    float -> calibration -> predict / deploy / export
"""

import copy
import os

import numpy as np
import torch
from common import (
    bev_h_,
    bev_w_,
    ckpt_dir,
    convert_mode,
    deploy_inputs,
    march,
    orig_shape,
    qat_mode,
    task_name,
    use_lidar2img,
    val_data_shape,
)
from horizon_plugin_pytorch.dtype import qint8
from horizon_plugin_pytorch.quantization import get_qconfig, observer_v2
from horizon_plugin_pytorch.quantization.qconfig_setter import (
    ConvDtypeTemplate,
    EqualizeInOutScaleTemplate,
    MatmulDtypeTemplate,
    ModuleNameTemplate,
    QconfigSetter,
)

_num_cams = 6
_homo_key = "lidar2img" if use_lidar2img else "ego2img"


def _resize_homo(homo: np.ndarray, scale_hw) -> np.ndarray:
    view = np.eye(4, dtype=np.float64)
    view[0, 0] = scale_hw[1]
    view[1, 1] = scale_hw[0]
    return np.stack([view @ h for h in homo]).astype(np.float32)


def _build_trace_homo() -> np.ndarray:
    """Valid camera ego2img for non-compile point_sampling.

    Identity/random mats overflow restore_bev_grid. Prefer demo sample; else
    use a baked-in copy (same sample, resized to val 480x800).
    """
    demo = os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../demo/maptr_henet_tinym_bevformer/sample_idx_0",
            f"{_homo_key}.npy",
        )
    )
    if os.path.isfile(demo):
        homo = np.load(demo)
        if homo.ndim == 4:
            homo = homo[0]
        scale = (
            val_data_shape[1] / orig_shape[1],
            val_data_shape[2] / orig_shape[2],
        )
        return _resize_homo(homo, scale)
    # baked-in fallback (demo sample @ 480x800); do not use eye/randn
    return np.array(
        [
            758.36023,
            -24.275229,
            -0.3370879,
            -1143.1324,
            145.9391,
            212.18965,
            -678.0892,
            696.11005,
            0.5712754,
            0.82075477,
            0.00244738,
            -1.280219,
            0.0,
            0.0,
            0.0,
            1.0,
            411.7201,
            -630.8799,
            -1.7926393,
            -687.4815,
            258.32156,
            0.92357117,
            -676.8903,
            583.38684,
            0.99996793,
            0.00568015,
            -0.00564133,
            -1.6923034,
            0.0,
            0.0,
            0.0,
            1.0,
            -301.5422,
            -685.2617,
            -11.222281,
            146.32037,
            146.18408,
            -208.99846,
            -675.9672,
            681.2466,
            0.553379,
            -0.83281773,
            -0.0136479,
            -1.2487091,
            0.0,
            0.0,
            0.0,
            1.0,
            469.2562,
            575.7544,
            -3.9810114,
            -758.79364,
            -77.99393,
            239.69153,
            -674.3792,
            1037.4935,
            -0.3188655,
            0.94766474,
            -0.01601021,
            -0.10370525,
            0.0,
            0.0,
            0.0,
            1.0,
            -413.57056,
            405.6378,
            5.3236246,
            1.9082532,
            -264.14227,
            -1.0693067,
            -427.21826,
            682.10767,
            -0.9998567,
            0.00248837,
            0.01674384,
            0.00187313,
            0.0,
            0.0,
            0.0,
            1.0,
            -731.9219,
            -153.68678,
            -13.369918,
            689.8436,
            -84.19762,
            -242.22983,
            -675.9605,
            1025.16,
            -0.354884,
            -0.93476886,
            -0.01626597,
            -0.06364233,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        dtype=np.float32,
    ).reshape(_num_cams, 4, 4)


# Synthetic img/osm only — avoid val dataloader OOM under parallel smoke.
trace_inputs = dict(
    img=torch.randn((_num_cams, 3, 480, 800)),
    osm_mask=torch.randn((1, 1, bev_h_, bev_w_)),
    seq_meta=[
        {
            "meta": [{"scene": "trace"}],
            _homo_key: [_build_trace_homo()],
        }
    ],
)

q_templates = [
    ModuleNameTemplate({"": qint8}),
    MatmulDtypeTemplate(input_dtypes=[qint8, qint8]),
    ConvDtypeTemplate(input_dtype=qint8, weight_dtype=qint8),
]

cali_qconfig_setter = QconfigSetter(
    reference_qconfig=get_qconfig(observer=(observer_v2.HistogramObserver)),
    templates=q_templates,
    enable_optimize=True,
    save_dir=ckpt_dir,
    custom_qconfig_mapping=None,
    optimize_templates_kwargs={
        EqualizeInOutScaleTemplate: {"only_fuse_relu_after_conv": True}
    },
)

float_predict_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    converters=[
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "float-checkpoint-best.pth.tar"
            ),
            ignore_extra=True,
            verbose=True,
        ),
    ],
)

calibration_train_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "float-checkpoint-best.pth.tar"
            ),
            ignore_extra=True,
            verbose=True,
            check_hash=False,
        ),
        dict(type="RepModel2Deploy"),
        dict(
            type="Float2Calibration",
            convert_mode=convert_mode,
            example_inputs=copy.deepcopy(trace_inputs),
            qconfig_setter=cali_qconfig_setter,
        ),
        dict(type="FixWeightQScale"),
    ],
)

calibration_val_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(type="RepModel2Deploy"),
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            example_inputs=copy.deepcopy(trace_inputs),
            qconfig_setter=cali_qconfig_setter,
        ),
    ],
)

calibration_predict_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(type="RepModel2Deploy"),
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            example_inputs=copy.deepcopy(trace_inputs),
            qconfig_setter=cali_qconfig_setter,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "calibration-checkpoint-best.pth.tar"
            ),
            allow_miss=True,
            ignore_extra=True,
            verbose=True,
            check_hash=False,
        ),
    ],
)

# Backward-compatible alias: ``-s qat`` uses the same calibration pipeline.
qat_predict_model_convert_pipeline = copy.deepcopy(
    calibration_predict_model_convert_pipeline
)

deploy_model_convert_pipeline = copy.deepcopy(
    calibration_predict_model_convert_pipeline
)

export_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(type="RepModel2Deploy"),
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            example_inputs=copy.deepcopy(deploy_inputs),
            qconfig_setter=cali_qconfig_setter,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "calibration-checkpoint-best.pth.tar"
            ),
            allow_miss=True,
            ignore_extra=True,
            verbose=True,
            check_hash=False,
        ),
    ],
)

compile_dir = os.path.join(ckpt_dir, "compile")
compile_cfg = dict(
    march=march,
    name=task_name,
    out_dir=compile_dir,
    hbm=os.path.join(compile_dir, "model.hbm"),
    layer_details=True,
    input_source=["pyramid", "ddr", "ddr", "ddr", "ddr", "ddr"],
    opt="O2",
    split_dim=dict(inputs={"0": [0, 6]}),
    debug=True,
)
