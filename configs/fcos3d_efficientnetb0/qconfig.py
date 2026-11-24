import copy
import os

from common import ckpt_dir
from data import calibration_data_loader, val_data_loader
from horizon_plugin_pytorch.dtype import qint8, qint16
from horizon_plugin_pytorch.quantization import get_qconfig, observer_v2
from horizon_plugin_pytorch.quantization.qconfig_setter import (
    ConvDtypeTemplate,
    EqualizeInOutScaleTemplate,
    MatmulDtypeTemplate,
    ModuleNameTemplate,
    QconfigSetter,
)

convert_mode = "jit-strip"

q_templates = [
    ModuleNameTemplate({"": qint8}),
    MatmulDtypeTemplate(input_dtypes=[qint8, qint16]),
    ConvDtypeTemplate(input_dtype=qint8, weight_dtype=qint8),
]

cali_qconfig_setter = QconfigSetter(
    reference_qconfig=get_qconfig(observer=(observer_v2.MSEObserver)),
    templates=q_templates,
    enable_optimize=True,
    save_dir=ckpt_dir,
    custom_qconfig_mapping=None,
    # Match qat_qconfig_setter's fuse behaviour so calib and qat build an
    # identical graph (same observer insertion points). Without this the
    # calib ckpt (graph with add->relu fused, observer at .3.0) cannot be
    # loaded by qat train (graph with only conv->relu fused, observer at
    # .2.floatF) under the newer plugin -> 72 miss / 72 unexpect at every
    # calib->qat / calib->predict handoff. Aligned with flashocc's config.
    optimize_templates_kwargs={
        EqualizeInOutScaleTemplate: {"only_fuse_relu_after_conv": True}
    },
)
qat_qconfig_setter = QconfigSetter(
    reference_qconfig=get_qconfig(
        observer=(observer_v2.MinMaxObserver), fix_scale=True
    ),
    templates=q_templates,
    enable_optimize=True,
    save_dir=ckpt_dir,
    custom_qconfig_mapping=None,
    optimize_templates_kwargs={
        EqualizeInOutScaleTemplate: {"only_fuse_relu_after_conv": True}
    },
)

calibration_example_data_loader = copy.deepcopy(calibration_data_loader)
calibration_example_data_loader["num_workers"] = 0

val_example_data_loader = copy.deepcopy(val_data_loader)
val_example_data_loader.pop("sampler")
val_example_data_loader["num_workers"] = 0

calibration_train_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "float-checkpoint-best.pth.tar"
            ),
        ),
        dict(
            type="Float2Calibration",
            convert_mode=convert_mode,
            example_data_loader=calibration_example_data_loader,
            qconfig_setter=cali_qconfig_setter,
        ),
    ],
)

calibration_predict_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        # cali_qconfig_setter now carries the same only_fuse_relu_after_conv
        # as qat_qconfig_setter, so calib predict (Float2QAT + cali setter)
        # builds the identical graph as calib train and qat train — the
        # calib ckpt loads with no miss/unexpect keys. LoadCheckpoint keeps
        # ignore_extra/verbose (matches qat_train / flashocc) but NOT
        # allow_miss, so any future graph drift surfaces loudly.
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            example_data_loader=copy.deepcopy(val_example_data_loader),
            qconfig_setter=cali_qconfig_setter,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "calibration-checkpoint-best.pth.tar"
            ),
            ignore_extra=True,
            verbose=True,
        ),
    ],
)

qat_train_model_convert_pipeline = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            example_data_loader=copy.deepcopy(calibration_example_data_loader),
            qconfig_setter=qat_qconfig_setter,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "calibration-checkpoint-best.pth.tar"
            ),
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
            example_data_loader=copy.deepcopy(val_example_data_loader),
            qconfig_setter=qat_qconfig_setter,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "qat-checkpoint-best.pth.tar"
            ),
        ),
    ],
)
