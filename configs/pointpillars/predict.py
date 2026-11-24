import copy
import os

from callback import val_metric_updater
from common import ckpt_dir, class_names
from data import hbir_data_loader, val_batch_processor, val_data_loader
from model import example_inputs, hbir_model, hbm_model, model
from qconfig import qat_qconfig_setter

float_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
            ),
        ],
    ),
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    metrics=dict(
        type="Kitti3DMetricDet",
        compute_aos=True,
        current_classes=class_names,
        difficultys=[0, 1, 2],
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=100,
)

calibration_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="Float2QAT",
                convert_mode="jit-strip",
                example_inputs=example_inputs,
                qconfig_setter=qat_qconfig_setter,
            ),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "calibration-checkpoint-best.pth.tar"
                ),
                verbose=True,
                ignore_extra=True,
                allow_miss=True,
            ),
        ],
    ),
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    metrics=dict(
        type="Kitti3DMetricDet",
        compute_aos=True,
        current_classes=class_names,
        difficultys=[0, 1, 2],
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=100,
)

qat_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="Float2QAT",
                convert_mode="jit-strip",
                example_inputs=example_inputs,
                qconfig_setter=qat_qconfig_setter,
            ),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "qat-checkpoint-best.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
                verbose=True,
            ),
        ],
    ),
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    device=None,
    metrics=dict(
        type="Kitti3DMetricDet",
        compute_aos=True,
        current_classes=class_names,
        difficultys=[0, 1, 2],
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=100,
)

hbir_predictor = dict(
    type="Predictor",
    model=hbir_model,
    data_loader=[hbir_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=dict(
        type="Kitti3DMetricDet",
        compute_aos=True,
        current_classes=class_names,
        difficultys=[0, 1, 2],
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)

align_bpu_predictor = copy.deepcopy(hbir_predictor)
align_bpu_predictor["log_interval"] = 1
align_bpu_predictor["data_loader"] = align_bpu_predictor["data_loader"][0]
align_bpu_predictor["batch_processor"] = dict(
    type="BasicBatchProcessor", need_grad_update=False
)

hbm_predictor = dict(
    type="Predictor",
    model=hbm_model,
    data_loader=[hbir_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=dict(
        type="Kitti3DMetricDet",
        compute_aos=True,
        current_classes=class_names,
        difficultys=[0, 1, 2],
    ),
    callbacks=[
        val_metric_updater,
    ],
    log_interval=1,
)
