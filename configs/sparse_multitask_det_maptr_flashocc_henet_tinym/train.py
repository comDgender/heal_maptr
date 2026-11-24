"""Trainer definitions for sparse multitask det + MapTR + FlashOcc."""

import copy
import os

import torch
from callback import (
    ckpt_callback,
    ckpt_callback_merge,
    ckpt_callback_stage1,
    ckpt_callback_stage2,
    freeze_module_callback,
    grad_callback,
    loss_show_update,
    stat_callback,
    val_callback,
    val_metrics,
    val_metrics_stage2,
)
from common import (
    calibration_num_steps,
    calibration_step,
    float_lr,
    float_stage2_lr,
    num_steps,
    pretrain_ckpt_dir,
)
from data import (
    batch_processor,
    calibration_batch_processor,
    calibration_data_loader,
    data_loader,
)
from model import model, model_stage2
from qconfig import (
    calibration_train_model_convert_pipeline,
    float_stage1_checkpoint,
    float_stage2_checkpoint,
)

# Stage2 contributes only the om(map) + occ head and neck weights to the
# merge — same semantics as the k8s `merge_checkpoints.py --prefixes
# head.om,head.occ,necks.om,necks.occ` (stage1 keeps `all`, stage2 overrides
# only these). depth_branch / backbone / head.det / necks.det stay from
# stage1 (depth_branch is NOT frozen in stage2, so an unfiltered stage2 load
# would wrongly overwrite it).
_STAGE2_KEEP_PREFIXES = (
    "head.om.",
    "head.occ.",
    "necks.om.",
    "necks.occ.",
)


def _filter_stage2_state_dict(state_dict):
    """Keep only stage2's om/occ head+neck keys (with _metadata)."""
    from collections import OrderedDict

    new_sd = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith(_STAGE2_KEEP_PREFIXES):
            new_sd[key] = value
    if hasattr(state_dict, "_metadata"):
        new_sd._metadata = OrderedDict(
            (k, v)
            for k, v in state_dict._metadata.items()
            if k.startswith(_STAGE2_KEEP_PREFIXES)
        )
    return new_sd


float_stage1_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    pretrain_ckpt_dir, "float-checkpoint-best.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
                ignore_tensor_shape=True,
                verbose=True,
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        params={"backbone": dict(lr=float_lr)},
        lr=float_lr,
        weight_decay=0.001,
    ),
    batch_processor=batch_processor,
    num_steps=num_steps,
    stop_by="step",
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        dict(type="ExponentialMovingAverage"),
        grad_callback,
        dict(
            type="CosineAnnealingLrUpdater",
            warmup_len=500,
            warmup_by="step",
            warmup_lr_ratio=1.0 / 3,
            step_log_interval=500,
            update_by="step",
            min_lr_ratio=1e-3,
        ),
        val_callback,
        ckpt_callback_stage1,
    ],
    train_metrics=dict(type="LossShow"),
    sync_bn=True,
    val_metrics=val_metrics,
)

float_stage2_trainer = dict(
    type="distributed_data_parallel_trainer",
    model=model_stage2,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        qat_mode="fuse_bn",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=float_stage1_checkpoint,
                load_ema_model=True,
                allow_miss=True,
                ignore_extra=True,
                ignore_tensor_shape=True,
                verbose=True,
            ),
        ],
    ),
    data_loader=data_loader,
    optimizer=dict(
        type=torch.optim.AdamW,
        lr=float_stage2_lr,
        weight_decay=0.001,
    ),
    batch_processor=batch_processor,
    num_steps=int(num_steps * 0.7),
    stop_by="step",
    device=None,
    callbacks=[
        stat_callback,
        loss_show_update,
        freeze_module_callback,
        # dict(type="ExponentialMovingAverage"),
        grad_callback,
        dict(
            type="CosineAnnealingLrUpdater",
            warmup_len=500,
            warmup_by="step",
            warmup_lr_ratio=1.0 / 3,
            step_log_interval=500,
            update_by="step",
            min_lr_ratio=1e-3,
        ),
        val_callback,
        ckpt_callback_stage2,
    ],
    train_metrics=dict(type="LossShow"),
    sync_bn=True,
    val_metrics=val_metrics_stage2,
)

# float_merge: no training — load stage1 (all) + stage2 (om/occ only) and save
# the merged complete model as float_merged-checkpoint-last.pth.tar. Replaces
# the external k8s merge_checkpoints shell step. num_epochs=0 skips the train
# loop (only on_loop_begin/end run); the Checkpoint callback saves `-last` on
# on_loop_end. Requires a CUDA-available node (Trainer asserts cuda) but does
# no GPU compute (device=None, merge runs on CPU).
float_merge_trainer = dict(
    type="Trainer",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=float_stage1_checkpoint,
                load_ema_model=True,
                allow_miss=True,
                ignore_extra=True,
                ignore_tensor_shape=True,
                verbose=True,
            ),
            dict(
                type="LoadCheckpoint",
                checkpoint_path=float_stage2_checkpoint,
                load_ema_model=True,
                allow_miss=True,
                ignore_extra=True,
                ignore_tensor_shape=True,
                verbose=True,
                state_dict_update_func=_filter_stage2_state_dict,
            ),
        ],
    ),
    data_loader=None,
    optimizer=None,
    batch_processor=None,
    num_epochs=0,
    device=None,
    callbacks=[ckpt_callback_merge],
)

calibration_val_callback = copy.deepcopy(val_callback)
calibration_val_callback["val_interval"] = 1

calibration_trainer = dict(
    type="Calibrator",
    model=model,
    model_convert_pipeline=calibration_train_model_convert_pipeline,
    data_loader=calibration_data_loader,
    batch_processor=calibration_batch_processor,
    num_steps=calibration_num_steps,
    device=None,
    callbacks=[ckpt_callback, val_callback],
    log_interval=max(1, calibration_step // 10),
    val_metrics=val_metrics,
)
