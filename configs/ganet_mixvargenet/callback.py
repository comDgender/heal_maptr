from typing import Dict, Sequence

from common import ckpt_dir, task_name, training_step
from data import val_batch_processor, val_data_loader
from model import model


def update_loss(
    metrics: Sequence[object], batch: Dict, model_outs: Dict
) -> None:
    for metric in metrics:
        metric.update(model_outs)


def update_metric(
    metrics: Sequence[object], batch: Dict, model_outs: Dict
) -> None:
    target = batch["ori_gt_lines"]
    for metric in metrics:
        metric.update(target, model_outs)


loss_show_update = dict(
    type="MetricUpdater",
    metric_update_func=update_loss,
    step_log_freq=20,
    epoch_log_freq=1,
    log_prefix="loss_" + task_name,
)

metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=20,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=1000,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=False,
    mode="max",
    monitor_metric_key="CulaneF1Score",
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[metric_updater],
    val_on_train_end=False,
)

# Calibrator uses deploy_model (post_process=None) so jit-strip trace skips
# GaNetDecoder (has .cpu().numpy() + meta_data dict access, untraceable).
# But CulaneF1Score needs decoded lanes, so val must run the decoder. Attach
# val_model=model (with GaNetDecoder); Validation deepcopies it, runs the
# Float2Calibration pipeline (attached in train.py) to match the trainer's
# qconfig, then loads deploy_model's state_dict (backbone/neck/head
# compatible, decoder has no trainable weights — unexpect_key tolerated by
# strict_match=False). Mirrors motr_efficientnetb3 pattern.
calibration_val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[metric_updater],
    val_model=model,
    init_with_train_model=True,
    strict_match=False,
    val_on_train_end=False,
)
