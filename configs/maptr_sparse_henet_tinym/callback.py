"""Training / validation callbacks and metrics."""

from common import (
    ckpt_dir,
    eval_use_same_gt_sample_num_flag,
    fixed_ptsnum_per_gt_line,
    log_freq,
    map_classes,
    point_cloud_range,
    task_name,
    training_step,
    val_interval,
)
from data import val_batch_processor, val_data_loader
from model import test_model

val_map_metric = dict(
    type="NuscenesMapMetric",
    classes=map_classes,
    save_prefix="./WORKSPACE/results" + task_name,
    fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
    pc_range=point_cloud_range,
    metric="chamfer",
    eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
)


def update_loss(metrics, batch, model_outs):
    for model_out in model_outs:
        for metric in metrics:
            metric.update(model_out)


loss_show_update = dict(
    type="MetricUpdater",
    metric_update_func=update_loss,
    step_log_freq=log_freq,
    epoch_log_freq=1,
    log_prefix="",
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=log_freq,
)


def update_val_metric(metrics, batch, model_outs):
    preds = model_outs
    for metric in metrics:
        metric.update(batch, preds)


val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_val_metric,
    step_log_freq=1000000,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)

val_callback = dict(
    type="Validation",
    val_interval=val_interval,
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=test_model,
    val_on_train_end=True,
    init_with_train_model=True,
    log_interval=200,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    save_interval=1,
    strict_match=True,
    mode="max",
    monitor_metric_key="MAP",
)

grad_callback = dict(
    type="GradScale",
    module_and_scale=[],
    clip_grad_norm=35,
    clip_norm_type=2,
)
