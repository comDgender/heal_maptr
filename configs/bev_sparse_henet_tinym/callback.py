"""Training / validation callbacks and metrics."""

from common import (
    CLASSES,
    batch_size_per_gpu,
    ckpt_dir,
    lossshow,
    meta_rootdir,
    num_steps_per_epoch,
    task_name,
    training_step,
)
from data import val_batch_processor, val_data_loader

val_nuscenes_metric = dict(
    type="NuscenesMetric",
    data_root=meta_rootdir,
    use_lidar=True,
    trans_lidar_dim=True,
    trans_lidar_rot=False,
    use_ddp=False,
    lidar_key="sensor2ego",
    version="v1.0-trainval",
    classes=CLASSES,
    save_prefix="./WORKSPACE/results" + task_name,
)


def update_loss(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


loss_show_update = dict(
    type="MetricUpdater",
    metric_update_func=update_loss,
    step_log_freq=lossshow,
    epoch_log_freq=1,
    log_prefix="loss_" + task_name,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=500,
    batch_size=batch_size_per_gpu,
)


def update_metric(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(batch, model_outs)


val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=10000,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=None,
    init_with_train_model=False,
    val_interval=num_steps_per_epoch * 5,
    interval_by="step",
    val_on_train_end=True,
    log_interval=500,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    interval_by="step",
    save_interval=num_steps_per_epoch * 5,
    strict_match=False,
    mode="max",
)

grad_callback = dict(
    type="GradClip",
    max_norm=25,
    norm_type=2,
)
