from common import ckpt_dir, task_name, training_step
from data import val_batch_processor, val_data_loader


def update_metric(metrics, batch, model_outs):
    preds = model_outs
    for metric in metrics:
        metric.update(preds, batch)


def update_loss(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=10000,
    epoch_log_freq=1,
    log_prefix=f"Validation {task_name}",
)

loss_show_update = dict(
    type="MetricUpdater",
    metric_update_func=update_loss,
    step_log_freq=10,
    epoch_log_freq=1,
    log_prefix=f"loss_{task_name}",
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=10,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=f"{training_step}-",
    strict_match=True,
    mode="max",
    monitor_metric_key="mAP_3D_moderate",
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=None,
    val_on_train_end=True,
    val_interval=1,
    log_interval=200,
)
