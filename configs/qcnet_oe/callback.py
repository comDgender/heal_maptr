from common import ckpt_dir, task_name, training_step
from data import val_batch_processor, val_data_loader


def update_loss(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


loss_show_update = dict(
    type="MetricUpdater",
    metric_update_func=update_loss,
    step_log_freq=50,
    epoch_log_freq=1,
    log_prefix="loss_" + task_name,
)


def update_metric(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(**model_outs)


val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=50,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=50,
)

val_metric = dict(type="HitRate")
val_metrics = [
    dict(type="BrierMetric"),
    dict(type="MinADE"),
    dict(type="MinFDE"),
    dict(type="MissRate"),
    dict(type="HitRate"),
]
ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    save_interval=1,
    interval_by="epoch",
    strict_match=False,
    mode="max",
    best_refer_metric=val_metric,
)


val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=None,
    init_with_train_model=False,
    interval_by="epoch",
    val_interval=1,
    val_on_train_end=True,
    log_interval=50,
)
