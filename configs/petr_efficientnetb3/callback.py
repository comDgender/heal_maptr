from typing import Dict, Sequence

from common import ckpt_dir, meta_rootdir, task_name, training_step
from data import val_batch_processor, val_data_loader

val_nuscenes_metric = dict(
    type="NuscenesMetric",
    data_root=meta_rootdir,
    version="v1.0-trainval",
    save_prefix="./WORKSPACE/results" + task_name,
)


def update_val_metric(
    metrics: Sequence[object], batch: Dict, model_outs: Dict
) -> None:
    metrics[0].update(batch, model_outs)


def update_loss(
    metrics: Sequence[object], batch: Dict, model_outs: Dict
) -> None:
    for metric in metrics:
        metric.update(model_outs)


loss_show_update = dict(
    type="MetricUpdater",
    metric_update_func=update_loss,
    step_log_freq=500,
    epoch_log_freq=1,
    log_prefix="loss_" + task_name,
)

val_metric_updater = dict(
    type="MetricUpdater",
    metrics=[val_nuscenes_metric],
    metric_update_func=update_val_metric,
    step_log_freq=50000,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=500,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    save_interval=1,
    strict_match=False,
    mode="max",
    best_refer_metric=val_nuscenes_metric,
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=None,
    init_with_train_model=False,
    val_interval=1,
    val_on_train_end=True,
    log_interval=200,
)
