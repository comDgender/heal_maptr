"""Training / validation callbacks and metrics."""

from common import (
    CLASSES,
    ckpt_dir,
    ckpt_save_interval,
    lossshow,
    meta_rootdir,
    task_name,
    training_step,
    val_interval,
)
from data import val_batch_processor, val_data_loader
from model import test_model

val_nuscenes_metric = dict(
    type="NuscenesMetric",
    data_root=meta_rootdir,
    version="v1.0-trainval",
    use_lidar=False,
    classes=CLASSES,
    save_prefix="./WORKSPACE/results" + task_name,
    use_ddp=False,
)


def update_loss(metrics, batch, model_outs):
    for model_out in model_outs:
        for metric in metrics:
            metric.update(model_out)


loss_show_update = dict(
    type="MetricUpdater",
    metric_update_func=update_loss,
    step_log_freq=lossshow,
    epoch_log_freq=1,
    log_prefix="",
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=500,
)


def update_val_metric(metrics, batch, model_outs):
    # Convert one hot to inde
    preds = model_outs
    metrci_gt = {}
    metrci_gt["meta"] = batch["seq_meta"][0]["meta"]
    metrics[0].update(metrci_gt, preds)


val_metric_updater = dict(
    type="MetricUpdater",
    # metrics=[val_nuscenes_metric],
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
)
ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    save_interval=ckpt_save_interval,
    strict_match=True,
    mode="max",
    # best_refer_metric=val_nuscenes_metric,
    monitor_metric_key="NDS",
)
grad_callback = dict(
    type="GradScale",
    module_and_scale=[],
    clip_grad_norm=35,
    clip_norm_type=2,
)
