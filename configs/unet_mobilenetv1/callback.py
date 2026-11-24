import torch
import torch.nn.functional as F
from common import ckpt_dir, num_classes, task_name, training_step
from data import val_batch_processor, val_data_loader
from hat_engine.metrics.mean_iou import MeanIOU
from model import deploy_model

val_miou_metric = MeanIOU(seg_class=[str(i) for i in range(num_classes)])


def update_val_metric(metrics, batch, model_outs):
    target = batch["gt_seg"]
    ignore_points = target.sum(dim=1) == 0
    target = target.argmax(dim=1)
    target[ignore_points] = 255
    preds = model_outs[0]
    preds = (
        F.interpolate(preds.float(), scale_factor=4, mode="nearest")
        .to(dtype=torch.uint8)
        .squeeze(1)
    )
    for metric in metrics:
        metric.update(target, preds)


val_metric_updater = dict(
    type="MetricUpdater",
    metrics=[val_miou_metric],
    metric_update_func=update_val_metric,
    step_log_freq=100,
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
    save_interval=1,
    strict_match=True,
    mode="max",
    best_refer_metric=val_miou_metric,
)

float_val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=deploy_model,
)

calibration_val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=deploy_model,
)

qat_val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=deploy_model,
)
