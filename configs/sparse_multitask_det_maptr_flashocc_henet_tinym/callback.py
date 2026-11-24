"""Training / validation callbacks and metrics."""

from common import (
    batch_size_per_gpu,
    ckpt_dir,
    enable_det_head,
    enable_occ_head,
    enable_om_head,
    eval_use_same_gt_sample_num_flag,
    fixed_ptsnum_per_gt_line,
    map_classes,
    map_point_cloud_range,
    meta_rootdir,
    num_steps_per_epoch,
    occ3d_seg_class,
    task_name,
    training_step,
    val_interval,
)
from data import val_batch_processor, val_data_loader

val_nuscenes_metric = dict(
    type="NuscenesMetric",
    data_root=meta_rootdir,
    use_lidar=True,
    trans_lidar_dim=True,
    trans_lidar_rot=False,
    use_ddp=True,
    lidar_key="sensor2ego",
    version="v1.0-trainval",
    save_prefix="./WORKSPACE/results" + task_name,
)
val_map_metric = dict(
    type="NuscenesMapMetric",
    classes=map_classes,
    save_prefix="./WORKSPACE/results" + task_name,
    fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
    pc_range=map_point_cloud_range,
    metric="chamfer",
    eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
)
val_occ_metric = dict(
    type="MeanIOU",
    seg_class=occ3d_seg_class,
    ignore_index=17,
)

val_metrics = []
if enable_det_head:
    val_metrics.append(val_nuscenes_metric)
if enable_om_head:
    val_metrics.append(val_map_metric)
if enable_occ_head:
    val_metrics.append(val_occ_metric)

# Stage2 trains only om(map) + occ (det + backbone frozen), so its val
# metrics are the map + occ ones.
val_metrics_stage2 = []
if enable_om_head:
    val_metrics_stage2.append(val_map_metric)
if enable_occ_head:
    val_metrics_stage2.append(val_occ_metric)


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

stat_callback = dict(
    type="StatsMonitor",
    log_freq=100,
    batch_size=batch_size_per_gpu,
)


def update_val_metric(metrics, batch, model_outs):
    for metric in metrics:
        if metric.name == "MeanIOU":
            k = "occ"
        elif metric.name == "NuscenesMetric":
            k = "det"
        elif metric.name == "NuscenesMapMetric":
            k = "om"
        else:
            continue
        if k in ("det", "om"):
            metric.update(batch, model_outs[k])
        else:
            gt_semantics = batch["gt_occ_info"]["voxel_semantics"][0].squeeze()
            semantics_pred = model_outs[k][0].squeeze()
            mask = batch["gt_occ_info"]["mask_camera"][0].squeeze()
            masked_semantics_gt = gt_semantics[mask]
            masked_semantics_pred = semantics_pred[mask]
            metric.update(
                **{
                    "label": masked_semantics_gt.reshape(-1),
                    "preds": masked_semantics_pred.reshape(-1),
                }
            )


val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_val_metric,
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
    val_interval=val_interval,
    interval_by="step",
    val_on_train_end=True,
    log_interval=500,
)

_ckpt_callback_common = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    interval_by="step",
    save_interval=num_steps_per_epoch * 10,
    strict_match=False,
)

ckpt_callback = dict(
    **_ckpt_callback_common,
    mode=None,
)

ckpt_callback_stage1 = dict(
    **_ckpt_callback_common,
    mode="max",
    monitor_metric_key="NDS",
)

ckpt_callback_stage2 = dict(
    **_ckpt_callback_common,
    mode="max",
    monitor_metric_key="MAP",
)

# float_merge stage: save the merged complete model (stage1 all + stage2
# om/occ). mode=None (no best tracking — merge produces a single result saved
# as -last). name_prefix is fixed to "float_merged-" (overriding the common
# `training_step + "-"` prefix) so the downstream `float_merged_checkpoint`
# path (qconfig.py) resolves the stage output.
ckpt_callback_merge = dict(
    _ckpt_callback_common,
    name_prefix="float_merged-",
    mode=None,
)

grad_callback = dict(
    type="GradClip",
    max_norm=25,
    norm_type=2,
)

# Stage2 trains only om(map) + occ with the backbone frozen; det is absent
# from the stage2 model (merged back from stage1 after training).
freeze_module_callback = dict(
    type="FreezeModule",
    modules=[["backbone"]],
    step_or_epoch=[0],
    update_by="step",
)
