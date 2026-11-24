"""Training callbacks and metrics for BevFusion multisensor multitask."""

from common import (
    camera_input,
    ckpt_dir,
    class_names,
    lidar_input,
    log_loss_show,
    meta_rootdir,
    occ3d_seg_class,
    task_name,
    training_step,
    use_bev_head,
    use_occ_head,
    val_interval,
)
from data import val_batch_processor, val_data_loader


def update_metric(metrics, batch, model_outs):
    idx = 0
    if use_bev_head:
        metric_gt = {"meta": batch["seq_meta"][0]["meta"]}
        metrics[idx].update(metric_gt, model_outs[idx])
        idx += 1
    if use_occ_head:
        gt_semantics = batch["seq_meta"][0]["gt_occ_info"]["voxel_semantics"][
            0
        ].squeeze()
        semantics_pred = model_outs[idx][0].squeeze()

        if lidar_input and camera_input:
            lidar_mask = batch["seq_meta"][0]["gt_occ_info"]["mask_lidar"][
                0
            ].squeeze()
            camera_mask = batch["seq_meta"][0]["gt_occ_info"]["mask_camera"][
                0
            ].squeeze()
            mask = lidar_mask | camera_mask
        elif lidar_input:
            mask = batch["seq_meta"][0]["gt_occ_info"]["mask_lidar"][
                0
            ].squeeze()
        elif camera_input:
            mask = batch["seq_meta"][0]["gt_occ_info"]["mask_camera"][
                0
            ].squeeze()
        masked_semantics_gt = gt_semantics[mask]
        masked_semantics_pred = semantics_pred[mask]
        results = {
            "label": masked_semantics_gt.reshape(-1),
            "preds": masked_semantics_pred.reshape(-1),
        }
        metrics[idx].update(**results)


def update_loss(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


bev_val_nuscenes_metric = dict(
    type="NuscenesMetric",
    data_root=meta_rootdir,
    version="v1.0-trainval",
    use_lidar=True,
    classes=class_names,
    save_prefix="./WORKSPACE/results" + task_name,
    lidar_key="sensor2ego",
    trans_lidar_dim=True,
    trans_lidar_rot=False,
)
occ_val_nuscenes_metric = dict(
    type="MeanIOU",
    seg_class=occ3d_seg_class,
    ignore_index=17,
)

val_metrics = []
if use_bev_head:
    val_metrics.append(bev_val_nuscenes_metric)
if use_occ_head:
    val_metrics.append(occ_val_nuscenes_metric)

val_metric_updater = dict(
    type="MetricUpdater",
    metric_update_func=update_metric,
    step_log_freq=10000,
    epoch_log_freq=1,
    log_prefix="Validation " + task_name,
)

loss_show_update = dict(
    type="MetricUpdater",
    metric_update_func=update_loss,
    step_log_freq=log_loss_show,
    epoch_log_freq=1,
    log_prefix="loss_" + task_name,
)

stat_callback = dict(
    type="StatsMonitor",
    log_freq=log_loss_show,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    strict_match=True,
    save_interval=1,
    mode=None,
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=None,
    val_on_train_end=False,
    val_interval=val_interval,
    log_interval=200,
)

grad_callback = dict(
    type="GradScale",
    module_and_scale=[],
    clip_grad_norm=35,
    clip_norm_type=2,
)
