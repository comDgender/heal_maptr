# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import torch
from common import (
    camera_input,
    ckpt_dir,
    discrete_map_class_names,
    dynamic_objects_class_names,
    dynamic_objects_detection_mapping,
    eval_use_same_gt_sample_num_flag,
    fixed_ptsnum_per_gt_line,
    lidar_input,
    log_loss_show,
    map_classes,
    map_point_cloud_range,
    meta_rootdir,
    num_steps_per_epoch,
    occ3d_seg_class,
    static_objects_class_names,
    static_objects_detection_mapping,
    task_name,
    training_step,
    use_discrete_map_head,
    use_dynamic_det_head,
    use_maptr_head,
    use_occ_head,
    use_static_objects_det_head,
)
from data import val_batch_processor, val_data_loader


def update_metric(metrics, batch, model_outs):
    metric_name_to_task = {
        "dynamic_objects_det_metric": "dynamic_objects_det",
        "dynamic_objects_tracking_metric": "dynamic_objects_det",
        "static_objects_det_metric": "static_objects_det",
        "occ_metric": "occ",
        "maptr_metric": "maptr",
        "discrete_map_metric": "discrete_map",
    }
    for metric in metrics:
        if metric.name in [
            "dynamic_objects_det_metric",
            "dynamic_objects_tracking_metric",
            "static_objects_det_metric",
        ]:
            metric_gt = {"meta": batch["seq_meta"][0]["meta"]}
            metric.update(
                metric_gt, model_outs[metric_name_to_task[metric.name]]
            )
        elif metric.name == "maptr_metric":
            metric.update(batch, model_outs[metric_name_to_task[metric.name]])
        elif metric.name == "occ_metric":
            # Float/calib: OccDecoder returns [occ_pre], occ_pre [B,Dx,Dy,Dz].
            # BevMultitaskIrInfer (hbir/hbm): returns bare argmax tensor
            # [B,Dx,Dy,Dz] (or logits [B,Dx,Dy,Dz,C]).
            gt_occ = batch["seq_meta"][0]["gt_occ_info"]
            occ_out = model_outs[metric_name_to_task[metric.name]]
            if isinstance(occ_out, (list, tuple)):
                semantics_pred_all = occ_out[0]
            else:
                semantics_pred_all = occ_out
            if (
                isinstance(semantics_pred_all, torch.Tensor)
                and semantics_pred_all.dim() == 5
            ):
                semantics_pred_all = semantics_pred_all.argmax(-1)
            batch_size = semantics_pred_all.shape[0]
            for b_idx in range(batch_size):
                gt_semantics = gt_occ["voxel_semantics"][b_idx].squeeze()
                semantics_pred = semantics_pred_all[b_idx].squeeze()
                if lidar_input and camera_input:
                    mask = (
                        gt_occ["mask_lidar"][b_idx].squeeze().bool()
                        | gt_occ["mask_camera"][b_idx].squeeze().bool()
                    )
                elif lidar_input:
                    mask = gt_occ["mask_lidar"][b_idx].squeeze().bool()
                elif camera_input:
                    mask = gt_occ["mask_camera"][b_idx].squeeze().bool()
                else:
                    raise ValueError(
                        "occ_metric requires lidar_input and/or camera_input"
                    )
                metric.update(
                    label=gt_semantics[mask].reshape(-1),
                    preds=semantics_pred[mask].reshape(-1),
                )
        elif metric.name == "discrete_map_metric":
            gt_info = batch["seq_meta"][0]["gt_discrete_map_info"]
            if "lidar_bboxes_labels" in gt_info:
                bbox_key = "lidar_bboxes_labels"
            else:
                bbox_key = "ego_bboxes_labels"
            preds = model_outs[metric_name_to_task[metric.name]]
            gts = gt_info[bbox_key]
            for cur_batch_pred, cur_batch_gt in zip(preds, gts):
                pred, gt = {}, {}
                pred["boxes"] = cur_batch_pred[:, :5]
                pred["scores"] = cur_batch_pred[:, -2]
                pred["labels"] = cur_batch_pred[:, -1]
                if len(cur_batch_gt) == 0:
                    gt["boxes"], gt["labels"] = torch.tensor([]), torch.tensor(
                        []
                    )
                else:
                    gt["boxes"] = cur_batch_gt[:, :5]
                    gt["labels"] = cur_batch_gt[:, -1]
                metric.update(pred, gt)
        else:
            raise ValueError(f"Does not support metric:{metric.name}")


def update_loss(metrics, batch, model_outs):
    for metric in metrics:
        metric.update(model_outs)


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
    batch_size=4,
)

ckpt_callback = dict(
    type="Checkpoint",
    save_dir=ckpt_dir,
    name_prefix=training_step + "-",
    interval_by="step",
    save_interval=num_steps_per_epoch * 20,
    strict_match=False,
    mode=None,
)

val_callback = dict(
    type="Validation",
    data_loader=val_data_loader,
    batch_processor=val_batch_processor,
    callbacks=[val_metric_updater],
    val_model=None,
    val_on_train_end=True,
    log_interval=200,
    interval_by="step",
    val_interval=num_steps_per_epoch * 20,
)

grad_callback = dict(
    type="GradScale",
    module_and_scale=[],
    clip_grad_norm=35,
    clip_norm_type=2,
)

dynamic_objects_nuscenes_metric = dict(
    type="NuscenesMetric",
    name="dynamic_objects_det_metric",
    data_root=meta_rootdir,
    version="v1.0-trainval",
    use_lidar=True,
    classes=dynamic_objects_class_names,
    save_prefix="./WORKSPACE/results" + task_name,
    lidar_key="sensor2ego",
    trans_lidar_dim=True,
    trans_lidar_rot=False,
    detection_mapping=dynamic_objects_detection_mapping,
)

static_objects_nuscenes_metric = dict(
    type="NuscenesMetric",
    name="static_objects_det_metric",
    data_root=meta_rootdir,
    version="v1.0-trainval",
    use_lidar=True,
    classes=static_objects_class_names,
    save_prefix="./WORKSPACE/results" + task_name,
    lidar_key="sensor2ego",
    trans_lidar_dim=True,
    trans_lidar_rot=False,
    detection_mapping=static_objects_detection_mapping,
)

occ_val_nuscenes_metric = dict(
    type="MeanIOU",
    name="occ_metric",
    seg_class=occ3d_seg_class,
    ignore_index=17,
)

maptr_val_nuscenes_metric = dict(
    type="NuscenesMapMetric",
    name="maptr_metric",
    classes=map_classes,
    save_prefix="./WORKSPACE/results" + "_maptr_" + task_name,
    fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
    pc_range=map_point_cloud_range,
    metric="chamfer",
    eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
)

discrete_map_val_metrics = dict(
    type="BEV2DNDSMetric",
    name="discrete_map_metric",
    class_names=discrete_map_class_names,
    dist_ths=[0.5, 1.0, 2.0, 4.0],
    dist_th_tp=2.0,
    weights=dict(ap=0.5, ate=0.2, ase=0.15, aoe=0.15),
    verbose=True,
    period=2 * torch.pi,
)

val_metrics = []
if use_dynamic_det_head:
    val_metrics.append(dynamic_objects_nuscenes_metric)
if use_static_objects_det_head:
    val_metrics.append(static_objects_nuscenes_metric)
if use_occ_head:
    val_metrics.append(occ_val_nuscenes_metric)
if use_discrete_map_head:
    val_metrics.append(discrete_map_val_metrics)
if use_maptr_head:
    val_metrics.append(maptr_val_nuscenes_metric)

# stage1: dynamic + static + occ + discrete map
stage1_val_metrics = [
    dynamic_objects_nuscenes_metric,
    static_objects_nuscenes_metric,
    occ_val_nuscenes_metric,
    discrete_map_val_metrics,
]
# stage2: maptr only
stage2_val_metrics = [
    maptr_val_nuscenes_metric,
]
# stage3: all tasks
stage3_val_metrics = [
    dynamic_objects_nuscenes_metric,
    static_objects_nuscenes_metric,
    occ_val_nuscenes_metric,
    maptr_val_nuscenes_metric,
    discrete_map_val_metrics,
]

# freeze shared parts (backbone, neck, view_transformer, fuse) in stage2.
freeze_model_callback = dict(
    type="FreezeModule",
    modules=[
        [
            "camera_net.backbone",
            "camera_net.neck",
            "camera_net.view_transformer",
            "fuse_module",
        ]
    ],
    step_or_epoch=[0],
    update_by="step",
    only_batchnorm=False,
)

freeze_model_callback = dict(
    type="FreezeModule",
    modules=[
        [
            "camera_net.backbone",
            "camera_net.neck",
            "camera_net.view_transformer",
            "fuse_module",
        ]
    ],
    step_or_epoch=[0],
    update_by="step",
    only_batchnorm=False,
)
