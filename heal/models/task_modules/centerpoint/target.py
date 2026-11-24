# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import List

import torch
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.apply_func import multi_apply
from torch import nn

from heal.core.center_utils import draw_umich_gaussian_torch, gaussian_radius

__all__ = [
    "CenterPointLidarTarget",
    "CenterPointTargetForLidarBBOX",
    "clip_sigmoid",
]


def clip_sigmoid(x, eps=1e-4):
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y


class _CenterPointLidarTarget(nn.Module):
    """Generate CenterPoint targets for lidar-format 3D boxes."""

    def __init__(
        self,
        grid_size: List[int],
        voxel_size: List[float],
        point_cloud_range: List[float],
        tasks: List[dict],
        dense_reg: int = 1,
        max_objs: int = 500,
        gaussian_overlap: float = 0.1,
        min_radius: int = 2,
        out_size_factor: int = 4,
        norm_bbox: bool = True,
        with_velocity: bool = False,
    ):
        super().__init__()
        self.class_names = [t["class_names"] for t in tasks]
        self.num_classes = [len(t["class_names"]) for t in tasks]
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.dense_reg = dense_reg
        self.max_objs = max_objs
        self.out_size_factor = out_size_factor
        self.gaussian_overlap = gaussian_overlap
        self.min_radius = min_radius
        self.norm_bbox = norm_bbox
        self.with_velocity = with_velocity

    def _task_label_masks(self, gt_labels: torch.Tensor) -> List[List]:
        grouped_masks = []
        global_cls_bias = 0
        for names in self.class_names:
            grouped_masks.append(
                [
                    torch.where(
                        gt_labels == names.index(name) + 1 + global_cls_bias
                    )
                    for name in names
                ]
            )
            global_cls_bias += len(names)
        return grouped_masks

    def _partition_gt_by_tasks(
        self,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        grouped_masks: List[List],
    ):
        boxes_per_task = []
        labels_per_task = []
        task_label_base = 0
        device = gt_labels.device
        for masks in grouped_masks:
            cls_boxes = [gt_boxes[m] for m in masks]
            cls_labels = [gt_labels[m] - task_label_base for m in masks]
            boxes_per_task.append(torch.cat(cls_boxes, dim=0).to(device))
            labels_per_task.append(torch.cat(cls_labels).long().to(device))
            task_label_base += len(masks)
        return boxes_per_task, labels_per_task

    def _allocate_task_buffers(
        self,
        task_idx: int,
        ref_tensor: torch.Tensor,
        map_size: torch.Tensor,
        cap: int,
    ):
        n_cls = len(self.class_names[task_idx])
        heatmap = ref_tensor.new_zeros((n_cls, map_size[1], map_size[0]))
        anno_dim = 10 if self.with_velocity else 8
        anno_box = ref_tensor.new_zeros((cap, anno_dim), dtype=torch.float32)
        slot_idx = ref_tensor.new_zeros((cap,), dtype=torch.int64)
        slot_mask = ref_tensor.new_zeros((cap,), dtype=torch.uint8)
        return heatmap, anno_box, slot_idx, slot_mask

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        max_objs = self.max_objs * self.dense_reg
        grid_size = torch.tensor(self.grid_size)
        pc_range = torch.tensor(self.point_cloud_range)
        voxel_size = torch.tensor(self.voxel_size)
        feature_map_size = grid_size[:2] // self.out_size_factor

        grouped_masks = self._task_label_masks(gt_labels_3d)
        task_boxes, task_classes = self._partition_gt_by_tasks(
            gt_bboxes_3d, gt_labels_3d, grouped_masks
        )

        heatmaps, anno_boxes, inds, masks = [], [], [], []
        for task_idx in range(len(self.num_classes)):
            heatmap, anno_box, ind, mask = self._allocate_task_buffers(
                task_idx, gt_bboxes_3d, feature_map_size, max_objs
            )
            obj_limit = min(task_boxes[task_idx].shape[0], max_objs)

            for k in range(obj_limit):
                cls_id = task_classes[task_idx][k] - 1
                width = task_boxes[task_idx][k][3]
                length = task_boxes[task_idx][k][4]
                width = width / voxel_size[0] / self.out_size_factor
                length = length / voxel_size[1] / self.out_size_factor
                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (length.item(), width.item()),
                        min_overlap=self.gaussian_overlap,
                    )
                    radius = max(self.min_radius, int(radius))
                    x, y, z = task_boxes[task_idx][k][0:3]
                    center = torch.stack(
                        (
                            (x - pc_range[0])
                            / voxel_size[0]
                            / self.out_size_factor,
                            (y - pc_range[1])
                            / voxel_size[1]
                            / self.out_size_factor,
                        )
                    ).to(dtype=torch.float32)
                    center_int = center.to(torch.int32)
                    if not (
                        0 <= center_int[0] < feature_map_size[0]
                        and 0 <= center_int[1] < feature_map_size[1]
                    ):
                        continue
                    draw_umich_gaussian_torch(
                        heatmap[cls_id], center_int, radius
                    )

                    cx, cy = center_int[0], center_int[1]
                    ind[k] = cy * feature_map_size[0] + cx
                    mask[k] = 1

                    rot = task_boxes[task_idx][k][8]
                    box_dim = task_boxes[task_idx][k][3:6]
                    if self.norm_bbox:
                        box_dim = box_dim.log()

                    anno_parts = [
                        center - center_int.to(center.dtype),
                        z.unsqueeze(0),
                        box_dim,
                        torch.sin(rot).unsqueeze(0),
                        torch.cos(rot).unsqueeze(0),
                    ]
                    if self.with_velocity:
                        vx, vy = task_boxes[task_idx][k][6:8]
                        anno_parts.extend((vx.unsqueeze(0), vy.unsqueeze(0)))
                    anno_box[k] = torch.cat(anno_parts)

            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)
        return heatmaps, anno_boxes, inds, masks

    def forward(self, gt_bboxes_3d, gt_labels_3d):
        heatmaps, anno_boxes, inds, masks = multi_apply(
            self.get_targets_single, gt_bboxes_3d, gt_labels_3d
        )
        heatmaps = list(map(list, zip(*heatmaps)))
        heatmaps = [torch.stack(hms_) for hms_ in heatmaps]
        anno_boxes = list(map(list, zip(*anno_boxes)))
        anno_boxes = [torch.stack(anno_boxes_) for anno_boxes_ in anno_boxes]
        inds = list(map(list, zip(*inds)))
        inds = [torch.stack(inds_) for inds_ in inds]
        masks = list(map(list, zip(*masks)))
        masks = [torch.stack(masks_) for masks_ in masks]
        return heatmaps, anno_boxes, inds, masks


@OBJECT_REGISTRY.register
class CenterPointLidarTarget(_CenterPointLidarTarget):
    """Generate CenterPoint targets for lidar-only detection."""


@OBJECT_REGISTRY.register
class CenterPointTargetForLidarBBOX(_CenterPointLidarTarget):
    """CenterPoint target generator for lidar bbox labels in BEV pipeline."""

    def forward(self, label, preds):
        gt_bboxes_3d = []
        gt_labels_3d = []
        for single_label in label:
            if len(single_label) != 0:
                bboxes = torch.tensor(single_label[..., :9])
                locs = bboxes[..., 0:3]
                dims = bboxes[..., 3:6]
                rot = bboxes[..., 6:7]
                vel = bboxes[..., 7:9]
                bboxes = torch.cat([locs, dims, vel, rot], dim=-1)
                gt_bboxes_3d.append(
                    torch.tensor(bboxes).to(device=preds[0]["heatmap"].device)
                )
                gt_labels_3d.append(
                    torch.tensor(single_label[..., 9] + 1).to(
                        device=preds[0]["heatmap"].device
                    )
                )
            else:
                gt_bboxes_3d.append(
                    torch.zeros([0, 9]).to(device=preds[0]["heatmap"].device)
                )
                gt_labels_3d.append(
                    torch.zeros([0]).to(device=preds[0]["heatmap"].device)
                )
        heatmaps, anno_boxes, inds, masks = super().forward(
            gt_bboxes_3d, gt_labels_3d
        )
        return [
            {
                "task_name": "bbox",
                "heatmaps": heatmaps,
                "anno_boxes": anno_boxes,
                "inds": inds,
                "masks": masks,
                "preds_dicts": preds,
            }
        ]
