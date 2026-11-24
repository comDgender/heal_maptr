# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
"""Map PV segmentation GT transforms."""

from typing import Mapping

from hat_engine.registry import OBJECT_REGISTRY

from heal.data.datasets.nuscenes.nuscenes_map_dataset import (
    rasterize_pv_seg_masks,
)

__all__ = ["RasterizeMapPvSeg"]


@OBJECT_REGISTRY.register
class RasterizeMapPvSeg(object):
    """Rasterize map polylines into PV seg masks after image geometric aug.

    In multitask pipelines, ``MergeDataset`` applies image transforms after the
    map dataset has already built ``gt_pv_seg_mask`` at the original resolution.  # noqa: E501
    This transform regenerates the masks from ``gt_instances`` using the
    current image size and projection matrices.
    """

    def __init__(self, aux_seg: dict, use_lidar_gt: bool = True):
        self.aux_seg = aux_seg
        self.use_lidar_gt = use_lidar_gt

    def __call__(self, data: Mapping):
        if not self.aux_seg or not self.aux_seg.get("pv_seg"):
            return data
        gt_instances = data.get("gt_instances")
        instance_lines = []
        if gt_instances is not None:
            instance_lines = getattr(gt_instances, "instance_list", None) or []
        return rasterize_pv_seg_masks(
            data, instance_lines, self.aux_seg, self.use_lidar_gt
        )

    def __repr__(self):
        return "RasterizeMapPvSeg"
