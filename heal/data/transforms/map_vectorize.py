# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
"""Map vector GT transforms for multitask MergeDataset pipelines."""

from typing import Mapping, Optional, Sequence, Tuple

from hat_engine.registry import OBJECT_REGISTRY

from heal.data.datasets.nuscenes.nuscenes_map_dataset import (
    NuscenesMapDataset,
    VectorizedLocalMap,
    apply_online_vectormap,
)

__all__ = ["VectorizeMapGT"]


@OBJECT_REGISTRY.register
class VectorizeMapGT(object):
    """Generate map vector GT after geometric augmentation.

    In multitask pipelines, ``MergeDataset`` applies geometric transforms
    (resize/crop/flip/rotate/BevBBoxRotation) after child datasets are merged.
    Map GT must be built from the augmented ``lidar2global`` and image shapes,
    matching HAT's ``transforms -> vectormap_pipeline`` order.
    """

    def __init__(
        self,
        map_path: str,
        pc_range: Sequence[float],
        map_classes: Sequence[str],
        fixed_ptsnum_per_line: int,
        padding_value: int,
        bev_size: Tuple[int, int],
        aux_seg: Optional[dict] = None,
        use_lidar_gt: bool = True,
        sd_map_path: Optional[str] = None,
    ):
        map_class_names = NuscenesMapDataset.get_map_classes(map_classes)
        patch_h = pc_range[4] - pc_range[1]
        patch_w = pc_range[3] - pc_range[0]
        self.use_lidar_gt = use_lidar_gt
        self.aux_seg = aux_seg
        self.vector_map = VectorizedLocalMap(
            map_path,
            patch_size=(patch_h, patch_w),
            map_classes=map_class_names,
            fixed_ptsnum_per_line=fixed_ptsnum_per_line,
            padding_value=padding_value,
            canvas_size=bev_size,
            aux_seg=aux_seg,
            sd_map_path=sd_map_path,
        )

    def __call__(self, data: Mapping):
        return apply_online_vectormap(
            data,
            self.vector_map,
            use_lidar_gt=self.use_lidar_gt,
            aux_seg=self.aux_seg,
        )

    def __repr__(self):
        return "VectorizeMapGT"
