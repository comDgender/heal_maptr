# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from nuscenes.eval.common.utils import Quaternion
except ImportError:
    Quaternion = None

from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.package_helper import require_packages

from heal.data.datasets.nuscenes.nuscenes_dataset import NuscenesBevDataset
from heal.data.datasets.nuscenes.nuscenes_map_dataset import (
    apply_online_vectormap,
    ensure_lidar2global,
)
from heal.data.datasets.nuscenes.nuscenes_map_geometry import (
    VectorizedLocalMap,
)


def apply_online_discrete_map(
    example,
    vector_det,
    discrete_map_classes,
    with_yaw=True,
    use_lidar_gt=True,
):
    """Build discrete-map 2D OBBs from the current (possibly augmented) pose.

    Matches HAT ``gen_bboxes_from_map`` after ``BevBBoxRotation``.
    """
    example = ensure_lidar2global(example)
    lidar2global = example["lidar2global"]
    translation = list(lidar2global[:3, 3])
    rotation = list(Quaternion(matrix=lidar2global).q)

    if not use_lidar_gt:
        translation = example["meta"]["ego2global_translation"]
        rotation = example["meta"]["ego2global_rotation"]

    location = example["meta"]["location"]
    anns_results = vector_det.gen_polygon_elements_with_bbox(
        location,
        translation,
        rotation,
        discrete_map_classes,
        with_yaw,
    )
    bboxes_labels = []
    for label_id, label in enumerate(anns_results.keys()):
        bboxes = anns_results[label]
        for bbox in bboxes:
            bboxes_labels.append(
                torch.cat([bbox, torch.Tensor([label_id])], dim=0)
            )
    if len(bboxes_labels) > 0:
        bboxes_labels = torch.vstack(bboxes_labels)
    box_key = "lidar_bboxes_labels" if use_lidar_gt else "ego_bboxes_labels"
    example["gt_discrete_map_info"] = {box_key: bboxes_labels}
    return example


def apply_split_dynamic_static_objects(example, use_lidar_gt=True):
    """Split 3D boxes into nested dynamic/static dicts (HAT order)."""
    key = "lidar_bboxes_labels" if use_lidar_gt else "ego_bboxes_labels"
    instance_ids = torch.tensor(example["instance_ids"])
    bbox = torch.tensor(example[key])
    example.pop(key)
    example.pop("instance_ids")
    if bbox.shape[0] == 0:
        example.update(
            {
                "gt_moving_objects_info": {
                    key: [],
                    "instance_ids": [],
                },
                "gt_static_objects_info": {
                    key: [],
                    "instance_ids": [],
                },
            }
        )
        return example

    mask_dynamic = bbox[:, -1] <= 7
    mask_static = bbox[:, -1] >= 8

    bbox_dynamic = bbox[mask_dynamic]
    instance_ids_dynamic = instance_ids[mask_dynamic].tolist()
    if bbox_dynamic.shape[0] == 0:
        bbox_dynamic = []

    bbox_static = bbox[mask_static]
    bbox_static[:, -1] = bbox_static[:, -1] - 8
    instance_ids_static = instance_ids[mask_static].tolist()
    if bbox_static.shape[0] == 0:
        bbox_static = []

    example.update(
        {
            "gt_moving_objects_info": {
                key: bbox_dynamic,
                "instance_ids": instance_ids_dynamic,
            },
            "gt_static_objects_info": {
                key: bbox_static,
                "instance_ids": instance_ids_static,
            },
        }
    )
    return example


@OBJECT_REGISTRY.register
class NuscenesMultitaskDataset(NuscenesBevDataset):

    MAPCLASSES: Tuple[str] = ("divider",)

    @require_packages("nuscenes")
    def __init__(
        self,
        queue_length: int = 1,
        fixed_ptsnum_per_line: int = -1,
        padding_value: int = -10000,
        map_classes: Optional[Tuple[str]] = None,
        map_path: Optional[str] = None,
        aux_seg: Optional[any] = None,
        test_mode: Optional[bool] = False,
        filter_empty_gt: Optional[bool] = True,
        use_lidar_gt: bool = True,
        sd_map_path: Optional[str] = None,
        discrete_map_classes: Optional[Tuple[str]] = None,
        with_bev_bboxes: bool = True,
        with_ego_bboxes: bool = False,
        with_lidar_bboxes: bool = False,
        with_bev_mask: bool = True,
        max_dist: Optional[float] = 55.0,
        discrete_map_bev_size: Optional[Tuple] = None,
        bev_size: Optional[Tuple] = None,
        bev_range: Optional[Tuple] = None,
        map_size: Optional[Tuple] = None,
        with_yaw: bool = False,
        split_dynamic_det: bool = False,
        **kwargs,
    ):
        super(NuscenesMultitaskDataset, self).__init__(
            with_bev_bboxes=with_bev_bboxes,
            with_ego_bboxes=with_ego_bboxes,
            with_lidar_bboxes=with_lidar_bboxes,
            with_bev_mask=with_bev_mask,
            max_dist=max_dist,
            map_path=map_path,
            bev_range=bev_range,
            map_size=map_size,
            **kwargs,
        )

        self.queue_length = queue_length
        self.bev_size = bev_size

        self.MAPCLASSES = self.get_map_classes(map_classes)
        self.NUM_MAPCLASSES = len(self.MAPCLASSES)

        self.padding_value = padding_value
        self.fixed_num = fixed_ptsnum_per_line
        self.aux_seg = aux_seg
        map_patch_size = (map_size[1] * 2, map_size[0] * 2)
        map_canvas_size = (
            int(map_size[1] * 2 / map_size[2]),
            int(map_size[0] * 2 / map_size[2]),
        )
        self.vector_map = VectorizedLocalMap(
            map_path,
            patch_size=map_patch_size,
            map_classes=self.MAPCLASSES,
            fixed_ptsnum_per_line=fixed_ptsnum_per_line,
            padding_value=self.padding_value,
            canvas_size=map_canvas_size,
            aux_seg=aux_seg,
            sd_map_path=sd_map_path,
        )
        self.filter_empty_gt = filter_empty_gt
        self.test_mode = test_mode
        self.flag = np.zeros(len(self), dtype=np.uint8)
        self.use_lidar_gt = use_lidar_gt
        self.split_dynamic_det = split_dynamic_det

        if discrete_map_classes is not None:
            discrete_map_patch_size = (
                discrete_map_bev_size[1] * 2,
                discrete_map_bev_size[0] * 2,
            )
            self.vector_det = VectorizedLocalMap(
                map_path,
                patch_size=discrete_map_patch_size,
            )
        self.discrete_map_classes = discrete_map_classes
        self.with_yaw = with_yaw

    def gen_bboxes_from_map(self, example):
        return apply_online_discrete_map(
            example,
            self.vector_det,
            self.discrete_map_classes,
            with_yaw=self.with_yaw,
            use_lidar_gt=self.use_lidar_gt,
        )

    def split_dynamic_static_objects(self, example):
        return apply_split_dynamic_static_objects(
            example, use_lidar_gt=self.use_lidar_gt
        )

    @classmethod
    def get_map_classes(
        cls, map_classes: Optional[Sequence[str]] = None
    ) -> List[str]:
        """Get class names of current dataset.

        Args:
            map_classes : If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Return:
            A list of class names.
        """
        if map_classes is None:
            return cls.MAPCLASSES

        if isinstance(map_classes, (tuple, list)):
            class_names = map_classes
        else:
            raise ValueError(
                f"Unsupported type {type(map_classes)} of map classes."
            )

        return class_names

    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def _fetch_item(self, idx, aug_cfg=None):
        if aug_cfg is not None:
            return {"idx": idx, "aug": aug_cfg}
        return idx

    def prepare_train_data(self, idx, aug_cfg=None):
        start = idx
        end = idx - self.queue_length
        queue_list = list(range(start, end, -1))

        seq_data = []
        for i in queue_list:
            idx = max(i, 0)
            # Parent applies image/geo transforms (incl. BevBBoxRotation).
            # Then map GT / split run on the augmented lidar2global (HAT).
            example = super().__getitem__(self._fetch_item(idx, aug_cfg))
            example = apply_online_vectormap(
                example,
                self.vector_map,
                use_lidar_gt=self.use_lidar_gt,
                aux_seg=self.aux_seg,
            )
            if self.discrete_map_classes is not None:
                example = self.gen_bboxes_from_map(example)
            if self.split_dynamic_det:
                example = self.split_dynamic_static_objects(example)
            if self.filter_empty_gt and (
                example is None or ~(example["gt_labels_map"] != -1).any()
            ):
                return None

            seq_data.append(example)

        return seq_data

    def prepare_test_data(self, item):
        start = item
        end = item - self.queue_length
        seq_data = []

        for i in range(start, end, -1):
            idx = max(i, 0)
            example = super().__getitem__(idx)
            example = apply_online_vectormap(
                example,
                self.vector_map,
                use_lidar_gt=self.use_lidar_gt,
                aux_seg=self.aux_seg,
            )
            if self.discrete_map_classes is not None:
                example = self.gen_bboxes_from_map(example)
            if self.split_dynamic_det:
                example = self.split_dynamic_static_objects(example)
            seq_data.append(example)
        return seq_data

    def __getitem__(self, item):
        if isinstance(item, dict):
            idx = item["idx"]
            aug_cfg = item["aug"]
        else:
            idx = item
            aug_cfg = None
        if self.test_mode:
            return self.prepare_test_data(idx)[0]
        while True:
            data = self.prepare_train_data(idx, aug_cfg)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data[0]
