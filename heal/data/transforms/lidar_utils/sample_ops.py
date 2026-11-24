# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import copy
import pathlib
import pickle
from functools import reduce
from typing import Dict, List

import numpy as np
from hat_engine.registry import OBJECT_REGISTRY, build_from_registry

from heal.core import box_np_ops, box_utils
from heal.data.transforms.lidar_utils.augmentations import (
    box_collision_test,
    noise_per_object_v3_,
    random_crop_frustum,
)
from heal.data.transforms.lidar_utils.filters import mask_points_in_corners
from heal.data.transforms.lidar_utils.samplers import (
    BatchSampler,
    DataBasePreprocessor,
)

__all__ = ["DataBaseSampler"]


def is_array_like(x):
    return isinstance(x, (list, tuple, np.ndarray))


def shape_mergeable(x, expected_shape):
    mergeable = True
    if is_array_like(x) and is_array_like(expected_shape):
        x = np.array(x)
        if len(x.shape) == len(expected_shape):
            for s, s_ex in zip(x.shape, expected_shape):
                if s_ex is not None and s != s_ex:
                    mergeable = False
                    break
    return mergeable


@OBJECT_REGISTRY.register_module
class DataBaseSampler:
    def __init__(
        self,
        enable,
        db_info_path: str,
        sample_groups: Dict,
        db_prep_steps: Dict,
        global_random_rotation_range_per_object: List[float],
        rate: float = 1.0,
        root_path: str = None,
    ):
        """DB_sampler for lidar dataset.

        Sample some object's points to put into current points.

        Args:
            db_info_path : thd db_sample data path.
            sample_groups : the sample class group.
            db_prep_steps :pre-process for filter sample data.
            global_random_rotation_range_per_object.
        """
        prepors = [build_from_registry(c) for c in db_prep_steps]
        db_prepor = DataBasePreprocessor(prepors)
        self.enable = enable
        rate = rate
        groups = sample_groups
        # groups = [dict(g.name_to_max_num) for g in groups]
        info_path = db_info_path
        self.root_path = root_path
        with open(info_path, "rb") as f:
            db_infos = pickle.load(f)
        global_random_rotation_range_per_object = list(
            global_random_rotation_range_per_object
        )
        if len(global_random_rotation_range_per_object) == 0:
            global_random_rotation_range_per_object = None
        global_rot_range = global_random_rotation_range_per_object
        for cls_name, infos in db_infos.items():
            print(f"load {len(infos)} {cls_name} database infos")

        if db_prepor is not None:
            db_infos = db_prepor(db_infos)
            print("After filter database:")
            for cls_name, infos in db_infos.items():
                print(f"load {len(infos)} {cls_name} database infos")

        self.db_infos = db_infos
        self._rate = rate
        self._groups = groups
        self._group_db_infos = {}
        self._group_name_to_names = []
        self._sample_classes = []
        self._sample_max_nums = []
        self._use_group_sampling = any(len(g) > 1 for g in groups)

        if not self._use_group_sampling:
            self._group_db_infos = self.db_infos
            for group_info in groups:
                names = list(group_info.keys())
                self._sample_classes.extend(names)
                self._sample_max_nums.extend(group_info.values())
        else:
            for group_info in groups:
                by_gid = {}
                names = list(group_info.keys())
                joint_name = ", ".join(names)
                self._sample_classes.extend(names)
                self._sample_max_nums.extend(group_info.values())
                self._group_name_to_names.append((joint_name, names))
                for name in names:
                    for item in db_infos[name]:
                        by_gid.setdefault(item["group_id"], []).append(item)
                if joint_name in self._group_db_infos:
                    raise ValueError("group must be unique")
                members = list(by_gid.values())
                self._group_db_infos[joint_name] = members
                combo_counts = {}
                if len(group_info) > 1:
                    for members_i in members:
                        key = ", ".join(sorted(x["name"] for x in members_i))
                        combo_counts[key] = combo_counts.get(key, 0) + 1
                print(combo_counts)

        self._sampler_dict = {
            name: BatchSampler(infos, name)
            for name, infos in self._group_db_infos.items()
        }
        self._enable_global_rot = False
        if global_rot_range is not None:
            if not isinstance(global_rot_range, (list, tuple, np.ndarray)):
                global_rot_range = [-global_rot_range, global_rot_range]
            else:
                assert shape_mergeable(global_rot_range, [2])
            span = abs(global_rot_range[0] - global_rot_range[1])
            self._enable_global_rot = span >= 1e-3
        self._global_rot_range = global_rot_range

    @property
    def use_group_sampling(self):
        return self._use_group_sampling

    def sample_all(
        self,
        root_path: str,
        gt_boxes: np.ndarray,
        gt_names: List,
        num_point_features: int,
        random_crop: bool = False,
        gt_group_ids: List = None,
        calib: np.ndarray = None,
        road_planes: List[int] = None,
    ):
        sampled_num_dict = {}
        sample_num_per_class = []
        for cls_name, max_n in zip(
            self._sample_classes, self._sample_max_nums
        ):
            present = sum(name == cls_name for name in gt_names)
            need = int(max_n - present)
            need = np.round(self._rate * need).astype(np.int64)
            sampled_num_dict[cls_name] = need
            sample_num_per_class.append(need)

        sampled_groups = list(self._sample_classes)
        if self._use_group_sampling:
            assert gt_group_ids is not None
            sampled_groups = []
            sample_num_per_class = []
            for gname, cls_names in self._group_name_to_names:
                need_g = max(sampled_num_dict[c] for c in cls_names)
                sample_num_per_class.append(need_g)
                sampled_groups.append(gname)
            total_group_ids = gt_group_ids

        sampled = []
        sampled_gt_boxes = []
        avoid_coll_boxes = gt_boxes

        for gname, need in zip(sampled_groups, sample_num_per_class):
            if need <= 0:
                continue
            if self._use_group_sampling:
                picked = self.sample_group(
                    gname, need, avoid_coll_boxes, total_group_ids
                )
            else:
                picked = self.sample_class_v2(gname, need, avoid_coll_boxes)

            sampled.extend(picked)
            if not picked:
                continue

            boxes = np.stack([s["box3d_lidar"] for s in picked], axis=0)
            if boxes.ndim == 1:
                boxes = boxes[np.newaxis, ...]
            sampled_gt_boxes.append(boxes)
            avoid_coll_boxes = np.concatenate(
                [avoid_coll_boxes, boxes], axis=0
            )
            if self._use_group_sampling:
                gids = np.asarray([s["group_id"] for s in picked])
                if gids.ndim == 0:
                    gids = gids[np.newaxis, ...]
                total_group_ids = np.concatenate(
                    [total_group_ids, gids], axis=0
                )

        if len(sampled) > 0:
            sampled_gt_boxes = np.concatenate(sampled_gt_boxes, axis=0)

            if road_planes is not None:
                # Only support KITTI
                # image plane
                # assert False, "Not correct yet!"
                a, b, c, d = road_planes

                center = sampled_gt_boxes[:, :3]
                center[:, 2] -= sampled_gt_boxes[:, 5] / 2
                center_cam = box_np_ops.lidar_to_camera(
                    center, calib["rect"], calib["Trv2c"]
                )

                cur_height_cam = (
                    -d - a * center_cam[:, 0] - c * center_cam[:, 2]
                ) / b
                center_cam[:, 1] = cur_height_cam
                lidar_tmp_point = box_np_ops.camera_to_lidar(
                    center_cam, calib["rect"], calib["Trv2c"]
                )
                cur_lidar_height = lidar_tmp_point[:, 2]

                # botom to middle center
                # kitti [0.5, 0.5, 0] center to [0.5, 0.5, 0.5]
                sampled_gt_boxes[:, 2] = (
                    cur_lidar_height + sampled_gt_boxes[:, 5] / 2
                )

                # mv_height = sampled_gt_boxes[:, 2] - cur_lidar_height
                # sampled_gt_boxes[:, 2] -= mv_height

            num_sampled = len(sampled)
            s_points_list = []
            if self.root_path and not root_path:
                root_path = self.root_path

            for info in sampled:
                try:
                    pt_path = pathlib.Path(root_path) / info["path"]
                    s_points = np.fromfile(
                        str(pt_path), dtype=np.float32
                    ).reshape(-1, num_point_features)
                    if "rot_transform" in info:
                        s_points[:, :3] = (
                            box_np_ops.rotation_points_single_angle(
                                s_points[:, :4], info["rot_transform"], axis=2
                            )
                        )
                    s_points[:, :3] = s_points[:, :3] + info["box3d_lidar"][:3]
                    s_points_list.append(s_points)
                except FileNotFoundError:
                    continue

            if random_crop:
                assert calib is not None
                rect = calib["rect"]
                Trv2c = calib["Trv2c"]
                P2 = calib["P2"]
                img_boxes = box_np_ops.box3d_to_bbox(
                    sampled_gt_boxes, rect, Trv2c, P2
                )
                frustums = random_crop_frustum(img_boxes, rect, Trv2c, P2)
                kept_clouds = []
                for fi in range(frustums.shape[0]):
                    cloud = s_points_list[fi]
                    drop = mask_points_in_corners(
                        cloud, frustums[fi : fi + 1]
                    ).reshape(-1)
                    n_in = int(drop.sum())
                    remain = cloud.shape[0] - n_in
                    if n_in > 0 and remain > 15:
                        cloud = cloud[np.logical_not(drop)]
                    kept_clouds.append(cloud)
                s_points_list = kept_clouds

            ret = {
                "gt_names": np.asarray([s["name"] for s in sampled]),
                "difficulty": np.asarray([s["difficulty"] for s in sampled]),
                "gt_boxes": sampled_gt_boxes,
                "points": np.concatenate(s_points_list, axis=0),
                "gt_masks": np.ones(num_sampled, dtype=np.bool_),
            }
            if self._use_group_sampling:
                ret["group_ids"] = np.asarray([s["group_id"] for s in sampled])
            else:
                base_id = gt_boxes.shape[0]
                ret["group_ids"] = np.arange(base_id, base_id + len(sampled))
        else:
            ret = None
        return ret

    def sample(self, name, num):
        sampler = self._sampler_dict[name]
        if self._use_group_sampling:
            groups = sampler.sample(num)
            sizes = [len(g) for g in groups]
            flat = []
            for g in groups:
                flat.extend(g)
            return flat, sizes
        items = sampler.sample(num)
        return items, np.ones(len(items), dtype=np.int64)

    def sample_v1(self, name, num):
        if isinstance(name, (list, tuple)):
            key = ", ".join(name)
            batches = self._sampler_dict[key].sample(num)
            sizes = [len(g) for g in batches]
            return reduce(lambda a, b: a + b, batches), sizes
        batches = self._sampler_dict[name].sample(num)
        return batches, np.ones((len(batches),), dtype=np.int64)

    def sample_class_v2(self, name, num, gt_boxes):
        sampled = copy.deepcopy(self._sampler_dict[name].sample(num))
        n_gt = gt_boxes.shape[0]
        n_sp = len(sampled)
        gt_bv = box_utils.center_to_corner_box2d(
            gt_boxes[:, 0:2], gt_boxes[:, 3:5], gt_boxes[:, -1]
        )
        sp_boxes = np.stack([item["box3d_lidar"] for item in sampled], axis=0)

        valid_mask = np.concatenate(
            [
                np.zeros(n_gt, dtype=np.bool_),
                np.ones(sp_boxes.shape[0], dtype=np.bool_),
            ],
            axis=0,
        )
        boxes = np.concatenate([gt_boxes, sp_boxes], axis=0).copy()
        if self._enable_global_rot:
            noise_per_object_v3_(
                boxes,
                None,
                valid_mask,
                0,
                0,
                self._global_rot_range,
                num_try=100,
            )

        sp_new = boxes[n_gt:]
        sp_bv = box_utils.center_to_corner_box2d(
            sp_new[:, 0:2], sp_new[:, 3:5], sp_new[:, -1]
        )
        all_bv = np.concatenate([gt_bv, sp_bv], axis=0)
        coll = box_collision_test(all_bv, all_bv)
        eye = np.arange(all_bv.shape[0])
        coll[eye, eye] = False

        kept = []
        for i in range(n_gt, n_gt + n_sp):
            if coll[i].any():
                coll[i] = False
                coll[:, i] = False
                continue
            local_i = i - n_gt
            if self._enable_global_rot:
                sampled[local_i]["box3d_lidar"][:2] = boxes[i, :2]
                sampled[local_i]["box3d_lidar"][-1] = boxes[i, -1]
                sampled[local_i]["rot_transform"] = (
                    boxes[i, -1] - sp_boxes[local_i, -1]
                )
            kept.append(sampled[local_i])
        return kept

    def sample_group(self, name, num, gt_boxes, gt_group_ids):
        sampled, group_num = self.sample(name, num)
        sampled = copy.deepcopy(sampled)

        remapped = {}
        next_gid = int(np.max(gt_group_ids)) + 1
        for s in sampled:
            old = s["group_id"]
            if old not in remapped:
                remapped[old] = next_gid
                next_gid += 1
            s["group_id"] = remapped[old]

        n_gt = gt_boxes.shape[0]
        gt_bv = box_utils.center_to_corner_box2d(
            gt_boxes[:, 0:2], gt_boxes[:, 3:5], gt_boxes[:, -1]
        )
        sp_boxes = np.stack([i["box3d_lidar"] for i in sampled], axis=0)
        sp_gids = np.stack([i["group_id"] for i in sampled], axis=0)
        valid_mask = np.concatenate(
            [
                np.zeros(n_gt, dtype=np.bool_),
                np.ones(sp_boxes.shape[0], dtype=np.bool_),
            ],
            axis=0,
        )
        boxes = np.concatenate([gt_boxes, sp_boxes], axis=0).copy()
        group_ids = np.concatenate([gt_group_ids, sp_gids], axis=0)
        if self._enable_global_rot:
            noise_per_object_v3_(
                boxes,
                None,
                valid_mask,
                0,
                0,
                self._global_rot_range,
                group_ids=group_ids,
                num_try=100,
            )
        sp_new = boxes[n_gt:]
        sp_bv = box_utils.center_to_corner_box2d(
            sp_new[:, 0:2], sp_new[:, 3:5], sp_new[:, -1]
        )
        all_bv = np.concatenate([gt_bv, sp_bv], axis=0)
        coll = box_collision_test(all_bv, all_bv)
        eye = np.arange(all_bv.shape[0])
        coll[eye, eye] = False

        kept = []
        idx = n_gt
        for gsize in group_num:
            if coll[idx : idx + gsize].any():
                coll[idx : idx + gsize] = False
                coll[:, idx : idx + gsize] = False
            else:
                for j in range(gsize):
                    local_i = idx - n_gt + j
                    if self._enable_global_rot:
                        sampled[local_i]["box3d_lidar"][:2] = boxes[
                            idx + j, :2
                        ]
                        sampled[local_i]["box3d_lidar"][-1] = boxes[
                            idx + j, -1
                        ]
                        sampled[local_i]["rot_transform"] = (
                            boxes[idx + j, -1] - sp_boxes[local_i, -1]
                        )
                    kept.append(sampled[local_i])
            idx += gsize
        return kept
