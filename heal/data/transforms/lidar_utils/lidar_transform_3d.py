# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from hat_engine.registry import OBJECT_REGISTRY, build_from_registry

from heal.core.box_np_ops import (
    center_to_corner_box3d,
    corner_to_surfaces_3d,
    limit_period,
    points_in_rbbox,
    remove_outside_points,
)
from heal.core.point_geometry import dropout_points_in_box
from heal.data.transforms.lidar_utils.augmentations import (
    global_rotation,
    global_scaling_v2,
    noise_per_object_v3_,
    random_flip,
    random_flip_both,
)
from heal.data.transforms.lidar_utils.filters import (
    filter_gt_box_outside_range,
)

__all__ = [
    "ObjectSample",
    "ObjectNoise",
    "PointRandomFlip",
    "PointGlobalRotation",
    "PointGlobalScaling",
    "ShufflePoints",
    "ObjectRangeFilter",
    "LidarReformat",
    "LidarMultiPreprocess",
]


def drop_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds


def _dict_select(dict_, inds):
    for k, v in dict_.items():
        if isinstance(v, dict):
            _dict_select(v, inds)
        else:
            if k not in ["gt_seg_labels"]:
                dict_[k] = v[inds]


@OBJECT_REGISTRY.register
class ObjectSample(object):
    """Sample GT objects to the data.

    Args:
        db_sampler: Database sampler.
        class_names: Class names.
        random_crop: Whether to random crop.
        remove_points_after_sample: Whether to remove points after
            sample.
        remove_outside_points: Whether to remove outsize points.
    """

    def __init__(
        self,
        db_sampler: Callable,
        class_names: List[str],
        random_crop: bool = False,
        remove_points_after_sample: bool = False,
        remove_outside_points: bool = False,
    ):
        self.db_sampler = db_sampler
        self.class_names = class_names
        self.random_crop = random_crop
        self.remove_points_after_sample = remove_points_after_sample
        self.remove_outside_points = remove_outside_points

    def __call__(self, input_dict):
        """Sample GT objects to the data.

        Args:
            input_dict: {
                "points": np.ndarray()  # pointcloud
                "lidar": {
                    "annotations": {
                        "boxes": np.ndarray()  # (x,y,z, w,l,h, rot)
                        "name": gt_names,
                    }
                }
            }

        """

        points = input_dict["lidar"]["points"]

        anno_dict = input_dict["lidar"]["annotations"]

        gt_dict = {
            "gt_boxes": anno_dict["boxes"],
            "gt_names": np.array(anno_dict["names"]).reshape(-1),
        }

        if "difficulty" not in anno_dict:
            difficulty = np.zeros(
                [anno_dict["boxes"].shape[0]], dtype=np.int32
            )  # noqa E501
            gt_dict["difficulty"] = difficulty
        else:
            gt_dict["difficulty"] = anno_dict["difficulty"]

        if "calib" in input_dict:
            calib = input_dict["calib"]
        else:
            calib = None

        if self.remove_outside_points:
            assert calib is not None
            image_shape = input_dict["image"]["image_shape"]
            points = remove_outside_points(
                points,
                calib["rect"],
                calib["Trv2c"],
                calib["P2"],
                image_shape,
            )

        selected = drop_arrays_by_name(
            gt_dict["gt_names"], ["DontCare", "ignore"]
        )

        _dict_select(gt_dict, selected)

        gt_dict.pop("difficulty")

        # Keep only categories we train on.
        keep_cls = np.fromiter(
            (name in self.class_names for name in gt_dict["gt_names"]),
            dtype=np.bool_,
            count=len(gt_dict["gt_names"]),
        )
        gt_boxes_mask = keep_cls

        if self.db_sampler:
            sampled_dict = self.db_sampler.sample_all(
                None,
                gt_dict["gt_boxes"],
                gt_dict["gt_names"],
                input_dict["metadata"]["num_point_features"],
                self.random_crop,
                gt_group_ids=None,
                calib=calib,
            )

            if sampled_dict is not None:
                add_names = sampled_dict["gt_names"]
                add_boxes = sampled_dict["gt_boxes"]
                sampled_points = sampled_dict["points"]
                sampled_gt_masks = sampled_dict["gt_masks"]
                gt_dict["gt_names"] = np.concatenate(
                    (gt_dict["gt_names"], add_names), axis=0
                )
                gt_dict["gt_boxes"] = np.concatenate(
                    (gt_dict["gt_boxes"], add_boxes), axis=0
                )
                gt_boxes_mask = np.concatenate(
                    [gt_boxes_mask, sampled_gt_masks], axis=0
                )

                if self.remove_points_after_sample:
                    masks = points_in_rbbox(points, add_boxes)
                    points = points[np.logical_not(masks.any(-1))]

                points = np.concatenate([sampled_points, points], axis=0)
        gt_classes = np.array(
            [
                self.class_names.index(n) + 1 if n in self.class_names else -1
                for n in gt_dict["gt_names"]
            ],
            dtype=np.int32,
        )
        gt_dict["gt_classes"] = gt_classes
        gt_dict["gt_boxes_mask"] = gt_boxes_mask
        input_dict["lidar"]["points"] = points
        input_dict["lidar"]["annotations"] = gt_dict
        return input_dict


@OBJECT_REGISTRY.register
class ObjectNoise(object):
    """Apply noise to each GT objects in the scene.

    Args:
        gt_rotation_noise: Object rotation range.
        gt_loc_noise_std: Object noise std.
        global_random_rot_range: Global rotation to the scene.
        num_try: Number of times to try if the noise applied is invalid.
    """

    def __init__(
        self,
        gt_rotation_noise: List[float],
        gt_loc_noise_std: List[float],
        global_random_rot_range: List[float],
        num_try: int = 100,
    ):

        self.gt_rotation_noise = gt_rotation_noise
        self.gt_loc_noise_std = gt_loc_noise_std
        self.global_random_rot_range = global_random_rot_range
        self.num_try = num_try

    def __call__(self, data):

        gt_dict = data["lidar"]["annotations"]
        points = data["lidar"]["points"]

        noise_per_object_v3_(
            gt_dict["gt_boxes"],
            points,
            gt_dict["gt_boxes_mask"],
            rotation_perturb=self.gt_rotation_noise,
            center_noise_std=self.gt_loc_noise_std,
            global_random_rot_range=self.global_random_rot_range,
            group_ids=None,
            num_try=self.num_try,
        )
        _dict_select(gt_dict, gt_dict["gt_boxes_mask"])

        data["lidar"]["annotations"] = gt_dict
        data["lidar"]["points"] = points

        return data


@OBJECT_REGISTRY.register
class PointRandomFlip(object):
    """Flip the points & bbox.

    Args:
        probability: The flipping probability.
    """

    def __init__(self, probability: float = 0.5) -> None:

        self.probability = probability

    def __call__(self, data: Dict):
        """Flip the points & bbox.

        Args:
            data: Input data, like {
                "lidar": {
                    "points": np.ndarray
                    "annotations": {
                        "gt_boxes": np.ndarray,
                        ...
                    },
                    ...
                }
            }

        """

        gt_boxes = data["lidar"]["annotations"]["gt_boxes"]
        points = data["lidar"]["points"]

        gt_boxes, points = random_flip(
            gt_boxes,
            points,
            probability=self.probability,
        )
        data["lidar"]["points"] = points
        data["lidar"]["annotations"]["gt_boxes"] = gt_boxes

        return data


@OBJECT_REGISTRY.register
class PointGlobalRotation(object):
    """Apply global rotation to a 3D scene.

    Args:
        rotation: Range of rotation angle.
    """

    def __init__(self, rotation: float = np.pi / 4):
        self.global_rotation_noise = rotation

    def __call__(self, data):

        gt_boxes, points = global_rotation(
            gt_boxes=data["lidar"]["annotations"]["gt_boxes"],
            points=data["lidar"]["points"],
            rotation=self.global_rotation_noise,
        )

        data["lidar"]["points"] = points
        data["lidar"]["annotations"]["gt_boxes"] = gt_boxes

        return data


@OBJECT_REGISTRY.register
class PointGlobalScaling(object):
    """Apply global scaling to a 3D scene.

    Args:
        min_scale: Min scale ratio.
        max_scale: Max scale ratio.
    """

    def __init__(self, min_scale: float = 0.95, max_scale: float = 1.05):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, data):

        gt_boxes, points = global_scaling_v2(
            gt_boxes=data["lidar"]["annotations"]["gt_boxes"],
            points=data["lidar"]["points"],
            min_scale=self.min_scale,
            max_scale=self.max_scale,
        )

        data["lidar"]["points"] = points
        data["lidar"]["annotations"]["gt_boxes"] = gt_boxes

        return data


@OBJECT_REGISTRY.register
class ShufflePoints(object):
    """Shuffle Points.

    Args:
        shuffle: Whether to shuffle
    """

    def __init__(self, shuffle: bool = True):
        self.shuffle = shuffle

    def __call__(self, data):
        # shuffle is a little slow.
        if self.shuffle:
            points = data["lidar"]["points"]
            np.random.shuffle(points)
            data["lidar"]["points"] = points

        return data


@OBJECT_REGISTRY.register
class ObjectRangeFilter(object):
    """Filter objects by point cloud range.

    Args:
        point_cloud_range: Point cloud range.
    """

    def __init__(self, point_cloud_range: List[float]):
        self.pc_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, data):
        is_tuple = False
        if isinstance(data, (list, tuple)):
            is_tuple = True
            data, info = data
        gt_dict = data["lidar"]["annotations"]
        bv_range = self.pc_range[[0, 1, 3, 4]]
        mask = filter_gt_box_outside_range(gt_dict["gt_boxes"], bv_range)
        _dict_select(gt_dict, mask)
        # limit rad to [-pi, pi]
        gt_dict["gt_boxes"][:, -1] = limit_period(
            gt_dict["gt_boxes"][:, -1], offset=0.5, period=2 * np.pi
        )

        data["lidar"]["annotations"] = gt_dict
        return data if (not is_tuple) else (data, info)


def dropout_points_in_gt(
    points: np.ndarray,
    rbbox: np.ndarray,
    prob: float = 0.2,
    z_axis: int = 2,
    origin=(0.5, 0.5, 0.5),
) -> np.ndarray:
    rbbox_corners = center_to_corner_box3d(
        rbbox[:, :3], rbbox[:, 3:6], rbbox[:, -1], origin=origin, axis=z_axis
    )
    surfaces = corner_to_surfaces_3d(rbbox_corners)
    return dropout_points_in_box(points, surfaces, prob)


@OBJECT_REGISTRY.register
class LidarMultiPreprocess(object):
    """Point cloud preprocessing for CenterPoint-style lidar detection."""

    def __init__(
        self,
        class_names: List[str],
        global_rot_noise: Tuple[float] = (0.0, 0.0),
        global_scale_noise: Tuple[float] = (0.0, 0.0),
        db_sampler: Optional[Dict] = None,
        shuffle_points: bool = False,
        flip_both: bool = False,
        flip_both_prob: float = 0.5,
        drop_points_in_gt: bool = False,
    ):
        self.global_rotation_noise = list(global_rot_noise)
        self.global_scaling_noise = list(global_scale_noise)
        self.db_sampler = (
            build_from_registry(db_sampler) if db_sampler is not None else None
        )
        self.class_names = class_names
        self.shuffle_points = shuffle_points
        self.flip_both = flip_both
        self.flip_both_prob = flip_both_prob
        self.drop_points_in_gt = drop_points_in_gt

    def __call__(self, res):
        points = res["lidar"]["points"]
        if res["mode"] == "train":
            anno_dict = res["lidar"]["annotations"]
            labels = {
                "gt_boxes": anno_dict["boxes"],
                "gt_names": np.asarray(anno_dict["names"]).reshape(-1),
            }
            gt_dict = labels
            valid = drop_arrays_by_name(
                gt_dict["gt_names"], ["DontCare", "ignore", "UNKNOWN"]
            )
            _dict_select(gt_dict, valid)

            gt_boxes_mask = np.fromiter(
                (name in self.class_names for name in gt_dict["gt_names"]),
                dtype=np.bool_,
                count=len(gt_dict["gt_names"]),
            )

            if self.db_sampler:
                sampled_dict = self.db_sampler.sample_all(
                    res["metadata"]["image_prefix"],
                    gt_dict["gt_boxes"],
                    gt_dict["gt_names"],
                    res["metadata"]["num_point_features"],
                    False,
                    gt_group_ids=None,
                    calib=None,
                    road_planes=None,
                )

                if sampled_dict is not None:
                    add_names = sampled_dict["gt_names"]
                    add_boxes = sampled_dict["gt_boxes"]
                    sampled_points = sampled_dict["points"]
                    sampled_gt_masks = sampled_dict["gt_masks"]
                    gt_dict["gt_names"] = np.concatenate(
                        (gt_dict["gt_names"], add_names), axis=0
                    )
                    gt_dict["gt_boxes"] = np.concatenate(
                        (gt_dict["gt_boxes"], add_boxes), axis=0
                    )
                    gt_boxes_mask = np.concatenate(
                        [gt_boxes_mask, sampled_gt_masks], axis=0
                    )

                    if self.drop_points_in_gt:
                        points = dropout_points_in_gt(
                            points, np.array(add_boxes), prob=1.0
                        )
                        points = np.concatenate(
                            [sampled_points, points], axis=0
                        )
                    else:
                        points = np.concatenate(
                            [points, sampled_points], axis=0
                        )

            _dict_select(gt_dict, gt_boxes_mask)

            gt_classes = np.array(
                [self.class_names.index(n) + 1 for n in gt_dict["gt_names"]],
                dtype=np.int32,
            )
            gt_dict["gt_classes"] = gt_classes

            if self.flip_both:
                gt_dict["gt_boxes"], points = random_flip_both(
                    gt_dict["gt_boxes"],
                    points,
                    probability=self.flip_both_prob,
                )

            gt_dict["gt_boxes"], points = global_rotation(
                gt_dict["gt_boxes"],
                points,
                rotation=self.global_rotation_noise,
            )
            gt_dict["gt_boxes"], points = global_scaling_v2(
                gt_dict["gt_boxes"],
                points,
                min_scale=self.global_scaling_noise[0],
                max_scale=self.global_scaling_noise[1],
            )

        if self.shuffle_points:
            np.random.shuffle(points)

        res["lidar"]["points"] = points
        if "gt_seg_labels" in res["lidar"]["annotations"]:
            gt_dict["gt_seg_labels"] = res["lidar"]["annotations"][
                "gt_seg_labels"
            ]

        res["lidar"]["annotations"] = gt_dict
        return res


@OBJECT_REGISTRY.register
class LidarReformat(object):
    """Reformat data.

    Args:
        with_gt: Whether to expand gt labels.
    """

    def __init__(self, with_gt: bool = False, **kwargs):
        self.with_gt = with_gt

    def __call__(self, res):
        is_tuple = False
        if isinstance(res, (list, tuple)):
            is_tuple = True
            res, info = res
        points = res["lidar"]["points"]
        voxels = res["lidar"].get("voxels", None)
        if voxels:
            data_bundle = dict(  # noqa C408
                metadata=res["metadata"],
                points=points,
                voxels=voxels["voxels"],
                shape=voxels["shape"],
                num_points=voxels["num_points"],
                num_voxels=voxels["num_voxels"],
                coordinates=voxels["coordinates"],
            )
        else:

            data_bundle = dict(  # noqa C408
                metadata=res["metadata"],
                points=points,
            )

        calib = res.get("calib", None)
        if calib:
            data_bundle["calib"] = calib

        if res["mode"] != "test":
            data_bundle["annotations"] = res["lidar"]["annotations"]

        if self.with_gt:
            if "gt_boxes" in res["lidar"]["annotations"]:
                data_bundle["gt_boxes"] = res["lidar"]["annotations"][
                    "gt_boxes"
                ]
            if "gt_classes" in res["lidar"]["annotations"]:
                data_bundle["gt_classess"] = res["lidar"]["annotations"][
                    "gt_classes"
                ]
            if "gt_seg_labels" in res["lidar"]["annotations"]:
                data_bundle["gt_seg_labels"] = res["lidar"]["annotations"][
                    "gt_seg_labels"
                ]
            if "gt_seg_mask" in res["lidar"]["annotations"]:
                data_bundle["gt_seg_mask"] = res["lidar"]["annotations"][
                    "gt_seg_mask"
                ]
        return data_bundle if (not is_tuple) else (data_bundle, info)
