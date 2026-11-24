# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

# filter helpers for lidar database and gt boxes.
import abc
from typing import List, Tuple

import numpy as np
from hat_engine.registry import OBJECT_REGISTRY

from heal.core import box_np_ops, box_utils, point_geometry

__all__ = ["DBFilterByDifficulty", "DBFilterByMinNumPoint"]


class DataBasePreprocessing:
    """
    The PreprocessingBase class.

    Args:
        db_infos (Dict): db values dict
    """

    def __call__(self, db_infos):
        return self._preprocess(db_infos)

    @abc.abstractclassmethod
    def _preprocess(self, db_infos):
        pass


@OBJECT_REGISTRY.register_module
class DBFilterByDifficulty(DataBasePreprocessing):
    """Filter sampled data by diffculties.

    Args:
        removed_difficulties (list): class diffculties
    """

    def __init__(self, filter_by_difficulty):
        self._removed_difficulties = filter_by_difficulty
        print(f"{filter_by_difficulty}")

    def _preprocess(self, db_infos):
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            new_db_infos[key] = [
                info
                for info in dinfos
                if info["difficulty"] not in self._removed_difficulties
            ]
        return new_db_infos


@OBJECT_REGISTRY.register_module
class DBFilterByMinNumPoint(DataBasePreprocessing):
    """Filter sampled data by NumPoint.

    Args:
        min_gt_point_dict (dict): class numpoint thershold
    """

    def __init__(self, filter_by_min_num_points):
        self._min_gt_point_dict = filter_by_min_num_points
        print(f"{filter_by_min_num_points}")

    def _preprocess(self, db_infos):
        for name, min_num in self._min_gt_point_dict.items():
            if min_num > 0:
                filtered_infos = []
                for info in db_infos[name]:
                    if info["num_points_in_gt"] >= min_num:
                        filtered_infos.append(info)
                db_infos[name] = filtered_infos
        return db_infos


def filter_gt_box_outside_range(
    gt_boxes: np.ndarray, limit_range: List[float]
) -> np.ndarray:
    """Remove gtbox outside training range.

    This function should be applied after other prep functions
    Args:
        gt_boxes ([type]): [description]
        limit_range ([type]): [description]
    """
    gt_boxes_bv = box_utils.center_to_corner_box2d(
        gt_boxes[:, [0, 1]], gt_boxes[:, [3, 3 + 1]], gt_boxes[:, -1]
    )
    bounding_box = box_utils.minmax_to_corner_2d(
        np.asarray(limit_range)[np.newaxis, ...]
    )
    # minmax_to_corner_2d produces a CCW polygon (y ascending), so tell the
    # point-in-polygon helper the polygon is not clockwise — otherwise
    # in-range corners test as outside (cross sign flipped) and valid boxes
    # get dropped.
    ret = point_geometry.points_in_convex_polygon_jit(
        gt_boxes_bv.reshape(-1, 2), bounding_box, clockwise=False
    )
    return np.any(ret.reshape(-1, 4), axis=1)


def filter_gt_box_outside_range_by_center(
    gt_boxes: np.ndarray, limit_range: List[float]
) -> np.ndarray:
    """Remove gtbox outside training range.

    This function should be applied after other prep functions
    Args:
        gt_boxes ([type]): [description]
        limit_range ([type]): [description]
    """
    gt_box_centers = gt_boxes[:, :2]
    bounding_box = box_utils.minmax_to_corner_2d(
        np.asarray(limit_range)[np.newaxis, ...]
    )
    # minmax_to_corner_2d produces a CCW polygon (y ascending) — see
    # filter_gt_box_outside_range for the clockwise=False rationale.
    ret = point_geometry.points_in_convex_polygon_jit(
        gt_box_centers, bounding_box, clockwise=False
    )
    return ret.reshape(-1)


def filter_gt_low_points(
    gt_boxes: np.ndarray,
    points: np.ndarray,
    num_gt_points: List[int],
    point_num_threshold: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove low points gtbox.

    Args:
        gt_boxes :(shape=[N, 7]).
        points :(shape=[N, 3] or shape=[N, 4]).
        num_gt_points:(every gt has points num.
        point_num_threshold:threshold of num points in gt.
    Returns:
        gt_boxes:(shape=[N, 7]).
        points:(shape=[N, 3]).
    """
    points_mask = np.ones([points.shape[0]], np.bool_)
    gt_boxes_mask = np.ones([gt_boxes.shape[0]], np.bool_)
    for i, num in enumerate(num_gt_points):
        if num <= point_num_threshold:
            masks = box_np_ops.points_in_rbbox(points, gt_boxes[i : i + 1])
            masks = masks.reshape([-1])
            points_mask &= np.logical_not(masks)
            gt_boxes_mask[i] = False
    return gt_boxes[gt_boxes_mask], points[points_mask]


def mask_points_in_corners(
    points: np.ndarray, box_corners: np.ndarray
) -> np.ndarray:
    """Convert camera world points to lidar world points.

    Check points is in 3d convex polygons.

    Args:
        box_corners :(shape=[N, 7]).
        points :(shape=[N, 3] or shape=[N, 4]).
    Returns:
        gt_boxes:(shape=[N, 7]).
        points:[num_points, num_polygon] bool array.
    """
    surfaces = box_np_ops.corner_to_surfaces_3d(box_corners)
    mask = point_geometry.points_in_convex_polygon_3d_jit(
        points[:, :3], surfaces
    )
    return mask
