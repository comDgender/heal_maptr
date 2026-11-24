# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
#
# Lidar point-cloud augmentation helpers: noise / flip / rotation / scaling /
# collision transforms, plus HEAL-specific utilities (_sample_flip_gate,
# _mirror_bev_lateral/longitudinal, random_flip_both, rotate_image).

from collections import OrderedDict
from typing import Any, Tuple

import cv2
import numpy as np
from scipy import ndimage

from heal.core import box_np_ops, point_geometry
from heal.core.numba_lazy import lazy_jit, lazy_njit


def random_crop_frustum(
    bboxes: np.ndarray,
    rect: np.ndarray,
    Trv2c: np.ndarray,
    P2: np.ndarray,
    max_crop_height: float = 1.0,
    max_crop_width: float = 0.9,
) -> np.ndarray:
    """Randomly crop 2D boxes and lift them to lidar frustums.

    Args:
        bboxes:(float array, shape=[N, 4]).
        rect:(float array, shape=[4, 4]) matrix rect.
        Trv2c:(float array, shape=[4, 4]) translate matrix vel2cam.
        P2:(float array, shape=[3, 4]). matrix projection.
        max_crop_height:(float).
        max_crop_width:(float).
    Returns:
        frustums :(float array, shape=[N, 8, 3]).
    """
    num_gt = bboxes.shape[0]
    crop_minxy = np.random.uniform(
        [1 - max_crop_width, 1 - max_crop_height], [0.3, 0.3], size=[num_gt, 2]
    )
    crop_maxxy = np.ones([num_gt, 2], dtype=bboxes.dtype)
    crop_bboxes = np.concatenate([crop_minxy, crop_maxxy], axis=1)
    align_left = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if align_left:
        crop_bboxes[:, [0, 2]] -= crop_bboxes[:, 0:1]
    box_wh = bboxes[:, 2:] - bboxes[:, :2]
    crop_bboxes = crop_bboxes * np.concatenate([box_wh, box_wh], axis=1)
    crop_bboxes = crop_bboxes + np.concatenate(
        [bboxes[:, :2], bboxes[:, :2]], axis=1
    )
    C, R, T = box_np_ops.projection_matrix_to_CRT_kitti(P2)
    frustums = box_np_ops.get_frustum(crop_bboxes, C)
    frustums = frustums - T
    frustums = np.einsum("ij, akj->aki", np.linalg.inv(R), frustums)
    frustums = box_np_ops.camera_to_lidar(frustums, rect, Trv2c)
    return frustums


@lazy_njit
def _bev_unit_corners(dtype):
    """Centered unit-square BEV corner offsets for noise_v2 helpers."""
    corners_norm = np.zeros((4, 2), dtype=dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=dtype)
    return corners_norm.reshape(4, 2)


@lazy_njit
def _rotation_matrix_3d_(rot_mat_T: np.ndarray, angle: np.ndarray, axis: int):
    """Ratate rotation_matrix.

    Args:
        rot_mat_T:(shape=[3, 3]).
        angle:rotate angele.
        axis:rotate axis
    """
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_T[:] = np.eye(3)
    if axis == 1:
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 2] = -rot_sin
        rot_mat_T[2, 0] = rot_sin
        rot_mat_T[2, 2] = rot_cos
    elif axis == 2 or axis == -1:
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 1] = -rot_sin
        rot_mat_T[1, 0] = rot_sin
        rot_mat_T[1, 1] = rot_cos
    elif axis == 0:
        rot_mat_T[1, 1] = rot_cos
        rot_mat_T[1, 2] = -rot_sin
        rot_mat_T[2, 1] = rot_sin
        rot_mat_T[2, 2] = rot_cos


@lazy_njit
def _rotation_box2d_jit_(
    corners: np.ndarray, angle: float, rot_mat_T: np.ndarray
):
    """Ratate 2d box.

    Args:
        rot_mat_T:(shape=[3, 3]).
        angle: rotate angele.
        corners:(shape=[N, 3]).
    """
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_T[0, 0] = rot_cos
    rot_mat_T[0, 1] = -rot_sin
    rot_mat_T[1, 0] = rot_sin
    rot_mat_T[1, 1] = rot_cos
    corners[:] = corners @ rot_mat_T


@lazy_njit
def noise_per_box(
    boxes: np.ndarray,
    valid_mask: np.ndarray,
    loc_noises: np.ndarray,
    rot_noises: np.ndarray,
) -> np.ndarray:
    """Add noise in per box.

    Args:
        boxes:(shape=[N, 5]).
        valid_mask: valid bool list.
        loc_noises:(float array, shape=[N,M, 3]).
        rot_noises:(float array, shape=[N,M, ]).

    Returns:
        success_mask:(float array, shape=[N,]).
    """
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    trial_corners = np.zeros((4, 2), dtype=boxes.dtype)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes,), dtype=np.int64)
    for box_i in range(num_boxes):
        if not valid_mask[box_i]:
            continue
        center_xy = boxes[box_i, :2]
        for try_j in range(num_tests):
            trial_corners[:] = box_corners[box_i]
            trial_corners -= center_xy
            _rotation_box2d_jit_(
                trial_corners, rot_noises[box_i, try_j], rot_mat_T
            )
            trial_corners += center_xy + loc_noises[box_i, try_j, :2]
            coll_mat = box_collision_test(
                trial_corners.reshape(1, 4, 2), box_corners
            )
            coll_mat[0, box_i] = False
            if not coll_mat.any():
                success_mask[box_i] = try_j
                box_corners[box_i] = trial_corners
                break
    return success_mask


@lazy_njit
def noise_per_box_group(
    boxes: np.ndarray,
    valid_mask: np.ndarray,
    loc_noises: np.ndarray,
    rot_noises: np.ndarray,
    group_nums: np.ndarray,
) -> np.ndarray:
    """Add noise in per box by group.

    Args:
        boxes:(shape=[N, 5]).
        valid_mask:(shape=[N]).
        loc_noises:(shape=[N,M, 3]).
        rot_noises:(shape=[N,M, ]).
        group_nums:(shape=[N,]).

    Returns:
        success_mask:(shape=[N,]).
    """
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    max_group_num = group_nums.max()
    trial_corners = np.zeros((max_group_num, 4, 2), dtype=boxes.dtype)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes,), dtype=np.int64)
    group_start = 0
    for group_size in group_nums:
        if not valid_mask[group_start]:
            group_start += group_size
            continue
        for try_j in range(num_tests):
            for member_i in range(group_size):
                abs_i = group_start + member_i
                trial_corners[member_i] = box_corners[abs_i]
                trial_corners[member_i] -= boxes[abs_i, :2]
                _rotation_box2d_jit_(
                    trial_corners[member_i],
                    rot_noises[abs_i, try_j],
                    rot_mat_T,
                )
                trial_corners[member_i] += (
                    boxes[abs_i, :2] + loc_noises[abs_i, try_j, :2]
                )
            coll_mat = box_collision_test(
                trial_corners[:group_size].reshape(group_size, 4, 2),
                box_corners,
            )
            group_end = group_start + group_size
            for member_i in range(group_size):
                coll_mat[member_i, group_start:group_end] = False
            if not coll_mat.any():
                for member_i in range(group_size):
                    abs_i = group_start + member_i
                    success_mask[abs_i] = try_j
                    box_corners[abs_i] = trial_corners[member_i]
                break
        group_start += group_size
    return success_mask


@lazy_njit
def noise_per_box_group_v2_(
    boxes: np.ndarray,
    valid_mask: np.ndarray,
    loc_noises: np.ndarray,
    rot_noises: np.ndarray,
    group_nums: np.ndarray,
    global_rot_noises: np.ndarray,
) -> np.ndarray:
    """Add noise in per box by group.

    V2 function need boxes to be sorted by group id.

    Args:
        boxes:(shape=[N, 5]).
        valid_mask:(shape=[N]).
        loc_noises:(shape=[N,M, 3]).
        rot_noises:(shape=[N,M, ]).
        group_nums: (shape=[N,]).
        global_rot_noises:(shape=[N,M, ]).

    Returns:
        success_mask:(shape=[N,]).
    """
    n_box = boxes.shape[0]
    n_try = loc_noises.shape[1]
    bev_corners = box_np_ops.box2d_to_corner_jit(boxes)
    g_max = int(group_nums.max())
    scratch_box = np.zeros((1, 5), dtype=boxes.dtype)
    trial = np.zeros((g_max, 4, 2), dtype=boxes.dtype)
    dst_xy = np.zeros((g_max, 2), dtype=boxes.dtype)
    src_yaw = np.zeros((g_max,), dtype=boxes.dtype)
    dst_yaw = np.zeros((g_max,), dtype=boxes.dtype)
    rot_T = np.zeros((2, 2), dtype=boxes.dtype)
    success_mask = -np.ones((n_box,), dtype=np.int64)
    unit_corners = _bev_unit_corners(boxes.dtype)

    # Alias names used by the trial loop below.
    box_corners = bev_corners
    current_box = scratch_box
    trial_corners = trial
    dst_pos = dst_xy
    src_grot = src_yaw
    dst_grot = dst_yaw
    rot_mat_T = rot_T
    corners_norm = unit_corners
    num_tests = n_try

    cursor = 0
    for g_size in group_nums:
        if not valid_mask[cursor]:
            cursor += g_size
            continue
        group_start = cursor
        group_size = g_size
        for try_j in range(num_tests):
            for member_i in range(group_size):
                abs_i = group_start + member_i
                current_box[0, :] = boxes[abs_i]
                radius = np.sqrt(
                    current_box[0, 0] ** 2 + current_box[0, 1] ** 2
                )
                src_grot[member_i] = np.arctan2(
                    current_box[0, 0], current_box[0, 1]
                )
                dst_grot[member_i] = (
                    src_grot[member_i] + global_rot_noises[abs_i, try_j]
                )
                dst_pos[member_i, 0] = radius * np.sin(dst_grot[member_i])
                dst_pos[member_i, 1] = radius * np.cos(dst_grot[member_i])
                current_box[0, :2] = dst_pos[member_i]
                current_box[0, -1] += dst_grot[member_i] - src_grot[member_i]

                yaw = current_box[0, -1]
                rot_sin = np.sin(yaw)
                rot_cos = np.cos(yaw)
                rot_mat_T[0, 0] = rot_cos
                rot_mat_T[0, 1] = -rot_sin
                rot_mat_T[1, 0] = rot_sin
                rot_mat_T[1, 1] = rot_cos
                trial_corners[member_i] = (
                    current_box[0, 2:4] * corners_norm @ rot_mat_T
                    + current_box[0, :2]
                )
                trial_corners[member_i] -= current_box[0, :2]
                _rotation_box2d_jit_(
                    trial_corners[member_i],
                    rot_noises[abs_i, try_j],
                    rot_mat_T,
                )
                trial_corners[member_i] += (
                    current_box[0, :2] + loc_noises[abs_i, try_j, :2]
                )
            coll_mat = box_collision_test(
                trial_corners[:group_size].reshape(group_size, 4, 2),
                box_corners,
            )
            group_end = group_start + group_size
            for member_i in range(group_size):
                coll_mat[member_i, group_start:group_end] = False
            if not coll_mat.any():
                for member_i in range(group_size):
                    abs_i = group_start + member_i
                    success_mask[abs_i] = try_j
                    box_corners[abs_i] = trial_corners[member_i]
                    loc_noises[abs_i, try_j, :2] += (
                        dst_pos[member_i] - boxes[abs_i, :2]
                    )
                    rot_noises[abs_i, try_j] += (
                        dst_grot[member_i] - src_grot[member_i]
                    )
                break
        cursor += g_size
    return success_mask


@lazy_njit
def noise_per_box_v2_(
    boxes: np.ndarray,
    valid_mask: np.ndarray,
    loc_noises: np.ndarray,
    rot_noises: np.ndarray,
    global_rot_noises: np.ndarray,
) -> np.ndarray:
    """Add noise in per box.

    V2 function need boxes to be sorted by group id.

    Args:
        boxes:(shape=[N, 5]).
        valid_mask: valid bool list.
        loc_noises:(shape=[N,M, 3]).
        rot_noises:(shape=[N,M, ]).
        global_rot_noises:(shape=[N,M ])
    Returns:
        success_mask:(shape=[N,]).
    """
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box_np_ops.box2d_to_corner_jit(boxes)
    trial_corners = np.zeros((4, 2), dtype=boxes.dtype)
    current_box = np.zeros((1, 5), dtype=boxes.dtype)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    dst_pos = np.zeros((2,), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes,), dtype=np.int64)
    corners_norm = _bev_unit_corners(boxes.dtype)
    for box_i in range(num_boxes):
        if not valid_mask[box_i]:
            continue
        for try_j in range(num_tests):
            current_box[0, :] = boxes[box_i]
            radius = np.sqrt(boxes[box_i, 0] ** 2 + boxes[box_i, 1] ** 2)
            src_grot = np.arctan2(boxes[box_i, 0], boxes[box_i, 1])
            dst_grot = src_grot + global_rot_noises[box_i, try_j]
            dst_pos[0] = radius * np.sin(dst_grot)
            dst_pos[1] = radius * np.cos(dst_grot)
            current_box[0, :2] = dst_pos
            current_box[0, -1] += dst_grot - src_grot

            yaw = current_box[0, -1]
            rot_sin = np.sin(yaw)
            rot_cos = np.cos(yaw)
            rot_mat_T[0, 0] = rot_cos
            rot_mat_T[0, 1] = -rot_sin
            rot_mat_T[1, 0] = rot_sin
            rot_mat_T[1, 1] = rot_cos
            trial_corners[:] = (
                current_box[0, 2:4] * corners_norm @ rot_mat_T
                + current_box[0, :2]
            )
            trial_corners -= current_box[0, :2]
            _rotation_box2d_jit_(
                trial_corners, rot_noises[box_i, try_j], rot_mat_T
            )
            trial_corners += current_box[0, :2] + loc_noises[box_i, try_j, :2]
            coll_mat = box_collision_test(
                trial_corners.reshape(1, 4, 2), box_corners
            )
            coll_mat[0, box_i] = False
            if not coll_mat.any():
                success_mask[box_i] = try_j
                box_corners[box_i] = trial_corners
                loc_noises[box_i, try_j, :2] += dst_pos - boxes[box_i, :2]
                rot_noises[box_i, try_j] += dst_grot - src_grot
                break
    return success_mask


@lazy_njit
def points_transform_(
    points: np.ndarray,
    centers: np.ndarray,
    point_masks: np.ndarray,
    loc_transform: np.ndarray,
    rot_transform: np.ndarray,
    valid_mask: np.ndarray,
):
    """Transfrom of points.

    Args:
        points:(shape=[N, 4]).
        centers:(shape=[N, 4]) center position of points.
        point_masks:(shape=[N,M]).
        loc_transform:(shape=[N]) location transfrom every points.
        rot_transform:(shape=[N]) rotate transform every points.
        valid_mask:(shape=[N]) valid points.
    """
    num_box = centers.shape[0]
    num_points = points.shape[0]
    rot_mat_T = np.zeros((num_box, 3, 3), dtype=points.dtype)
    for i in range(num_box):
        _rotation_matrix_3d_(rot_mat_T[i], rot_transform[i], 2)
    for i in range(num_points):
        for j in range(num_box):
            if valid_mask[j]:
                if point_masks[i, j] == 1:
                    points[i, :3] -= centers[j, :3]
                    points[i : i + 1, :3] = (
                        points[i : i + 1, :3] @ rot_mat_T[j]
                    )
                    points[i, :3] += centers[j, :3]
                    points[i, :3] += loc_transform[j]
                    break  # only apply first box's transform


@lazy_njit
def box3d_transform_(
    boxes: np.ndarray,
    loc_transform: np.ndarray,
    rot_transform: np.ndarray,
    valid_mask: np.ndarray,
):
    """Transfrom of box3d.

    Args:
        boxes:(shape=[N, 7]).
        loc_transform:(shape=[N,3 ]) location transfrom every box.
        rot_transform:(shape=[N,1 ]) rotate transform every box.
        valid_mask:(shape=[N ]) valid box.
    """
    num_box = boxes.shape[0]
    for i in range(num_box):
        if valid_mask[i]:
            boxes[i, :3] += loc_transform[i]
            boxes[i, 6] += rot_transform[i]


def _select_transform(
    transform: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    """Select transfrom from list.

    Args:
        boxes:(shape=[N, 7]).
        loc_transform:(shape=[N,3 ]) location transfrom every box.
        rot_transform:(shape=[N,1 ]) rotate transform every box.
        valid_mask:(shape=[N ]) valid box.

    Returns:
        result: (shape-[N,M]) select transform mask.
    """
    result = np.zeros(
        (transform.shape[0], *transform.shape[2:]), dtype=transform.dtype
    )
    for i in range(transform.shape[0]):
        if indices[i] != -1:
            result[i] = transform[i, indices[i]]
    return result


@lazy_njit
def group_transform_(
    loc_noise: np.ndarray,
    rot_noise: np.ndarray,
    locs: np.ndarray,
    rots: np.ndarray,
    group_center: np.ndarray,
    valid_mask: np.ndarray,
):
    """Group transfrom for loc_noise parameter.

    Args:
        loc_noise:(shape=[N, M, 3]).
        rot_noise:(shape=[N, M]) location transfrom by group.
        locs:(shape=[N, 3]).location transfrom by group.
        rots:(shape=[N, 1]).rotate transform by group.
        group_center:(shape=[N, 3]) rotate transform by group.
        valid_mask:(shape=[N ]).
    """
    num_try = loc_noise.shape[1]
    r = 0.0
    x = 0.0
    y = 0.0
    rot_center = 0.0
    for i in range(loc_noise.shape[0]):
        if valid_mask[i]:
            x = locs[i, 0] - group_center[i, 0]
            y = locs[i, 1] - group_center[i, 1]
            r = np.sqrt(x**2 + y**2)
            # calculate rots related to group center
            rot_center = np.arctan2(x, y)
            for j in range(num_try):
                loc_noise[i, j, 0] += r * (
                    np.sin(rot_center + rot_noise[i, j]) - np.sin(rot_center)
                )
                loc_noise[i, j, 1] += r * (
                    np.cos(rot_center + rot_noise[i, j]) - np.cos(rot_center)
                )


@lazy_njit
def group_transform_v2_(
    loc_noise: np.ndarray,
    rot_noise: np.ndarray,
    locs: np.ndarray,
    rots: np.ndarray,
    group_center: np.ndarray,
    grot_noise: np.ndarray,
    valid_mask: np.ndarray,
):
    """Group transfrom for loc_noise parameter.

    V2 make global noise with global rot_noise.

    Args:
        loc_noise:(shape=[N, M, 3]).
        rot_noise:(shape=[N, M]) location transfrom by group.
        locs:(shape=[N, 3]).location transfrom by group.
        rots:(shape=[N, 1]).rotate transform by group.
        group_center:(shape=[N, 3]) rotate transform by group.
        grot_noise: (shape=[N, 1]).globle rotate transform.
        valid_mask:(shape=[N ]).
    """
    num_try = loc_noise.shape[1]
    for i in range(loc_noise.shape[0]):
        if not valid_mask[i]:
            continue
        dx = locs[i, 0] - group_center[i, 0]
        dy = locs[i, 1] - group_center[i, 1]
        radius = np.sqrt(dx**2 + dy**2)
        rot_center = np.arctan2(dx, dy)
        for j in range(num_try):
            angle_with_rot = rot_center + rot_noise[i, j] + grot_noise[i, j]
            angle_base = rot_center + grot_noise[i, j]
            loc_noise[i, j, 0] += radius * (
                np.sin(angle_with_rot) - np.sin(angle_base)
            )
            loc_noise[i, j, 1] += radius * (
                np.cos(angle_with_rot) - np.cos(angle_base)
            )


def set_group_noise_same_(
    loc_noise: np.ndarray, rot_noise: np.ndarray, group_ids: np.ndarray
):
    """Broadcast each group's first noise sample to all members.

    Args:
        loc_noise:(shape=[N, M, 3]).
        rot_noise:(shape=[N, M]) location transform by group.
        group_ids:(shape=[N]) group id per box.
    """
    first_idx_by_gid = {}
    for idx, gid in enumerate(group_ids):
        if gid not in first_idx_by_gid:
            first_idx_by_gid[gid] = idx
    for idx in range(loc_noise.shape[0]):
        src = first_idx_by_gid[group_ids[idx]]
        loc_noise[idx] = loc_noise[src]
        rot_noise[idx] = rot_noise[src]


def set_group_noise_same_v2_(
    loc_noise: np.ndarray,
    rot_noise: np.ndarray,
    grot_noise: np.ndarray,
    group_ids: np.ndarray,
):
    """Broadcast each group's first loc/rot/global-rot noise to all members.

    V2 also unifies global rotation noise within a group.

    Args:
        loc_noise:(shape=[N, M, 3]).
        rot_noise:(shape=[N, M]) location transform by group.
        grot_noise: (shape=[N, M]). global rotate transform.
        group_ids:(shape=[N]) group id per box.
    """
    first_idx_by_gid = {}
    for idx, gid in enumerate(group_ids):
        if gid not in first_idx_by_gid:
            first_idx_by_gid[gid] = idx
    for idx in range(loc_noise.shape[0]):
        src = first_idx_by_gid[group_ids[idx]]
        loc_noise[idx] = loc_noise[src]
        rot_noise[idx] = rot_noise[src]
        grot_noise[idx] = grot_noise[src]


def get_group_center(
    locs: np.ndarray, group_ids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean center of each group and map it back to every member.

    Args:
        locs:(shape=[N, 3]).
        group_ids:(shape=[N]) group id per box.

    Returns:
        group_centers_ret:(shape=[N, 3]).
        group_id_num_dict:(OrderedDict {id: count}).
    """
    group_centers = np.zeros_like(locs)
    group_centers_ret = np.zeros_like(locs)
    gid_to_slot = {}
    group_id_num_dict = OrderedDict()
    next_slot = 0
    for i, gid in enumerate(group_ids):
        if gid < 0:
            continue
        if gid not in gid_to_slot:
            gid_to_slot[gid] = next_slot
            next_slot += 1
            group_id_num_dict[gid] = 1
            group_centers[gid_to_slot[gid]] = locs[i]
        else:
            group_centers[gid_to_slot[gid]] += locs[i]
            group_id_num_dict[gid] += 1
    for i, gid in enumerate(group_ids):
        slot = gid_to_slot[gid]
        group_centers_ret[i] = group_centers[slot] / group_id_num_dict[gid]
    return group_centers_ret, group_id_num_dict


def noise_per_object_v3_(
    gt_boxes: np.ndarray,
    points: np.ndarray = None,
    valid_mask: np.ndarray = None,
    rotation_perturb: float = np.pi / 4,
    center_noise_std: float = 1.0,
    global_random_rot_range: float = np.pi / 4,
    num_try: int = 5,
    group_ids: np.ndarray = None,
):
    """Randomly rotate/translate each ground-truth box independently.

    Use kitti viewer to test this function points_transform_.

    Args:
        gt_boxes: [N, 7], gt box in lidar.points_transform_.
        points: [M, 4], point cloud in lidar.
        valid_mask:(shape=[N ]).
        group_ids: (shape=[N]) rotate transform by group.
    """
    n_box = gt_boxes.shape[0]
    rot_range = rotation_perturb
    if not isinstance(rot_range, (list, tuple, np.ndarray)):
        rot_range = [-rot_range, rot_range]
    grot_range = global_random_rot_range
    if not isinstance(grot_range, (list, tuple, np.ndarray)):
        grot_range = [-grot_range, grot_range]
    use_global_rot = abs(grot_range[0] - grot_range[1]) >= 1e-3
    noise_std = center_noise_std
    if not isinstance(noise_std, (list, tuple, np.ndarray)):
        noise_std = [noise_std, noise_std, noise_std]
    if valid_mask is None:
        valid_mask = np.ones((n_box,), dtype=np.bool_)
    noise_std = np.asarray(noise_std, dtype=gt_boxes.dtype)
    # RNG call order must stay fixed for reproducibility.
    loc_noises = np.random.normal(scale=noise_std, size=[n_box, num_try, 3])
    rot_noises = np.random.uniform(
        rot_range[0], rot_range[1], size=[n_box, num_try]
    )
    box_azim = np.arctan2(gt_boxes[:, 0], gt_boxes[:, 1])
    grot_lo = grot_range[0] - box_azim
    grot_hi = grot_range[1] - box_azim
    global_rot_noises = np.random.uniform(
        grot_lo[..., np.newaxis],
        grot_hi[..., np.newaxis],
        size=[n_box, num_try],
    )

    group_nums = None
    if group_ids is not None:
        centers, id_counts = get_group_center(gt_boxes[:, :3], group_ids)
        if use_global_rot:
            set_group_noise_same_v2_(
                loc_noises, rot_noises, global_rot_noises, group_ids
            )
            group_transform_v2_(
                loc_noises,
                rot_noises,
                gt_boxes[:, :3],
                gt_boxes[:, 6],
                centers,
                global_rot_noises,
                valid_mask,
            )
        else:
            set_group_noise_same_(loc_noises, rot_noises, group_ids)
            group_transform_(
                loc_noises,
                rot_noises,
                gt_boxes[:, :3],
                gt_boxes[:, 6],
                centers,
                valid_mask,
            )
        group_nums = np.asarray(list(id_counts.values()), dtype=np.int64)

    origin = [0.5, 0.5, 0.5]
    corners3d = box_np_ops.center_to_corner_box3d(
        gt_boxes[:, :3],
        gt_boxes[:, 3:6],
        gt_boxes[:, 6],
        origin=origin,
        axis=2,
    )
    bev = gt_boxes[:, [0, 1, 3, 4, 6]]
    if group_ids is not None:
        if use_global_rot:
            selected = noise_per_box_group_v2_(
                bev,
                valid_mask,
                loc_noises,
                rot_noises,
                group_nums,
                global_rot_noises,
            )
        else:
            selected = noise_per_box_group(
                bev, valid_mask, loc_noises, rot_noises, group_nums
            )
    elif use_global_rot:
        selected = noise_per_box_v2_(
            bev, valid_mask, loc_noises, rot_noises, global_rot_noises
        )
    else:
        selected = noise_per_box(bev, valid_mask, loc_noises, rot_noises)
    loc_transforms = _select_transform(loc_noises, selected)
    rot_transforms = _select_transform(rot_noises, selected)
    surfaces = box_np_ops.corner_to_surfaces_3d_jit(corners3d)
    if points is not None:
        point_masks = point_geometry.points_in_convex_polygon_3d_jit(
            points[:, :3], surfaces
        )
        points_transform_(
            points,
            gt_boxes[:, :3],
            point_masks,
            loc_transforms,
            rot_transforms,
            valid_mask,
        )

    box3d_transform_(gt_boxes, loc_transforms, rot_transforms, valid_mask)


def global_scaling(
    gt_boxes: np.ndarray, points: np.ndarray, scale: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """Scaling for points and boxes.

    Args:
        gt_boxes:(shape=[N, 7]).
        points:(shape=[N, 4]).
        scale: scale value.

    Returns:
        gt_boxes:(shape=[N, 4]).
        points:(shape=[N, 4]).
    """

    if not isinstance(scale, list):
        scale = [-scale, scale]
    noise_scale = np.random.uniform(scale[0] + 1, scale[1] + 1)
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale
    return gt_boxes, points


def global_rotation(
    gt_boxes: np.ndarray,
    points: np.ndarray,
    rotation: float = np.pi / 4,
    mapGT: np.ndarray = None,
    pivot: np.ndarray = None,
    return_trans=False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Ratation for points and boxes.

    Args:
        gt_boxes:(shape=[N, 7]).
        points:(shape=[N, 4]).
        rotation:rotate angle.

    Returns:
        gt_boxes:(shape=[N, 4]).
        points:(shape=[N, 4]).
    """
    if not isinstance(rotation, list):
        rotation = [-rotation, rotation]
    noise_rotation = np.random.uniform(rotation[0], rotation[1])
    points[:, :3] = box_np_ops.rotation_points_single_angle(
        points[:, :3], noise_rotation, axis=2
    )
    gt_boxes[:, :3] = box_np_ops.rotation_points_single_angle(
        gt_boxes[:, :3], noise_rotation, axis=2
    )
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 6:8] = box_np_ops.rotation_points_single_angle(
            np.hstack([gt_boxes[:, 6:8], np.zeros((gt_boxes.shape[0], 1))]),
            noise_rotation,
            axis=2,
        )[:, :2]
    gt_boxes[:, -1] += noise_rotation
    if mapGT is not None:
        mapGT = rotate_image(mapGT, noise_rotation, pivot)
        return gt_boxes, points, mapGT
    if return_trans:
        return gt_boxes, points, noise_rotation
    else:
        return gt_boxes, points


def rotate_image(
    img: np.ndarray, angle: float, pivot: np.ndarray
) -> np.ndarray:
    """Ratation for image.

    Args:
        img:(shape=[H, W]).
        angle:ratate angle.
        pivot:(shape=[2]).

    Returns:
        img:(shape=[H, W]).
    """
    padX = [img.shape[1] - pivot[1], pivot[1]]
    padY = [img.shape[0] - pivot[0], pivot[0]]
    imgP = np.pad(
        img, [padY, padX], "constant", constant_values=-1
    )  # -1 ignore
    imgR = ndimage.rotate(imgP, angle / np.pi * 180, reshape=False, order=0)
    return imgR[padY[0] : -padY[1], padX[0] : -padX[1]]


def _sample_flip_gate(probability: float) -> bool:
    return bool(np.random.rand() < probability)


def _mirror_bev_lateral(gt_boxes: np.ndarray, points: np.ndarray) -> None:
    gt_boxes[:, 1] *= -1.0
    points[:, 1] *= -1.0
    gt_boxes[:, -1] = np.pi - gt_boxes[:, -1]
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7] *= -1.0


def _mirror_bev_longitudinal(
    gt_boxes: np.ndarray, points: np.ndarray, pivot_x=None
) -> None:
    if pivot_x is None:
        gt_boxes[:, 0] *= -1.0
        points[:, 0] *= -1.0
    else:
        mirrored_x = pivot_x * 2.0
        gt_boxes[:, 0] = mirrored_x - gt_boxes[:, 0]
        points[:, 0] = mirrored_x - points[:, 0]
    gt_boxes[:, -1] = 2.0 * np.pi - gt_boxes[:, -1]
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 6] *= -1.0


def random_flip(
    gt_boxes: np.ndarray,
    points: np.ndarray,
    probability: float = 0.5,
    return_trans: bool = False,
) -> np.ndarray:
    """Random flip for points and boxes.

    Args:
        gt_boxes:(shape=[N, 7]).
        points:(shape=[N, 4]).
        probability:flip probility.

    Returns:
        gt_boxes:(shape=[N, 7]).
        points:(shape=[N, 4]).
    """
    applied = _sample_flip_gate(probability)
    if applied:
        _mirror_bev_lateral(gt_boxes, points)
    if return_trans:
        return gt_boxes, points, applied
    return gt_boxes, points


def random_flip_both(
    gt_boxes,
    points,
    probability=0.5,
    flip_coor=None,
    mapGT=None,
    return_trans=False,
):
    """Random flip for points and boxes with mapGT.

    Args:
        gt_boxes:(shape=[N, 7]).
        points:(shape=[N, 4]).
        probability:flip probility.

    Returns:
        gt_boxes:(shape=[N, 4]).
        points:(shape=[N, 4]).
    """
    lateral_flip = _sample_flip_gate(probability)
    if lateral_flip:
        _mirror_bev_lateral(gt_boxes, points)
        if mapGT is not None:
            mapGT = mapGT[::-1, :]

    longitudinal_flip = _sample_flip_gate(probability)
    if longitudinal_flip:
        _mirror_bev_longitudinal(gt_boxes, points, flip_coor)
        if mapGT is not None:
            mapGT = mapGT[:, ::-1]
    if return_trans:
        flip_flags = (lateral_flip, longitudinal_flip)
        if mapGT is not None:
            return gt_boxes, points, mapGT, flip_flags
        return gt_boxes, points, flip_flags
    else:
        if mapGT is not None:
            return gt_boxes, points, mapGT
        return gt_boxes, points


def global_scaling_v2(
    gt_boxes: np.ndarray,
    points: np.ndarray,
    mapGT: np.ndarray = None,
    pivot: np.ndarray = None,
    min_scale: float = 0.95,
    max_scale: float = 1.05,
    return_trans: bool = False,
):
    """Scaling for points and boxes.

    V2 make random scale in range min_scale-max_scale.

    Args:
        gt_boxes:(shape=[N, 7]).
        points:(shape=[N, 4]).
        min_scale: scale min value.
        max_scale: scale max value.

    Returns:
        gt_boxes:(shape=[N, 4]).
        points:(shape=[N, 4]).
    """
    noise_scale = np.random.uniform(min_scale, max_scale)
    points[:, :3] *= noise_scale
    gt_boxes[:, :-1] *= noise_scale
    if mapGT is not None:
        new_pivot = [pivot[0], pivot[1]]
        imgX, imgY = mapGT.shape[1], mapGT.shape[0]
        RimgX, RimgY = round(imgX * noise_scale), round(imgY * noise_scale)
        mapGT = cv2.resize(
            mapGT, (RimgX, RimgY), interpolation=cv2.INTER_NEAREST
        )
        new_pivot[0], new_pivot[1] = round(pivot[0] * noise_scale), round(
            pivot[1] * noise_scale
        )
        if noise_scale >= 1:
            startX = new_pivot[1] - pivot[1]
            startY = new_pivot[0] - pivot[0]
            mapGT = mapGT[startY : startY + imgY, startX : startX + imgX]
        else:
            bpadX = pivot[1] - new_pivot[1]
            apadX = imgX - RimgX - bpadX
            bpadY = pivot[0] - new_pivot[0]
            apadY = imgY - RimgY - bpadY
            mapGT = np.pad(
                mapGT,
                ((bpadY, apadY), (bpadX, apadX)),
                mode="constant",
                constant_values=-1,
            )
        if return_trans:
            return gt_boxes, points, mapGT, pivot, noise_scale
        else:
            return gt_boxes, points, mapGT, pivot
    if return_trans:
        return gt_boxes, points, noise_scale
    else:
        return gt_boxes, points


def global_rotation_v2(
    gt_boxes: np.ndarray,
    points: np.ndarray,
    min_rad: float = -np.pi / 4,
    max_rad: float = np.pi / 4,
):
    """Rotation for points and boxes.

    V2 samples a random angle in range [min_rad, max_rad].

    Args:
        gt_boxes:(shape=[N, 7]).
        points:(shape=[N, 4]).
        min_rad: min rotate angle.
        max_rad: max rotate angle.

    Returns:
        gt_boxes:(shape=[N, 7]).
        points:(shape=[N, 4]).
    """
    noise_rotation = np.random.uniform(min_rad, max_rad)
    rotated_pts = box_np_ops.rotation_points_single_angle(
        points[:, :3], noise_rotation, axis=2
    )
    rotated_centers = box_np_ops.rotation_points_single_angle(
        gt_boxes[:, :3], noise_rotation, axis=2
    )
    points[:, :3] = rotated_pts
    gt_boxes[:, :3] = rotated_centers
    gt_boxes[:, -1] = gt_boxes[:, -1] + noise_rotation
    return gt_boxes, points


@lazy_jit(nopython=True)
def box_collision_test(
    boxes: np.ndarray, qboxes: np.ndarray, clockwise: bool = True
) -> np.ndarray:
    """Test collision between boxes and qboxes.

    Args:
        boxes:(shape=[N, 4, 2]) corner boxes.
        qboxes:(shape=[K, 4, 2]) query corner boxes.
        clockwise: clockwise or counterclockwise.

    Returns:
        ret:(shape=[N, K]).
    """
    N = boxes.shape[0]
    K = qboxes.shape[0]
    ret = np.zeros((N, K), dtype=np.bool_)
    next_corner = np.array([1, 2, 3, 0])
    lines_boxes = np.stack((boxes, boxes[:, next_corner, :]), axis=2)
    lines_qboxes = np.stack((qboxes, qboxes[:, next_corner, :]), axis=2)
    boxes_standup = box_np_ops.corner_to_standup_nd_jit(boxes)
    qboxes_standup = box_np_ops.corner_to_standup_nd_jit(qboxes)
    for i in range(N):
        for j in range(K):
            standup_w = min(boxes_standup[i, 2], qboxes_standup[j, 2]) - max(
                boxes_standup[i, 0], qboxes_standup[j, 0]
            )
            if standup_w <= 0:
                continue
            standup_h = min(boxes_standup[i, 3], qboxes_standup[j, 3]) - max(
                boxes_standup[i, 1], qboxes_standup[j, 1]
            )
            if standup_h <= 0:
                continue

            edges_intersect = False
            for k in range(4):
                for ll in range(4):
                    A = lines_boxes[i, k, 0]
                    B = lines_boxes[i, k, 1]
                    C = lines_qboxes[j, ll, 0]
                    D = lines_qboxes[j, ll, 1]
                    # A/B on opposite sides of segment CD?
                    a_side_of_cd = (D[1] - A[1]) * (C[0] - A[0]) > (
                        C[1] - A[1]
                    ) * (D[0] - A[0])
                    b_side_of_cd = (D[1] - B[1]) * (C[0] - B[0]) > (
                        C[1] - B[1]
                    ) * (D[0] - B[0])
                    if a_side_of_cd == b_side_of_cd:
                        continue
                    # C/D on opposite sides of segment AB?
                    c_side_of_ab = (C[1] - A[1]) * (B[0] - A[0]) > (
                        B[1] - A[1]
                    ) * (C[0] - A[0])
                    d_side_of_ab = (D[1] - A[1]) * (B[0] - A[0]) > (
                        B[1] - A[1]
                    ) * (D[0] - A[0])
                    if c_side_of_ab != d_side_of_ab:
                        edges_intersect = True
                        break
                if edges_intersect:
                    break
            if edges_intersect:
                ret[i, j] = True
                continue

            # Complete containment: box contains qbox.
            box_contains_qbox = True
            for ll in range(4):
                for k in range(4):
                    edge_vec = boxes[i, k] - boxes[i, (k + 1) % 4]
                    if clockwise:
                        edge_vec = -edge_vec
                    cross = edge_vec[1] * (boxes[i, k, 0] - qboxes[j, ll, 0])
                    cross -= edge_vec[0] * (boxes[i, k, 1] - qboxes[j, ll, 1])
                    if cross >= 0:
                        box_contains_qbox = False
                        break
                if not box_contains_qbox:
                    break
            if box_contains_qbox:
                ret[i, j] = True
                continue

            # Complete containment: qbox contains box.
            qbox_contains_box = True
            for ll in range(4):
                for k in range(4):
                    edge_vec = qboxes[j, k] - qboxes[j, (k + 1) % 4]
                    if clockwise:
                        edge_vec = -edge_vec
                    cross = edge_vec[1] * (qboxes[j, k, 0] - boxes[i, ll, 0])
                    cross -= edge_vec[0] * (qboxes[j, k, 1] - boxes[i, ll, 1])
                    if cross >= 0:
                        qbox_contains_box = False
                        break
                if not qbox_contains_box:
                    break
            if qbox_contains_box:
                ret[i, j] = True
    return ret


def global_translate_(
    gt_boxes: np.ndarray,
    points: np.ndarray,
    noise_translate_std: Any,
    return_trans: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply global translation to gt_boxes and points.

    Args:
        boxes:(shape=[N, 7]).
        points:(shape=[N, 4]).

    Returns:
        gt_boxes:(shape=[N, 4]).
        points:(shape=[N, 4]).

    """
    if not isinstance(noise_translate_std, (list, tuple, np.ndarray)):
        noise_translate_std = np.array(
            [noise_translate_std, noise_translate_std, noise_translate_std]
        )
    if all([e == 0 for e in noise_translate_std]):
        if return_trans:
            return gt_boxes, points, None
        else:
            return gt_boxes, points
    noise_translate = np.array(
        [
            np.random.normal(0, noise_translate_std[0], 1),
            np.random.normal(0, noise_translate_std[1], 1),
            np.random.normal(0, noise_translate_std[0], 1),
        ]
    ).T

    points[:, :3] += noise_translate
    gt_boxes[:, :3] += noise_translate

    if return_trans:
        return gt_boxes, points, noise_translate
    else:
        return gt_boxes, points
