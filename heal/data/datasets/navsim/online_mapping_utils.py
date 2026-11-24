# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import Union

import numpy as np
import torch


def cal_line_ver_dist(
    line_1: Union[np.array, torch.Tensor],
    line_2: Union[np.array, torch.Tensor],
) -> Union[np.array, torch.Tensor]:
    r"""Calculate vertical distance between two lines.

    Args:
        line_1: numpy ndarray or torch Tensor [n_points_x, n_dims]
            first point cloud
        line_2: numpy ndarray or torch Tensor [n_points_y, n_dims]
            second point cloud
    Returns:
        res: nupmy ndarray or torch Tensor [n_points_x, n_points_y]
    """  # noqa
    if isinstance(line_1, torch.Tensor):
        assert isinstance(line_2, torch.Tensor)
        return _cal_line_ver_dist_torch_impl(line_1, line_2)
    else:
        return _cal_line_ver_dist_numpy_impl(line_1, line_2)


def _cal_line_ver_dist_torch_impl(line_1, line_2):
    line_1 = torch.unsqueeze(line_1, dim=1)
    line_num = line_2.shape[0]
    # one point
    if line_num == 1:
        line_start = line_2
        line_end = line_2
    else:
        line_start = line_2[: line_num - 1]
        line_end = line_2[1:]

    line_start = torch.unsqueeze(line_start, axis=0)
    line_end = torch.unsqueeze(line_end, dim=0)
    segment = line_end - line_start
    point_to_start = line_1 - line_start
    projection = (
        point_to_start[:, :, 0] * segment[:, :, 0]
        + point_to_start[:, :, 1] * segment[:, :, 1]
    ) / (
        torch.square(segment[:, :, 0])
        + torch.square(segment[:, :, 1])
        + float(np.finfo(float).eps)
    )  # [n, m]
    zeros = torch.zeros(
        (line_1.shape[0], line_start.shape[0]), device=projection.device
    )
    res_1 = torch.where(
        projection <= 0.0, torch.linalg.norm(point_to_start, dim=2), zeros
    )

    res_2 = torch.where(
        projection >= 1.0,
        torch.linalg.norm(line_1 - line_end, dim=2),
        zeros,
    )
    projection_ex = torch.unsqueeze(projection, dim=2)
    closest_point = line_start + projection_ex * segment
    res_3 = torch.where(
        (projection > 0.0) & (projection < 1.0),
        torch.linalg.norm((line_1 - closest_point), dim=2),
        zeros,
    )
    return res_1 + res_2 + res_3


def _cal_line_ver_dist_numpy_impl(line_1, line_2):
    line_1 = np.expand_dims(line_1, axis=1)
    line_num = line_2.shape[0]
    # one point
    if line_num == 1:
        line_start = line_2
        line_end = line_2
    else:
        line_start = line_2[: line_num - 1]
        line_end = line_2[1:]

    line_start = np.expand_dims(line_start, axis=0)
    line_end = np.expand_dims(line_end, axis=0)
    segment = line_end - line_start
    point_to_start = line_1 - line_start
    projection = (
        point_to_start[:, :, 0] * segment[:, :, 0]
        + point_to_start[:, :, 1] * segment[:, :, 1]
    ) / (
        np.square(segment[:, :, 0])
        + np.square(segment[:, :, 1])
        + np.finfo(float).eps
    )  # [n, m]

    zeros = np.zeros((line_1.shape[0], line_start.shape[0]))
    res_1 = np.where(
        projection <= 0.0, np.linalg.norm(point_to_start, axis=2), zeros
    )
    res_2 = np.where(
        projection >= 1.0, np.linalg.norm(line_1 - line_end, axis=2), zeros
    )
    projection_ex = np.expand_dims(projection, axis=2)
    closest_point = line_start + projection_ex * segment
    res_3 = np.where(
        (projection > 0.0) & (projection < 1.0),
        np.linalg.norm((line_1 - closest_point), axis=2),
        zeros,
    )
    return res_1 + res_2 + res_3


def vertical_distance(
    x: Union[np.array, torch.Tensor],
    y: Union[np.array, torch.Tensor],
    direction: str = "bi",
    dist_thresh: float = 10.0,
) -> float:
    r"""Vertical distance between two point clouds.

    Args:
        x: numpy ndarray or torch Tensor [n_points_x, n_dims]
            first point cloud
        y: numpy ndarray or torch Tensor [n_points_y, n_dims]
            second point cloud
        direction: str
            direction of Vertical distance.
                "y_to_x": computes average minimal distance
                          from every point in y to x
                "x_to_y": computes average minimal distance
                          from every point in x to y
                "bi": compute both
    Returns:
        vertical_dist: float
    """  # noqa

    def _cal_min_mean(data):
        if isinstance(data, torch.Tensor):
            return torch.mean(torch.min(res, dim=1)[0], dim=0)
        else:
            assert isinstance(data, np.ndarray)
            return np.mean(np.min(res, axis=1))

    if direction == "x_to_y":
        if len(y) == 0:
            ver_dist = dist_thresh
        else:
            res = cal_line_ver_dist(x, y)
            if res.shape[0] == 0:
                ver_dist = 0.0
            else:
                ver_dist = _cal_min_mean(res)
    elif direction == "y_to_x":
        if len(x) == 0:
            ver_dist = dist_thresh
        else:
            res = cal_line_ver_dist(y, x)
            if res.shape[0] == 0:
                ver_dist = 0.0
            else:
                ver_dist = _cal_min_mean(res)

    elif direction == "bi":
        if len(y) == 0:
            ver_dist_x_to_y = dist_thresh
        else:
            res = cal_line_ver_dist(x, y)
            if res.shape[0] == 0:
                ver_dist_x_to_y = 0.0
            else:
                ver_dist_x_to_y = _cal_min_mean(res)
        if len(x) == 0:
            ver_dist_y_to_x = dist_thresh
        else:
            res = cal_line_ver_dist(y, x)
            if res.shape[0] == 0:
                ver_dist_y_to_x = 0.0
            else:
                ver_dist_y_to_x = _cal_min_mean(res)

        ver_dist = ver_dist_x_to_y + ver_dist_y_to_x
    else:
        raise ValueError(
            'Invalid direction type. Supported types: "y_x", "x_y", "bi"'
        )

    return ver_dist


def _min_vertical_distance(line1, line2):
    """Calculate minimum vertical distance between two lines."""
    return min(
        vertical_distance(line1, line2, direction="x_to_y"),
        vertical_distance(line1, line2, direction="y_to_x"),
    )


def _get_valid_aabb(candi_pts):
    """Compute axis-aligned bounding boxes for all candidate lines.

    Args:
        candi_pts: (N, M, 2) array of candidate line points.

    Returns:
        aabb_min: (N, 2) min corner of each AABB.
        aabb_max: (N, 2) max corner of each AABB.
    """
    aabb_min = np.nanmin(candi_pts, axis=1)
    aabb_max = np.nanmax(candi_pts, axis=1)
    return aabb_min, aabb_max


def _aabb_min_dist(aabb_min, aabb_max):
    """Compute minimum distance between all pairs of AABBs.

    Args:
        aabb_min: (N, 2) min corners.
        aabb_max: (N, 2) max corners.

    Returns:
        dist_matrix: (N, N) matrix of minimum AABB distances.
    """
    # (N, 1, 2) vs (1, N, 2) -> (N, N, 2)
    diff_min = aabb_min[:, np.newaxis, :] - aabb_max[np.newaxis, :, :]
    diff_max = aabb_min[np.newaxis, :, :] - aabb_max[:, np.newaxis, :]
    # If AABBs overlap in a dimension, distance is 0 in that dimension
    overlap = (diff_min <= 0) & (diff_max <= 0)
    # Separation in each dimension
    dist_per_dim = np.maximum(np.maximum(diff_min, diff_max), 0.0)
    dist_per_dim[overlap] = 0.0
    return np.sqrt(np.sum(dist_per_dim**2, axis=2))


def nms_candidates_pts(candi_scores, candi_pts, candi_labels, min_close_dist):
    """
    Filter candidate points, perform NMS and return retained candidates.

    Uses AABB pre-filter + score-descending greedy NMS + set-based deletion:
    1. AABB pre-filter to quickly exclude non-overlapping line pairs
    2. Score-descending greedy strategy (high scores retained first,
       suppress low-score neighbors)
    3. Set-based deletion for O(1) lookup

    Args:
        candi_scores: (N,) score tensor of candidates.
        candi_pts: (N, M, 2) coordinate tensor of candidates.
        candi_labels: (N, 1) label tensor of candidates.
        min_close_dist: minimum distance threshold between candidates.

    Returns:
        Tuple[ndarray, ndarray, ndarray]: retained candidate scores,
        labels and coordinates.
    """
    n = candi_scores.shape[0]
    if n <= 1:
        return candi_scores, candi_labels, candi_pts

    # Step 1: AABB pre-filter
    aabb_min, aabb_max = _get_valid_aabb(candi_pts)
    aabb_dist = _aabb_min_dist(aabb_min, aabb_max)
    # Only compute precise distance for pairs whose AABBs are close enough
    need_precise = aabb_dist < min_close_dist

    # Step 2: Sort by score descending (greedy NMS)
    order = np.argsort(-candi_scores)

    # Step 3: Greedy NMS with set-based deletion
    suppressed = set()
    for idx in range(n):
        i = order[idx]
        if i in suppressed:
            continue
        # Only check candidates with lower score (j > idx in order)
        for jdx in range(idx + 1, n):
            j = order[jdx]
            if j in suppressed:
                continue
            # AABB pre-filter: skip if AABBs are far apart
            if not need_precise[i, j]:
                continue
            # Precise distance check
            dist = min(
                vertical_distance(
                    candi_pts[i], candi_pts[j], direction="x_to_y"
                ),
                vertical_distance(
                    candi_pts[i], candi_pts[j], direction="y_to_x"
                ),
            )
            if dist < min_close_dist:
                suppressed.add(j)

    # Step 4: Collect surviving candidates
    graduate_list = [i for i in range(n) if i not in suppressed]

    graduated_scores = candi_scores[graduate_list]
    graduated_labels = candi_labels[graduate_list]
    graduated_pts = candi_pts[graduate_list]

    return graduated_scores, graduated_labels, graduated_pts
