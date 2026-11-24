# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

# Implements functions which operate on lidar point domain, such as point
# rotation, point counting, etc.
# Due to the sparsity of the point cloud, these functions rely JIT compilation
# to speed up. Numba is imported on first JIT call (see numba_lazy).
from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np

from heal.core.numba_lazy import lazy_jit, lazy_njit

__all__ = [
    "rotation_points_single_angle",
    "points_count_convex_polygon_3d_jit",
    "points_in_convex_polygon_jit",
    "points_in_convex_polygon_3d_jit",
    "get_rotation_matrix",
    "coor_transformation",
    "_points_to_truncate_points_all",
    "dropout_points_in_box",
    "points_to_truncate_points",
]


def rotation_points_single_angle(
    points: np.ndarray, angle: float, axis: Optional[int] = 0
) -> np.ndarray:
    """Rotate points around an axis.

    Args:
        points: [N, 3] point array.
        angle: rotation angle.
        axis: axis index. Defaults to 0.

    Raises:
        ValueError: if axis is not in [-1, 0, 1, 2].

    Returns:
        rotated points.
    """
    # points: [N, 3]
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    if axis == 1:
        rot_mat_T = np.array(
            [[rot_cos, 0, -rot_sin], [0, 1, 0], [rot_sin, 0, rot_cos]],
            dtype=points.dtype,
        )
    elif axis == 2 or axis == -1:
        rot_mat_T = np.array(
            [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]],
            dtype=points.dtype,
        )
    elif axis == 0:
        rot_mat_T = np.array(
            [[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]],
            dtype=points.dtype,
        )
    else:
        raise ValueError("axis should be in range -1, 0, 1, 2")

    return points @ rot_mat_T


@lazy_njit
def _points_count_convex_polygon_3d_jit(
    points: np.ndarray,
    polygon_surfaces: np.ndarray,
    normal_vec: np.ndarray,
    d: np.ndarray,
    num_surfaces: np.ndarray = None,
) -> np.ndarray:
    """Count points in 3d convex polygons.

    Args:
        points: [num_points, 3] array of points.
        polygon_surfaces: [M, S, P, 3] array of surfaces, where:
            M = number of polygons
            S = max surfaces each polygon has. In cuboid case, S = 6.
            P = max points each surface has. In cuboid case, P = 4.
            The last dimension is each point's coordinates.
        normal_vec: [M, S, 3] normal vector array.
        d: [M, S] distance matrix.
        num_surfaces: [M] array describing how many
            surfaces a polygon contains. Defaults to None.

    Returns:
        ret: number of points in each polygon.
    """
    max_num_surfaces = polygon_surfaces.shape[1]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    ret = np.full((num_polygons,), num_points, dtype=np.int64)
    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = (
                    points[i, 0] * normal_vec[j, k, 0]
                    + points[i, 1] * normal_vec[j, k, 1]
                    + points[i, 2] * normal_vec[j, k, 2]
                    + d[j, k]
                )
                if sign >= 0:
                    ret[j] -= 1
                    break
    return ret


_SURFACE_COUNT_SENTINEL = 9999999


def _resolve_surface_counts(
    num_polygons: int, surface_counts: Optional[np.ndarray]
) -> np.ndarray:
    if surface_counts is None:
        return np.full(
            (num_polygons,), _SURFACE_COUNT_SENTINEL, dtype=np.int64
        )
    return surface_counts


def _convex_hull_plane_coefficients(polygon_surfaces: np.ndarray):
    face_vertices = polygon_surfaces[:, :, :3, :]
    return surface_equ_3d_jitv2(face_vertices)


@lazy_njit
def _plane_signed_distance(point, plane_normal, plane_offset: float) -> float:
    return (
        point[0] * plane_normal[0]
        + point[1] * plane_normal[1]
        + point[2] * plane_normal[2]
        + plane_offset
    )


@lazy_jit
def points_count_convex_polygon_3d_jit(
    points: np.ndarray,
    polygon_surfaces: np.ndarray,
    num_surfaces: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Count number of points in each convex polygon.

    Args:
        points: [N, >=3] array of point locations.
        polygon_surfaces: [M, S, P, 3] array of surfaces, where:
            M = number of polygons
            S = max surfaces each polygon has. In cuboid case, S = 6.
            P = max points each surface has. In cuboid case, P = 4.
            The last dimension is each point's coordinates.
        num_surfaces: number of surfaces.
            Defaults to None.

    Returns:
        number of points in each polygon.
    """
    hull_count = polygon_surfaces.shape[0]
    face_limits = _resolve_surface_counts(hull_count, num_surfaces)
    plane_normals, plane_offsets = _convex_hull_plane_coefficients(
        polygon_surfaces
    )
    return _points_count_convex_polygon_3d_jit(
        points, polygon_surfaces, plane_normals, plane_offsets, face_limits
    )


@lazy_njit
def _points_in_convex_polygon_3d_jit(
    points: np.ndarray,
    polygon_surfaces: np.ndarray,
    normal_vec: np.ndarray,
    d: np.ndarray,
    num_surfaces: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Check points is in 3d convex polygons.

    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces,
            max_num_points_of_surface, 3] array. All surfaces' normal vector
            must direct to internal.
            max_num_points_of_surface must at least 3.
        normal_vec: normal vectors.
        d: distance matrix.
        num_surfaces: [num_polygon] array. indicate how many surfaces
            a polygon contain.
    Returns:
        [num_points, num_polygon] bool array.
    """
    max_faces = polygon_surfaces.shape[1]
    n_points = points.shape[0]
    n_hulls = polygon_surfaces.shape[0]
    inside_mask = np.ones((n_points, n_hulls), dtype=np.bool_)

    for pt_idx in range(n_points):
        sample = points[pt_idx]
        for hull_idx in range(n_hulls):
            active_faces = num_surfaces[hull_idx]
            is_inside = True
            for face_idx in range(max_faces):
                if face_idx > active_faces:
                    break
                signed_dist = _plane_signed_distance(
                    sample,
                    normal_vec[hull_idx, face_idx],
                    d[hull_idx, face_idx],
                )
                if signed_dist >= 0.0:
                    is_inside = False
                    break
            inside_mask[pt_idx, hull_idx] = is_inside
    return inside_mask


def points_in_convex_polygon_3d_jit(
    points: np.ndarray,
    polygon_surfaces: np.ndarray,
    num_surfaces: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Test whether 3D points lie inside convex polyhedral regions.

    Args:
        points: Query locations with shape (N, 3).
        polygon_surfaces: Face corner tensors with shape
            (M, max_faces, max_corners, 3); face normals must point inward.
        num_surfaces: Optional per-hull active face counts, shape (M,).

    Returns:
        Boolean membership mask of shape (N, M).
    """
    hull_count = polygon_surfaces.shape[0]
    face_limits = _resolve_surface_counts(hull_count, num_surfaces)
    plane_normals, plane_offsets = _convex_hull_plane_coefficients(
        polygon_surfaces
    )
    return _points_in_convex_polygon_3d_jit(
        points,
        polygon_surfaces,
        plane_normals,
        plane_offsets,
        face_limits,
    )


@lazy_njit
def _cross_product_3d(left, right):
    result = np.empty(3, dtype=left.dtype)
    result[0] = left[1] * right[2] - left[2] * right[1]
    result[1] = left[2] * right[0] - left[0] * right[2]
    result[2] = left[0] * right[1] - left[1] * right[0]
    return result


@lazy_njit
def _plane_from_triangle(vertex_a, vertex_b, vertex_c):
    edge_ab = vertex_a - vertex_b
    edge_bc = vertex_b - vertex_c
    unit_normal = _cross_product_3d(edge_ab, edge_bc)
    plane_offset = -(
        vertex_a[0] * unit_normal[0]
        + vertex_a[1] * unit_normal[1]
        + vertex_a[2] * unit_normal[2]
    )
    return unit_normal, plane_offset


@lazy_njit
def surface_equ_3d_jitv2(surfaces: np.ndarray) -> Tuple[np.ndarray]:
    """Derive unnormalized plane normals and offsets for triangular faces.

    Args:
        surfaces: Face vertices with shape (M, S, 3, 3).

    Returns:
        Tuple of plane normals (M, S, 3) and offsets (M, S).
    """
    hull_count = surfaces.shape[0]
    face_capacity = surfaces.shape[1]
    normals = np.zeros((hull_count, face_capacity, 3), dtype=surfaces.dtype)
    offsets = np.zeros((hull_count, face_capacity), dtype=surfaces.dtype)
    for hull_idx in range(hull_count):
        for face_idx in range(face_capacity):
            v0 = surfaces[hull_idx, face_idx, 0]
            v1 = surfaces[hull_idx, face_idx, 1]
            v2 = surfaces[hull_idx, face_idx, 2]
            face_normal, face_offset = _plane_from_triangle(v0, v1, v2)
            normals[hull_idx, face_idx] = face_normal
            offsets[hull_idx, face_idx] = face_offset
    return normals, offsets


@lazy_jit
def points_in_convex_polygon_jit(
    points: np.ndarray, polygon: np.ndarray, clockwise: bool = True
) -> np.ndarray:
    """Check points is in 2d convex polygons. True when point in polygon.

    Args:
        points: [num_points, 2] array.
        polygon: [num_polygon, num_points_of_polygon, 2] array.
        clockwise: bool. indicate polygon is clockwise.
    Returns:
        [num_points, num_polygon] bool array.
    """
    # first convert polygon to directed lines
    num_points_of_polygon = polygon.shape[1]
    num_points = points.shape[0]
    num_polygons = polygon.shape[0]
    if clockwise:
        vec1 = polygon - np.concatenate(
            (
                polygon[:, -1:, :],
                polygon[:, :-1, :],
            ),
            axis=1,
        )
    else:
        vec1 = (
            np.concatenate(
                (
                    polygon[:, -1:, :],
                    polygon[:, :-1, :],
                ),
                axis=1,
            )
            - polygon
        )
    # vec1: [num_polygon, num_points_of_polygon, 2]
    ret = np.zeros((num_points, num_polygons), dtype=np.bool_)
    success = True
    cross = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            success = True
            for k in range(num_points_of_polygon):
                cross = vec1[j, k, 1] * (polygon[j, k, 0] - points[i, 0])
                cross -= vec1[j, k, 0] * (polygon[j, k, 1] - points[i, 1])
                if cross >= 0:
                    success = False
                    break
            ret[i, j] = success
    return ret


def get_rotation_matrix(theta, translation=None):
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    rotation_mat = np.stack(
        [
            np.stack([cos_theta, -sin_theta], axis=-1),
            np.stack([sin_theta, cos_theta], axis=-1),
        ],
        axis=-2,
    )
    if translation is not None:
        padding_size = ((0, 0),) * (len(rotation_mat.shape) - 2)
        print(padding_size + ((0, 1), (0, 1)))
        expand_rot_mat = np.pad(
            rotation_mat, padding_size + ((0, 1), (0, 1)), mode="constant"
        )
        expand_rot_mat[..., 2, 2] = 1
        expand_rot_mat[..., :2, 2] = np.array(translation)
        return expand_rot_mat
    else:
        return rotation_mat


def coor_transformation(x, theta=None, translation=None):
    y = x.copy()
    if theta is not None:
        rotation_mat = get_rotation_matrix(theta)
        y = np.sum(y[..., None, :] * rotation_mat, axis=-1)
    if translation is not None:
        y += translation
    return y


@lazy_njit
def _points_to_truncate_points_all(
    points: np.ndarray,
    angle_range: Tuple[float, float] = (0, 180),
    center: Tuple[float, float, float] = (0, 0, 0),
    axis_rotat: float = 90,
) -> List[int]:
    """
    Axis_rotat:Degrees of clockwise rotation.

    Args:
        points: [num_points, 3].
        angle_range : truncate angle.
        center : truncate center point.
        axis_rotat: aixs rotat angle.
    Returns:
        [num_polygon] array.
    """
    keep_list = []
    for i, v in enumerate(points):
        x = v[0] - center[0]
        y = v[1] - center[1]
        if x > 0 and y >= 0:
            rad = np.arctan(y / x)
        elif x <= 0 and y > 0:
            rad = np.pi / 2 - np.arctan(x / y)
        elif x < 0 and y <= 0:
            rad = np.pi + np.arctan(y / x)
        elif x >= 0 and y < 0:
            rad = np.pi * 3 / 2 - np.arctan(x / y)
        angle = (np.rad2deg(rad) + axis_rotat) % 360

        if angle_range[0] <= angle <= angle_range[1]:
            keep_list.append(i)
    return keep_list


@lazy_njit
def _filter_points_to_trancate_points_polygon_3d_jit(
    points: np.ndarray,
    polygon_surfaces: np.ndarray,
    normal_vec: np.ndarray,
    d: np.ndarray,
    angle_range: Optional[Tuple[float, float]] = (0, 180),
) -> List[int]:
    """
    Kickout points in 3d convex polygons which not in the range.

    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces,
            max_num_points_of_surface, 3]
            array. all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.

    Returns:
        [num_polygon] array.
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    sign = 0.0
    delet_point_idx = []
    for i in range(num_points):
        sign_flag = True
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                sign = (
                    points[i, 0] * normal_vec[j, k, 0]
                    + points[i, 1] * normal_vec[j, k, 1]
                    + points[i, 2] * normal_vec[j, k, 2]
                    + d[j, k]
                )
                sign_flag = sign < 0 and sign_flag
            if sign_flag:
                delet_point_idx.append(i)
                break
    return delet_point_idx


def points_to_truncate_points(
    points: np.ndarray,
    polygon_surfaces: np.ndarray,
    ther_angle: Tuple[float, float] = (0, 180),
) -> np.ndarray:
    """Check points is in 3d convex polygons.

    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces,
            max_num_points_of_surface, 3]
            array. all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.

    Returns:
       truncate_poins
    """
    normal_vec, d = surface_equ_3d_jitv2(polygon_surfaces[:, :, :3, :])
    # normal_vec: [num_polygon, max_num_surfaces, 3] [n,6,3]
    # d: [num_polygon, max_num_surfaces]    #
    delet_point_idx = _filter_points_to_trancate_points_polygon_3d_jit(
        points, polygon_surfaces, normal_vec, d, ther_angle
    )

    points = np.delete(points, delet_point_idx, axis=0)
    return points


def dropout_points_in_box(
    points: np.ndarray,
    polygon_surfaces: np.ndarray,
    prob: Optional[float] = 0.2,
) -> np.ndarray:
    """Check points is in 3d convex polygons.

    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces,
            max_num_points_of_surface, 3]
            array. all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.

    Returns:
       truncate_poins
    """
    normal_vec, d = surface_equ_3d_jitv2(polygon_surfaces[:, :, :3, :])
    # normal_vec: [num_polygon, max_num_surfaces, 3] [n,6,3]
    # To express each face with three-dimensional vector
    # only three points are needed:
    # (vector representation of three-dimensional plane)
    # d: [num_polygon, max_num_surfaces]
    prob_array = np.random.random((points.shape[0], 1))
    points_temp = deepcopy(points)
    points_temp = np.hstack([points_temp, prob_array])
    # Since the internal operation of numba does
    # not support numpy random numbers,
    # the probability is generated externally

    delet_point_idx = _dropout_points_onbox_polygon_3d_jit(
        points_temp, polygon_surfaces, normal_vec, d, prob
    )

    points = np.delete(points, delet_point_idx, axis=0)

    return points


@lazy_njit
def _dropout_points_onbox_polygon_3d_jit(
    points: np.ndarray,
    polygon_surfaces: np.ndarray,
    normal_vec: np.ndarray,
    d: np.ndarray,
    prob: float,
) -> List[int]:
    """Dropout points in 3d convex polygons.

    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces,
            max_num_points_of_surface, 3]
            array. all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.

    Returns:
        [dropout points idx] array.
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    delet_point_idx = []
    for i in range(num_points):
        sign_flag = True
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                sign = (
                    points[i, 0] * normal_vec[j, k, 0]
                    + points[i, 1] * normal_vec[j, k, 1]
                    + points[i, 2] * normal_vec[j, k, 2]
                    + d[j, k]
                )
                sign_flag = sign < 0 and sign_flag
            if sign_flag and points[i, -1] < prob:
                delet_point_idx.append(i)
                break
    return delet_point_idx
