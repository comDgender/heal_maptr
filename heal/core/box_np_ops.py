# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import List, Tuple

import numpy as np

from heal.core.box_utils import corners_nd
from heal.core.numba_lazy import lazy_jit, lazy_njit
from heal.core.point_geometry import points_in_convex_polygon_3d_jit

corner_idxes = np.array(
    [
        0,
        1,
        2,
        3,
        7,
        6,
        5,
        4,
        0,
        3,
        7,
        4,
        1,
        5,
        6,
        2,
        0,
        4,
        5,
        1,
        3,
        2,
        6,
        7,
    ]
).reshape(6, 4)


def points_in_rbbox(
    points: np.ndarray,
    rbbox: np.ndarray,
    z_axis: int = 2,
    origin: Tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> np.ndarray:
    """Find points in each rubberband box.

    Args:
        points : [num_points, >=3] points array.
        rbbox : [num_boxes, 8] rubberband boxes array.
        z_axis : dimension index where z-axis is lcoated.
            Defaults to 2.
        origin : Origin point.
            Defaults to (0.5, 0.5, 0.5).

    Returns:
        indices of points in rubberband box.
    """
    rbbox_corners = center_to_corner_box3d(
        rbbox[:, :3], rbbox[:, 3:6], rbbox[:, -1], origin=origin, axis=z_axis
    )
    surfaces = corner_to_surfaces_3d(rbbox_corners)
    indices = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
    return indices


def corner_to_surfaces_3d(corners: np.ndarray) -> np.ndarray:
    """Convert 3d box corners.

    From corner function above to surfaces with normal vectors all directing
    to internal.

    Args:
        corners : [N, 8, 3], must from corner functions in this module.
    Returns:
        surfaces : surce array.
    """
    # Gather face vertices via shared index table: [N, 8, 3] -> [N, 6, 4, 3].
    return corners[:, corner_idxes]


@lazy_jit(nopython=True)
def corner_to_surfaces_3d_jit(corners: np.ndarray) -> np.ndarray:
    """Convert 3d box corners.

    From corner function above to surfaces that normal vectors all direct
    to internal.

    Args:
        corners : [N, 8, 3], must from corner functions in this module.
    Returns:
        surfaces : surfaces that normal vectors all direct to internal.
    """
    n_box = corners.shape[0]
    out = np.empty((n_box, 6, 4, 3), dtype=corners.dtype)
    for bi in range(n_box):
        for fi in range(6):
            for vi in range(4):
                src = corner_idxes[fi, vi]
                out[bi, fi, vi, 0] = corners[bi, src, 0]
                out[bi, fi, vi, 1] = corners[bi, src, 1]
                out[bi, fi, vi, 2] = corners[bi, src, 2]
    return out


def rotation_3d_in_axis(
    points: np.ndarray, angles: np.ndarray, axis: int = 0
) -> np.ndarray:
    """Rotate a set of points according to individual angles.

    Args:
        points: [N, ndim, 3] tensor of boxes. if 3d boxes, ndim
            should be 3.
        angles: [N] tensor of angles. Each box has its own
            rotation angle.
        axis : rotation axis. Defaults to 0.

    Raises:
        ValueError: raised when axis num is not in [-1, 2].

    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack(
            [
                [rot_cos, zeros, -rot_sin],
                [zeros, ones, zeros],
                [rot_sin, zeros, rot_cos],
            ]
        )
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack(
            [
                [rot_cos, -rot_sin, zeros],
                [rot_sin, rot_cos, zeros],
                [zeros, zeros, ones],
            ]
        )
    elif axis == 0:
        rot_mat_T = np.stack(
            [
                [zeros, rot_cos, -rot_sin],
                [zeros, rot_sin, rot_cos],
                [ones, zeros, zeros],
            ]
        )
    else:
        raise ValueError("axis should in range")

    return np.einsum("aij,jka->aik", points, rot_mat_T)


def center_to_corner_box3d(
    centers: np.ndarray,
    dims: np.ndarray,
    angles: np.ndarray = None,
    origin: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    axis: int = 2,
) -> np.ndarray:
    """Convert kitti locations, dimensions and angles to corners.

    Args:
        centers : locations in kitti label file.
        dims : dimensions in kitti label file.
        angles : rotation_y in kitti label file.
        origin : origin point relate to smallest point.
            use [0.5, 1.0, 0.5] in camera and [0.5, 0.5, 0] in lidar.
        axis : rotation axis. 1 for camera and 2 for lidar.
    Returns:
        [type]: [description]
    """
    # KITTI length lies on x; camera uses yzx(hwl), lidar uses z(-x)(-y)(wlh).
    local = corners_nd(dims, origin=origin)
    if angles is not None:
        local = rotation_3d_in_axis(local, angles, axis=axis)
    return local + centers.reshape(-1, 1, 3)


def rotation_points_single_angle(
    points: np.ndarray, angle: float, axis: int = 0
) -> np.ndarray:
    """Rotate points around an axis.

    Args:
        points : [N, 3] point array.
        angle : rotation angle.
        axis : axis index. Defaults to 0.

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
        raise ValueError("axis should in range -1, 0, 1, 2")

    return points @ rot_mat_T


def project_to_image(
    points_3d: np.ndarray, proj_mat: np.ndarray
) -> np.ndarray:
    """Convert vehicle world 3dpoints to image 2d points .

    Args:
        points_3d :[N, 3].
        proj_mat: [4, 4].

    Returns:
        points_2d_res :[N, 2].
    """
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1
    points_4 = np.concatenate([points_3d, np.ones(points_shape)], axis=-1)
    point_2d = points_4 @ proj_mat.T
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]

    return point_2d_res


def box3d_to_bbox(
    box3d: np.ndarray, rect: np.ndarray, Trv2c: np.ndarray, P2: np.ndarray
) -> np.ndarray:
    """Convert vehicle world box3d to image 2d box .

    Args:
        box3d:[N, 7].
        rect:[4, 4].
        Trv2c[N, 3] tranlate matix vehicle.
        P2:[N, 3] project matrix.

    Returns:
        bbox:(float array, shape=[N, 8,2]).
    """
    # box3d_to_cam = box_lidar_to_camera(box3d, rect, Trv2c)
    box_corners = center_to_corner_box3d(
        box3d[:, :3], box3d[:, 3:6], box3d[:, 6], [0.5, 1.0, 0.5], axis=1
    )
    box_corners_in_image = project_to_image(box_corners, P2)
    # box_corners_in_image: [N, 8, 2]
    minxy = np.min(box_corners_in_image, axis=1)
    maxxy = np.max(box_corners_in_image, axis=1)
    bbox = np.concatenate([minxy, maxxy], axis=1)
    return bbox


def limit_period(
    val: np.ndarray, offset: float = 0.5, period: float = np.pi
) -> np.ndarray:
    """Limit period.

    Args:
        val:[N, 6] value of period.
        offset:period offset.

    Returns:
        val:[N, 6] value of period.
    """
    return val - np.floor(val / period + offset) * period


def projection_matrix_to_CRT_kitti(
    proj: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert projection_matrix to CRT.

    Args:
        proj:(shape=[4, 4]).

    Returns:
        C:(shape=[3, 3]).
        R:(shape=[3, 3]).
        T:(shape=[3, 1]).
    """
    # P = C @ [R|T]
    # C is upper triangular matrix, so we need to inverse CR and use QR
    # stable for all kitti camera projection matrix
    CR = proj[0:3, 0:3]
    CT = proj[0:3, 3]
    RinvCinv = np.linalg.inv(CR)
    Rinv, Cinv = np.linalg.qr(RinvCinv)
    C = np.linalg.inv(Cinv)
    R = np.linalg.inv(Rinv)
    T = Cinv @ CT
    return C, R, T


def minmax_to_corner_2d_v2(minmax_box: np.ndarray) -> np.ndarray:
    """Box's min max type to center type.

    Convert kitti locations imensions and angles to corners.

    format: center(xy)

    Args:
        minmax_box :(shape=[N, 4]).

    Returns:
        [type]: [description]
    """
    # N, 4 -> N 4 2
    return minmax_box[..., [0, 1, 0, 3, 2, 3, 2, 1]].reshape(-1, 4, 2)


def get_frustum(
    bboxes: np.ndarray,
    C: np.ndarray,
    near_clip: float = 0.001,
    far_clip: float = 100,
) -> np.ndarray:
    fx = C[0, 0]
    fy = -C[1, 1]
    principal = C[0:2, 2]
    n_box = bboxes.shape[0]
    z_pts = np.concatenate(
        [
            np.full((n_box, 4, 1), near_clip, dtype=C.dtype),
            np.full((n_box, 4, 1), far_clip, dtype=C.dtype),
        ],
        axis=1,
    )
    box_corners = minmax_to_corner_2d_v2(bboxes)
    offset = box_corners - principal
    near_xy = offset / np.asarray(
        [fx / near_clip, -fy / near_clip], dtype=C.dtype
    )
    far_xy = offset / np.asarray(
        [fx / far_clip, -fy / far_clip], dtype=C.dtype
    )
    xy = np.concatenate([near_xy, far_xy], axis=1)
    return np.concatenate([xy, z_pts], axis=-1)


def camera_to_lidar(
    points: np.ndarray, r_rect: np.ndarray, velo2cam: np.ndarray
) -> np.ndarray:
    """Convert camera world points to lidar world points .

    Args:
        points :shape=[N, 3] or shape=[N, 4].
        r_rect: shape=[4, 4] matrix rect.
        velo2cam:shape=[4, 4] translate matrix vel2cam.
    Returns:
        points_2d_res : shape=[N, 2].
    """
    points_shape = list(points.shape[0:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    lidar_points = points @ np.linalg.inv((r_rect @ velo2cam).T)
    return lidar_points[..., :3]


@lazy_jit(nopython=True)
def box2d_to_corner_jit(boxes: np.ndarray) -> np.ndarray:
    """Box's center type to corner type.

    Args:
        boxes: shape=[N, 5].

    Returns:
        boxes: shape=[N, 5].
    """
    n_box = boxes.shape[0]
    # Unit-square corners centered at origin, clockwise from bottom-left.
    base = np.array(
        [[-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]],
        dtype=boxes.dtype,
    )
    local = np.empty((n_box, 4, 2), dtype=boxes.dtype)
    for i in range(n_box):
        for j in range(4):
            local[i, j, 0] = boxes[i, 2] * base[j, 0]
            local[i, j, 1] = boxes[i, 3] * base[j, 1]

    rot = np.empty((2, 2), dtype=boxes.dtype)
    out = np.empty((n_box, 4, 2), dtype=boxes.dtype)
    for i in range(n_box):
        s = np.sin(boxes[i, -1])
        c = np.cos(boxes[i, -1])
        rot[0, 0] = c
        rot[0, 1] = -s
        rot[1, 0] = s
        rot[1, 1] = c
        out[i] = local[i] @ rot + boxes[i, :2]
    return out


@lazy_njit
def corner_to_standup_nd_jit(boxes_corner: np.ndarray) -> np.ndarray:
    """Box's corner type to standup corner.

    Args:
        boxes_corner: shape=[N, ndim].

    Returns:
        result: shape=[N, ndim*2].
    """
    num_boxes = boxes_corner.shape[0]
    ndim = boxes_corner.shape[-1]
    result = np.zeros((num_boxes, ndim * 2), dtype=boxes_corner.dtype)
    for i in range(num_boxes):
        for j in range(ndim):
            result[i, j] = np.min(boxes_corner[i, :, j])
        for j in range(ndim):
            result[i, j + ndim] = np.max(boxes_corner[i, :, j])
    return result


def remove_outside_points(
    points: np.ndarray,
    rect: np.ndarray,
    Trv2c: np.ndarray,
    P2: np.ndarray,
    image_shape: List[int],
) -> np.ndarray:
    """Remove point clouds that are distributed outside the image range.

    Args:
        points: Point cloud, shape=[N, 3] or shape=[N, 4].
        rect: matrix rect, shape=[4, 4].
        Trv2c: Translate matrix vel2cam, shape=[4, 4].
        P2: Project matrix, shape=[4, 4].
        image_shape: Image shape, (H, W, ...) format.

    Returns: Point clouds are distributed inside the image range.
    """
    # 5x faster than remove_outside_points_v1(2ms vs 10ms)
    C, R, T = projection_matrix_to_CRT_kitti(P2)
    image_bbox = np.array(
        [[0, 0, image_shape[1], image_shape[0]]], dtype=np.int32
    )

    frustum = get_frustum(image_bbox, C)
    if len(frustum.shape) == 3:
        frustum = frustum.squeeze()
    frustum -= T
    frustum = np.linalg.inv(R) @ frustum.T
    frustum = camera_to_lidar(frustum.T, rect, Trv2c)
    frustum_surfaces = corner_to_surfaces_3d_jit(frustum[np.newaxis, ...])
    indices = points_in_convex_polygon_3d_jit(points[:, :3], frustum_surfaces)
    points = points[indices.reshape([-1])]
    return points


def box_camera_to_lidar(
    camera_box: np.ndarray, r_rect: np.ndarray, velo2cam: np.ndarray
) -> np.ndarray:
    """Convert camera 3d box to lidar 3d box.

    Args:
        camera_box: Camera box, shape=[N, 7], (x, y, z, l, h, w, rot_y).
        r_rect: Matrix rect, shape=[4, 4].
        velo2cam: Translate matrix vel2cam, shape=[4, 4].

    Returns:
        np.ndarray: Lidar 3d box, shape=[N, 7], (x, y, z, h, w, l, rot_y).
    """
    xyz = camera_box[:, 0:3]
    l, h, w = camera_box[:, 3:4], camera_box[:, 4:5], camera_box[:, 5:6]
    r = camera_box[:, 6:7]
    xyz_lidar = camera_to_lidar(xyz, r_rect, velo2cam)
    return np.concatenate([xyz_lidar, w, l, h, r], axis=1)


def change_box3d_center_(box3d, src, dst):
    dst = np.array(dst, dtype=box3d.dtype)
    src = np.array(src, dtype=box3d.dtype)
    # bo3d: [x_liar, y_lidar, z_lidar, w, l, h, rot_y]
    box3d[..., :3] += box3d[..., 3:6] * (dst - src)


def lidar_to_camera(
    points: np.ndarray,
    r_rect: np.ndarray,
    velo2cam: np.ndarray,
) -> np.ndarray:
    """Convert lidar world points to camera world points .

    Args:
        points :shape=[N, 3] or shape=[N, 4].
        r_rect: shape=[4, 4] matrix rect.
        velo2cam:shape=[4, 4] translate matrix vel2cam.
    Returns:
        points_2d_res : shape=[N, 2].
    """
    points_shape = list(points.shape[:-1])
    if points.shape[-1] == 3:
        points = np.concatenate([points, np.ones(points_shape + [1])], axis=-1)
    camera_points = points @ (r_rect @ velo2cam).T
    return camera_points[..., :3]


def box_lidar_to_camera(
    box3d_lidar: np.ndarray,
    r_rect: np.ndarray,
    velo2cam: np.ndarray,
) -> np.ndarray:
    """Convert Lidar3D boxes to Camera3D boxes.

    Args:
        box3d_lidar: Lidar3D boxes, shape=(N, 7).
        r_rect: Matrix rect, shape=[4, 4].
        velo2cam: Translate matrix vel2cam, shape=[4, 4].

    Returns:
        np.ndarray: Camera3D boxes, shape=(N, 7).
    """
    xyz_lidar = box3d_lidar[:, 0:3]
    w, l, h = box3d_lidar[:, 3:4], box3d_lidar[:, 4:5], box3d_lidar[:, 5:6]
    r = box3d_lidar[:, 6:7]
    xyz = lidar_to_camera(xyz_lidar, r_rect, velo2cam)
    return np.concatenate([xyz, l, h, w, r], axis=1)
