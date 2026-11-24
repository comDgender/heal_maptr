# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

# Copyright (c) Open-MMLab. All rights reserved.
# @misc{mmdet3d2020,
#     title={{MMDetection3D: OpenMMLab} next-generation platform for general {3D} object detection},  # noqa E501
#     author={MMDetection3D Contributors},
#     howpublished = {\url{https://github.com/open-mmlab/mmdetection3d}},
#     year={2020}
# }
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Added line36-line52, line88-line91, line100-line105, line142-line151, line451-line496             # noqa E501

"""Numba/CUDA implementation.

Import via heal.core.rotate_box_utils_v2 only when rotate IoU is needed.
Skipped by heal.register_default_config so training jobs do not initialize
numba.cuda.
"""

import logging
import math
import warnings
from typing import Optional

from heal.core.numba_cuda_init import configure_numba_cuda

configure_numba_cuda()

import numba
import numpy as np
from hat_engine.utils.numba_helper import numba_jit
from numba import cuda

try:
    from numba.core.errors import (
        NumbaDeprecationWarning,
        NumbaPerformanceWarning,
        NumbaWarning,
    )

    warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
    warnings.filterwarnings("ignore", category=NumbaWarning)
except ImportError:
    pass

numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)
numba_cuda_available = cuda.is_available()
EPSILON = 1e-8
THREADS_PER_BLOCK = 8 * 8
# cuda.shared.array shape must be a compile-time literal (Numba >= 0.59).
_SHARED_BOX_ARRAY_LEN = 64 * 5
assert _SHARED_BOX_ARRAY_LEN == THREADS_PER_BLOCK * 5


@numba.jit(nopython=True)
def div_up(m, n):
    return m // n + (m % n > 0)


@numba_jit(
    "(float32[:], float32[:], float32[:])",
    device=numba_cuda_available,
    inline=True,
)
def trangle_area(a, b, c):
    return (
        (a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0])
    ) / 2.0


@numba_jit("(float32[:], int32)", device=numba_cuda_available, inline=True)
def area(int_pts, num_of_inter):
    area_val = 0.0
    for i in range(num_of_inter - 2):
        area_val += abs(
            trangle_area(
                int_pts[:2],
                int_pts[2 * i + 2 : 2 * i + 4],
                int_pts[2 * i + 4 : 2 * i + 6],
            )
        )
    return area_val


@numba_jit(nopython=True, inline=True)
def _vertex_sort_key_numba(x, y, center_x, center_y):
    v0 = x - center_x
    v1 = y - center_y
    d = max(math.sqrt(v0 * v0 + v1 * v1), EPSILON)
    v0 /= d
    v1 /= d
    return -2 - v0 if v1 < 0 else v0


@numba_jit(
    "(float32[:], float32[:], int32)", device=numba_cuda_available, inline=True
)
def _insertion_sort_polygon_vertices_numba(int_pts, vs, num_of_inter):
    for i in range(1, num_of_inter):
        if vs[i - 1] <= vs[i]:
            continue
        sort_key = vs[i]
        tx = int_pts[2 * i]
        ty = int_pts[2 * i + 1]
        j = i
        while j > 0 and vs[j - 1] > sort_key:
            vs[j] = vs[j - 1]
            int_pts[j * 2] = int_pts[j * 2 - 2]
            int_pts[j * 2 + 1] = int_pts[j * 2 - 1]
            j -= 1
        vs[j] = sort_key
        int_pts[j * 2] = tx
        int_pts[j * 2 + 1] = ty


@numba_jit("(float32[:], int32)", device=numba_cuda_available, inline=True)
def sort_vertex_in_convex_polygon_numba(int_pts, num_of_inter):
    if num_of_inter <= 0:
        return

    if numba_cuda_available:
        center = cuda.local.array((2,), dtype=numba.float32)
        vs = cuda.local.array((16,), dtype=numba.float32)
    else:
        center = np.zeros((2,), dtype=np.float32)
        vs = np.zeros((16,), dtype=np.float32)

    center[:] = 0.0
    for i in range(num_of_inter):
        center[0] += int_pts[2 * i]
        center[1] += int_pts[2 * i + 1]
    center[0] /= num_of_inter
    center[1] /= num_of_inter

    for i in range(num_of_inter):
        vs[i] = _vertex_sort_key_numba(
            int_pts[2 * i], int_pts[2 * i + 1], center[0], center[1]
        )
    _insertion_sort_polygon_vertices_numba(int_pts, vs, num_of_inter)


@numba_jit(nopython=True, inline=True)
def _load_polygon_point(pts, idx, out):
    out[0] = pts[2 * idx]
    out[1] = pts[2 * idx + 1]


@numba_jit(
    "(float32[:], float32[:], int32, int32, float32[:])",
    device=numba_cuda_available,
    inline=True,
)
def line_segment_intersection_numba(pts1, pts2, i, j, temp_pts):
    if numba_cuda_available:
        A = cuda.local.array((2,), dtype=numba.float32)
        B = cuda.local.array((2,), dtype=numba.float32)
        C = cuda.local.array((2,), dtype=numba.float32)
        D = cuda.local.array((2,), dtype=numba.float32)
    else:
        A = np.zeros((2,), dtype=np.float32)
        B = np.zeros((2,), dtype=np.float32)
        C = np.zeros((2,), dtype=np.float32)
        D = np.zeros((2,), dtype=np.float32)

    _load_polygon_point(pts1, i, A)
    _load_polygon_point(pts1, (i + 1) % 4, B)
    _load_polygon_point(pts2, j, C)
    _load_polygon_point(pts2, (j + 1) % 4, D)
    BA0 = B[0] - A[0]
    BA1 = B[1] - A[1]
    DA0 = D[0] - A[0]
    CA0 = C[0] - A[0]
    DA1 = D[1] - A[1]
    CA1 = C[1] - A[1]
    acd = DA1 * CA0 > CA1 * DA0
    bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
    if acd != bcd:
        abc = CA1 * BA0 > BA1 * CA0
        abd = DA1 * BA0 > BA1 * DA0
        if abc != abd:
            DC0 = D[0] - C[0]
            DC1 = D[1] - C[1]
            ABBA = A[0] * B[1] - B[0] * A[1]
            CDDC = C[0] * D[1] - D[0] * C[1]
            DH = BA1 * DC0 - BA0 * DC1
            Dx = ABBA * DC0 - BA0 * CDDC
            Dy = ABBA * DC1 - BA1 * CDDC
            temp_pts[0] = Dx / DH
            temp_pts[1] = Dy / DH
            return True
    return False


@numba_jit(
    "(float32, float32, float32[:])", device=numba_cuda_available, inline=True
)
def point_in_quadrilateral_numba(pt_x, pt_y, corners):
    ab0 = corners[2] - corners[0]
    ab1 = corners[3] - corners[1]

    ad0 = corners[6] - corners[0]
    ad1 = corners[7] - corners[1]

    ap0 = pt_x - corners[0]
    ap1 = pt_y - corners[1]

    abab = ab0 * ab0 + ab1 * ab1
    abap = ab0 * ap0 + ab1 * ap1
    adad = ad0 * ad0 + ad1 * ad1
    adap = ad0 * ap0 + ad1 * ap1

    return abab >= abap and abap >= 0 and adad >= adap and adap >= 0


@numba_jit(nopython=True, inline=True)
def _append_intersection_point(int_pts, num_of_inter, x, y):
    int_pts[num_of_inter * 2] = x
    int_pts[num_of_inter * 2 + 1] = y
    return num_of_inter + 1


@numba_jit(
    "(float32[:], float32[:], float32[:])",
    device=numba_cuda_available,
    inline=True,
)
def quadrilateral_intersection_numba(pts1, pts2, int_pts):
    num_of_inter = 0
    for i in range(4):
        if point_in_quadrilateral_numba(pts1[2 * i], pts1[2 * i + 1], pts2):
            num_of_inter = _append_intersection_point(
                int_pts, num_of_inter, pts1[2 * i], pts1[2 * i + 1]
            )
        if point_in_quadrilateral_numba(pts2[2 * i], pts2[2 * i + 1], pts1):
            num_of_inter = _append_intersection_point(
                int_pts, num_of_inter, pts2[2 * i], pts2[2 * i + 1]
            )
    if numba_cuda_available:
        temp_pts = cuda.local.array((2,), dtype=numba.float32)
    else:
        temp_pts = np.zeros((2,), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            has_pts = line_segment_intersection_numba(
                pts1, pts2, i, j, temp_pts
            )
            if has_pts:
                num_of_inter = _append_intersection_point(
                    int_pts, num_of_inter, temp_pts[0], temp_pts[1]
                )

    return num_of_inter


@numba_jit(
    "(float32[:], float32[:])", device=numba_cuda_available, inline=True
)
def rbbox_to_corners_numba(corners, rbbox):
    # generate clockwise corners and rotate it clockwise
    angle = rbbox[4]
    a_cos = math.cos(angle)
    a_sin = math.sin(angle)
    center_x = rbbox[0]
    center_y = rbbox[1]
    hx = rbbox[2] * 0.5
    hy = rbbox[3] * 0.5

    for i in range(4):
        lx = -hx if i < 2 else hx
        ly = -hy if i == 0 or i == 3 else hy
        corners[2 * i] = a_cos * lx + a_sin * ly + center_x
        corners[2 * i + 1] = -a_sin * lx + a_cos * ly + center_y


@numba_jit(
    "(float32[:], float32[:])", device=numba_cuda_available, inline=True
)
def inter(rbbox1, rbbox2):
    if numba_cuda_available:
        corners1 = cuda.local.array((8,), dtype=numba.float32)
        corners2 = cuda.local.array((8,), dtype=numba.float32)
        intersection_corners = cuda.local.array((16,), dtype=numba.float32)
    else:
        corners1 = np.zeros((8,), dtype=np.float32)
        corners2 = np.zeros((8,), dtype=np.float32)
        intersection_corners = np.zeros((16,), dtype=np.float32)

    rbbox_to_corners_numba(corners1, rbbox1)
    rbbox_to_corners_numba(corners2, rbbox2)

    num_intersection = quadrilateral_intersection_numba(
        corners1, corners2, intersection_corners
    )
    sort_vertex_in_convex_polygon_numba(intersection_corners, num_intersection)

    return area(intersection_corners, num_intersection)


@numba_jit(nopython=True, inline=True)
def _iou_from_intersection(area_inter, area1, area2, criterion):
    if criterion == -1:
        return area_inter / max(area1 + area2 - area_inter, EPSILON)
    if criterion == 0:
        return area_inter / area1
    if criterion == 1:
        return area_inter / area2
    return area_inter


@numba_jit(
    "(float32[:], float32[:], int32)", device=numba_cuda_available, inline=True
)
def dev_rotate_iou_eval(rbox1, rbox2, criterion=-1):
    area_inter = inter(rbox1, rbox2)
    return _iou_from_intersection(
        area_inter, rbox1[2] * rbox1[3], rbox2[2] * rbox2[3], criterion
    )


def rotate_iou_cpu(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    criterion: Optional[int] = -1,
) -> np.ndarray:
    """Rotated box iou running in numba cpu.

    Args:
        pred_boxes: Predict rotated 2d Boxes,
            shape:[num_pred, 5], num_pred is the number of boxes.
        gt_boxes: GT rotated 2d boxes. shape:[num_gt, 5],
            num_gt is the number of boxes.
        criterion: Indicate different type of iou.
            -1 indicate `area_inter/(area_pred_box + area_gt_box - area_inter)`
            0 indicate `area_inter/area_pred_box`
            1 indicate `area_inter/area_gt_box`

    Returns:
        ious between pred and gt boxes.
    """

    num_pred = pred_boxes.shape[0]
    num_gt = gt_boxes.shape[0]
    rotate_ious = np.zeros((num_pred, num_gt), dtype=np.float32)
    for idx_pred in range(num_pred):
        for idx_gt in range(num_gt):
            iou = dev_rotate_iou_eval(
                pred_boxes[idx_pred], gt_boxes[idx_gt], criterion=criterion
            )
            rotate_ious[idx_pred][idx_gt] = iou

    return rotate_ious


@numba_jit(
    "(int64, int64, float32[:], float32[:], float32[:], int32)", fastmath=False
)  # noqa E501
def rotate_iou_kernel_eval(
    N,
    K,
    dev_boxes,
    dev_query_boxes,
    dev_iou,
    criterion=-1,
):
    row_start = cuda.blockIdx.x
    col_start = cuda.blockIdx.y
    tx = cuda.threadIdx.x
    row_size = min(N - row_start * THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    col_size = min(K - col_start * THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    block_boxes = cuda.shared.array(shape=(64 * 5,), dtype=numba.float32)
    block_qboxes = cuda.shared.array(shape=(64 * 5,), dtype=numba.float32)

    dev_query_box_idx = THREADS_PER_BLOCK * col_start + tx
    dev_box_idx = THREADS_PER_BLOCK * row_start + tx
    if tx < col_size:
        src = dev_query_box_idx * 5
        dst = tx * 5
        for k in range(5):
            block_qboxes[dst + k] = dev_query_boxes[src + k]
    if tx < row_size:
        src = dev_box_idx * 5
        dst = tx * 5
        for k in range(5):
            block_boxes[dst + k] = dev_boxes[src + k]
    cuda.syncthreads()
    if tx < row_size:
        for i in range(col_size):
            offset = (
                row_start * THREADS_PER_BLOCK * K
                + col_start * THREADS_PER_BLOCK
                + tx * K
                + i
            )
            dev_iou[offset] = dev_rotate_iou_eval(
                block_boxes[tx * 5 : tx * 5 + 5],  # pred_box
                block_qboxes[i * 5 : i * 5 + 5],  # gt_box
                criterion,
            )


def rotate_iou_gpu(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    criterion: Optional[int] = -1,
    device_id: Optional[int] = 0,
) -> np.ndarray:
    """Rotated box iou running in numba gpu.

    Args:
        pred_boxes: Predict rotated 2d Boxes,
            shape:[num_pred, 5], num_pred is the number of boxes.
        gt_boxes: GT rotated 2d boxes. shape:[num_gt, 5],
            num_gt is the number of boxes.
        criterion: Indicate different type of iou.
            -1 indicate `area_inter/(area_pred_box + area_gt_box - area_inter)`
            0 indicate `area_inter/area_pred_box`
            1 indicate `area_inter/area_gt_box`
        device_id: GPU device id.

    Returns:
        ious between pred and gt boxes.
    """
    # box_dtype = boxes.dtype
    pred_boxes = pred_boxes.astype(np.float32)
    gt_boxes = gt_boxes.astype(np.float32)
    N = pred_boxes.shape[0]
    K = gt_boxes.shape[0]
    iou = np.zeros((N, K), dtype=np.float32)
    if N == 0 or K == 0:
        return iou
    cuda.select_device(device_id)
    blockspergrid = (
        div_up(N, THREADS_PER_BLOCK),
        div_up(K, THREADS_PER_BLOCK),
    )

    stream = cuda.stream()
    with stream.auto_synchronize():
        boxes_dev = cuda.to_device(pred_boxes.reshape(-1), stream)
        query_boxes_dev = cuda.to_device(gt_boxes.reshape(-1), stream)
        iou_dev = cuda.to_device(iou.reshape(-1), stream)
        rotate_iou_kernel_eval[blockspergrid, THREADS_PER_BLOCK, stream](
            N, K, boxes_dev, query_boxes_dev, iou_dev, criterion
        )
        iou_dev.copy_to_host(iou.reshape([-1]), stream=stream)
    return iou.astype(pred_boxes.dtype)


def rotate_iou_v2(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    criterion: Optional[int] = -1,
    device_id: Optional[int] = 0,
) -> np.ndarray:
    """Compute rotated ious between GT and pred bboxes by numba.

    The iou func is mainly modify by the mmdetection3d:
    https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/core/evaluation/kitti_utils/rotate_iou.py, # noqa

    Note:
        When `criterion` in `[0,1]`, there are some differences in the
        calculation compared to mmdetection3d. In mmdetection3d,
        `criterion=0` indicate `area_inter/area_gt_box`,
        `criterion=1` indicate `area_inter/area_pred_box`.

    Args:
        pred_boxes: Predict rotated 2d Boxes,
            shape:[num_pred, 5], num_pred is the number of boxes.
        gt_boxes: GT rotated 2d boxes. shape:[num_gt, 5],
            num_gt is the number of boxes.
        criterion: Indicate different type of iou.
            -1 indicate `area_inter/(area_pred_box + area_gt_box - area_inter)`
            0 indicate `area_inter/area_pred_box`
            1 indicate `area_inter/area_gt_box`
        device_id: GPU device id, used in `rotate_iou_gpu`

    Returns:
        ious between pred and gt boxes.
    """
    if numba_cuda_available:
        iou = rotate_iou_gpu(
            pred_boxes,
            gt_boxes,
            criterion=criterion,
            device_id=device_id,
        )
    else:
        iou = rotate_iou_cpu(
            pred_boxes,
            gt_boxes,
            criterion=criterion,
        )

    return iou
