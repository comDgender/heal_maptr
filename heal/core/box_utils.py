# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

# Implements some commonly used bounding box utilities. Applies to both 2d box
# and 3d box.
from typing import Optional, Tuple, Union

import numpy as np
import torch

from heal.core.point_geometry import coor_transformation

__all__ = [
    "box_center_to_corner",
    "box_corner_to_center",
    "bbox_overlaps",
    "fp16_clamp",
    "is_center_of_bboxes_in_roi",
    "get_bev_bbox",
    "zoom_boxes",
    "recover_ori_bbox",
]


def box_center_to_corner(
    bboxes: torch.Tensor,
    split: Optional[bool] = False,
    legacy_bbox: Optional[bool] = False,
):  # noqa: D205,D400
    """
    Convert bounding box from center format (xcenter, ycenter,
    width, height) to corner format (x_low, y_low, x_high, y_high)

    Args:
        bboxes: Shape is (..., 4) represents bounding boxes.
        split: Whether to split the final output to
            for (..., 1) tensors, or keep the (..., 4) original output.
            Default to False.
        legacy_bbox: Whether the boxes are decoded
            in legacy manner (should add one to bottom or right coordinate
            before using) or not. Default to False.
    """

    border = int(legacy_bbox)
    cx, cy, w, h = torch.split(bboxes, 1, dim=-1)
    x1 = cx - (w - border) * 0.5
    y1 = cy - (h - border) * 0.5
    x2 = x1 + w - border
    y2 = y1 + h - border

    if split:
        return x1, y1, x2, y2
    else:
        return torch.cat([x1, y1, x2, y2], dim=-1)


def box_corner_to_center(
    bboxes: torch.Tensor,
    split: Optional[bool] = False,
    legacy_bbox: Optional[bool] = False,
):  # noqa: D205,D400
    """
    Convert bounding box from corner format (x_low, y_low, x_high, y_high)
    to center format (xcenter, ycenter, width, height)

    Args:
        bboxes: Shape is (..., 4) represents bounding boxes.
        split: Whether to split the final output to
            for (..., 1) tensors, or keep the (..., 4) original output.
            Default to False.
        legacy_bbox: Whether the boxes are decoded
            in legacy manner (should add one to bottom or right coordinate
            before using) or not. Default to False.
    """

    border = int(legacy_bbox)
    x1, y1, x2, y2 = torch.split(bboxes, 1, dim=-1)
    width = x2 - x1 + border
    height = y2 - y1 + border
    cx = x1 + (width - border) * 0.5
    cy = y1 + (height - border) * 0.5

    if split:
        return cx, cy, width, height
    else:
        return torch.cat([cx, cy, width, height], dim=-1)


def fp16_clamp(
    x: torch.Tensor,
    min: Optional[float] = None,
    max: Optional[float] = None,
) -> torch.Tensor:
    """Clamp tensor values with fp16-safe handling on CPU."""
    if not x.is_cuda and x.dtype == torch.float16:
        return x.float().clamp(min, max).half()
    return x.clamp(min, max)


def _xyxy_areas(boxes: torch.Tensor) -> torch.Tensor:
    """Area of axis-aligned boxes in ``(x1, y1, x2, y2)`` format."""
    widths = boxes[..., 2] - boxes[..., 0]
    heights = boxes[..., 3] - boxes[..., 1]
    return widths * heights


def _pairwise_overlap_and_hull(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
    aligned: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Intersection area plus the enclosing-box corners for two box sets."""
    a_min, a_max = boxes_a[..., :2], boxes_a[..., 2:]
    b_min, b_max = boxes_b[..., :2], boxes_b[..., 2:]

    if aligned:
        inter_min = torch.maximum(a_min, b_min)
        inter_max = torch.minimum(a_max, b_max)
        hull_min = torch.minimum(a_min, b_min)
        hull_max = torch.maximum(a_max, b_max)
    else:
        # Broadcast to (..., rows, cols, 2).
        a_min_exp = a_min[..., :, None, :]
        b_min_exp = b_min[..., None, :, :]
        a_max_exp = a_max[..., :, None, :]
        b_max_exp = b_max[..., None, :, :]
        inter_min = torch.maximum(a_min_exp, b_min_exp)
        inter_max = torch.minimum(a_max_exp, b_max_exp)
        hull_min = torch.minimum(a_min_exp, b_min_exp)
        hull_max = torch.maximum(a_max_exp, b_max_exp)

    inter_wh = fp16_clamp(inter_max - inter_min, min=0)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]
    return inter_area, hull_min, hull_max


def bbox_overlaps(
    bboxes1: Union[torch.Tensor, np.ndarray],
    bboxes2: Union[torch.Tensor, np.ndarray],
    mode: Optional[str] = "iou",
    is_aligned: Optional[bool] = False,
    eps: Optional[float] = 1e-6,
):
    """
    Calculate overlap between two set of bboxes.

    Args:
        bboxes1:
            shape (m, 4) or (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2:
            shape (n, 4)  or (B, n, 4) in <x1, y1, x2, y2> format or empty.
        mode: "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned: If True, then m and n must be equal.
            Default False.
        eps: A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        tensor of shape (m, n) or (B, m, n)
        if ``is_aligned `` is False else shape (m,) or (B, m,)
    """

    assert isinstance(bboxes1, type(bboxes2))

    if isinstance(bboxes1, (list, tuple)):
        if not len(bboxes1) or not len(bboxes2):
            return np.array([]).reshape(len(bboxes1), len(bboxes2))
        bboxes1 = np.asarray(bboxes1)
        bboxes2 = np.asarray(bboxes2)

    is_ndarray = False
    if isinstance(bboxes1, np.ndarray):
        is_ndarray = True
        bboxes1 = torch.from_numpy(bboxes1)
        bboxes2 = torch.from_numpy(bboxes2)

    assert mode in ["iou", "iof", "giou"], f"Unsupported mode {mode}"
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert bboxes1.size(-1) == 4 or bboxes1.size(0) == 0
    assert bboxes2.size(-1) == 4 or bboxes2.size(0) == 0

    # Leading batch dims (everything except box-count and xyxy) must match.
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_dims = bboxes1.shape[:-2]
    n_a = bboxes1.size(-2)
    n_b = bboxes2.size(-2)

    if is_aligned:
        assert n_a == n_b

    if n_a == 0 or n_b == 0:
        out_shape = batch_dims + ((n_a,) if is_aligned else (n_a, n_b))
        empty = bboxes1.new_zeros(out_shape)
        return empty.numpy() if is_ndarray else empty

    area_a = _xyxy_areas(bboxes1)
    area_b = _xyxy_areas(bboxes2)
    overlap, hull_min, hull_max = _pairwise_overlap_and_hull(
        bboxes1, bboxes2, bool(is_aligned)
    )

    if mode in ("iou", "giou"):
        if is_aligned:
            denom = area_a + area_b - overlap
        else:
            denom = area_a[..., None] + area_b[..., None, :] - overlap
    else:
        # iof: intersect over foreground (boxes_a).
        denom = area_a if is_aligned else area_a[..., None]

    eps_t = denom.new_tensor([eps])
    denom = torch.maximum(denom, eps_t)
    ious = overlap / denom
    if mode in ("iou", "iof"):
        return ious.numpy() if is_ndarray else ious

    # GIoU: subtract the normalized empty area inside the enclosing box.
    hull_wh = fp16_clamp(hull_max - hull_min, min=0)
    hull_area = hull_wh[..., 0] * hull_wh[..., 1]
    hull_area = torch.maximum(hull_area, eps_t)
    gious = ious - (hull_area - denom) / hull_area
    return gious.numpy() if is_ndarray else gious


def is_center_of_bboxes_in_roi(boxes, roi):
    """
    Check center of bboxes is in roi or not.

    Args:
        boxes(ndarray): shape (n, 4)
        roi(ndarray): shape (4)
    Return:
        mask(bool): if center of bboxes in roi.
    """

    center = (boxes[:, :2] + boxes[:, 2:]) / 2
    mask = (
        (center[:, 0] > roi[0])
        * (center[:, 1] > roi[1])
        * (center[:, 0] < roi[2])
        * (center[:, 1] < roi[3])
    )
    return mask


# =============================================================================
# The following methods are mostly used in lidar 3d box processing.
# =============================================================================

_CORNERS_ND_REORDER = {
    2: [0, 1, 3, 2],
    3: [0, 1, 3, 2, 4, 5, 7, 6],
}


def _unit_hypercube_vertices(ndim: int, dtype) -> np.ndarray:
    vertex_count = 1 << ndim
    lattice = np.empty((vertex_count, ndim), dtype=dtype)
    for vtx_id in range(vertex_count):
        for axis_id in range(ndim):
            lattice[vtx_id, axis_id] = (vtx_id >> axis_id) & 1
    return lattice


def _reorder_corner_vertices(vertices: np.ndarray, ndim: int) -> np.ndarray:
    perm = _CORNERS_ND_REORDER.get(ndim)
    if perm is None:
        return vertices
    return vertices[perm]


def _anchor_corner_template(
    vertices: np.ndarray,
    anchor: Union[Tuple[float, ...], float],
    dtype,
) -> np.ndarray:
    anchor_vec = np.asarray(anchor, dtype=dtype)
    return vertices - anchor_vec


def corners_nd(
    dims: np.ndarray, origin: Union[Tuple[float, ...], float] = 0.5
) -> torch.Tensor:
    """Build unit-box corner offsets scaled by per-box dimensions.

    Args:
        dims: Per-box sizes with shape (N, ndim).
        origin: Anchor within the unit hypercube relative to the minimum
            vertex.

    Returns:
        Corner coordinates with shape (N, 2**ndim, ndim).
    """
    ndim = int(dims.shape[1])
    vertex_count = 1 << ndim
    unit_vertices = _unit_hypercube_vertices(ndim, dims.dtype)
    ordered_vertices = _reorder_corner_vertices(unit_vertices, ndim)
    template = _anchor_corner_template(ordered_vertices, origin, dims.dtype)
    per_box = dims.reshape(-1, 1, ndim)
    corner_layout = template.reshape(1, vertex_count, ndim)
    return per_box * corner_layout


def rotation_2d(points: np.ndarray, angles: np.ndarray) -> np.ndarray:
    """Rotate 2d points based on origin point clockwise when angle positive.

    Args:
        points: points to be rotated, shape=[N, point_size, 2]
        angles (float array, ): rotation angle, shape=[N]

    Returns:
        same shape as points
    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    rot_mat_T = np.stack([[rot_cos, -rot_sin], [rot_sin, rot_cos]])
    return np.einsum("aij,jka->aik", points, rot_mat_T)


def center_to_corner_box2d(
    centers: Union[np.ndarray, torch.Tensor],
    dims: Union[np.ndarray, torch.Tensor],
    angles: Optional[Union[np.ndarray, torch.Tensor]] = None,
    origin: float = 0.5,
) -> np.ndarray:
    """Convert Kitti-style locations, dimensions and angles to corners.

    format: center(xy), dims(xy), angles(clockwise when positive)

    Args:
        centers: locations in kitti label file, shape=[N, 2].
        dims: dimensions in kitti label file, shape=[N, 2].
        angles: rotation_y in kitti label file, shape=[N].
        origin: origin point relative to the smallest point. Defaults to 0.5.

    Returns:
        corner representation of boxes.
    """
    if isinstance(centers, torch.Tensor):
        is_tensor = True
        device = centers.device
        centers = centers.cpu().numpy()
        dims = dims.cpu().numpy()
        angles = angles.cpu().numpy()
    else:
        is_tensor = False

    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d(corners, angles)
    corners += centers.reshape([-1, 1, 2])
    return torch.from_numpy(corners).to(device) if is_tensor else corners


def zoom_boxes(
    boxes: torch.Tensor, roi_wh_zoom_scale: Tuple[float, float]
) -> torch.Tensor:
    """Zoom boxes.

    Args:
        boxes: shape (m, 4) in <x1, y1, x2, y2> format.
        roi_wh_zoom_scale: (w_scale, h_scale).

    Returns:
        zoomed bboxes.
    """
    boxes = boxes[..., :4]
    boxes_w = boxes[..., 2] - boxes[..., 0]
    boxes_h = boxes[..., 3] - boxes[..., 1]

    w_bias = 0.5 * (roi_wh_zoom_scale[0] - 1) * boxes_w
    h_bias = 0.5 * (roi_wh_zoom_scale[1] - 1) * boxes_h

    return torch.stack(
        [
            boxes[..., 0] - w_bias,
            boxes[..., 1] - h_bias,
            boxes[..., 2] + w_bias,
            boxes[..., 3] + h_bias,
        ],
        dim=-1,
    )


def minmax_to_corner_2d(minmax_box: np.ndarray) -> np.ndarray:
    """Convert min-max representation of a box into corner representation.

    Args:
        minmax_box: [N, 2*ndim] box. ndim indicates whether it is
            a 2-d box or a 3-d box.

    Returns:
        corner representation of a boxes.
    """
    ndim = minmax_box.shape[-1] // 2
    center = minmax_box[..., :ndim]
    dims = minmax_box[..., ndim:] - center
    return center_to_corner_box2d(center, dims, origin=0.0)


def get_bev_bbox(coordinate, size, yaw):
    size = np.clip(size, a_min=1, a_max=None)
    if len(coordinate) == 0:
        return np.zeros([0, 4, 2])

    corners = size / 2
    corners = np.stack(
        [
            corners,
            corners * np.array([1, -1]),
            corners * np.array([-1, -1]),
            corners * np.array([-1, 1]),
        ],
        axis=-2,
    )
    bev_bbox = coor_transformation(corners, yaw[:, None], coordinate[:, None])

    return bev_bbox


def center_to_minmax_2d(
    centers: torch.Tensor, dims: torch.Tensor
) -> torch.Tensor:
    """Convert center box (x_center, y_center) to left top and right bottom box.

    Args:
        centers: Coordinate (x_center, y_center) format.
        dims: Width and height of center box.

    Returns:
         Min (left top) and Max (right bottom) box.
    """
    return torch.cat([centers - dims / 2, centers + dims / 2], dim=-1)


def recover_ori_bbox(
    bboxes: torch.Tensor,
    scale_factors: torch.Tensor,
    resized_shape: torch.Tensor,
):
    """Translate predicted normalized bbox to bbox in original image.

    Scale bbox according to input image size and then transform box
    according to scale_factors (image resize and padding).

    Args:
        bboxes: The normalized bbox output from model.
        scale_factors: scale factors from input data. if the shape is
            (1x4), it is [scale_w, scale_h, scale_w, scale_h];
            if the shape is (3x3), it is a transform matrix.
        resized_shape: The model input image shape
    """
    pred_boxes = []
    for bbox, scale_factor, img_size in zip(
        bboxes, scale_factors, resized_shape
    ):
        H, W, _ = img_size
        image_size_xyxy = torch.as_tensor(
            [W, H, W, H], dtype=torch.float, device=bbox.device
        ).reshape([1, 4])
        pred_box = bbox * image_size_xyxy
        if scale_factor.shape[-1] == 4:  # shape [1, 4]
            pred_box = pred_box / scale_factor
        elif scale_factor.shape[-1] == 3:  # R shape [3, 3]
            R_inv = torch.inverse(scale_factor)
            coord = torch.stack([pred_box[:, :2], pred_box[:, 2:]])
            coord1 = torch.cat(
                [
                    coord,
                    torch.ones([2, coord.shape[1], 1], device=bbox.device),
                ],
                dim=2,
            )
            new_coord = torch.matmul(
                R_inv.float(), coord1.float().unsqueeze(3)
            ).squeeze()
            pred_box = torch.cat(
                [new_coord[0, :, :2], new_coord[1, :, :2]], dim=1
            )

        pred_boxes.append(pred_box)
    return pred_boxes
