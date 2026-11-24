# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import Optional, Union

import numpy as np
import torch

from .rotate_box_utils import rotate_iou


def bev3d_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    thresh: float,
    cls_ids: Optional[torch.Tensor] = None,
    agnostic: bool = True,
) -> np.ndarray:
    """NMS for BEV 3D boxes on the ground plane."""
    assert len(boxes) == len(scores), "len(scores) must be equal to len(boxes)"
    if not agnostic:
        assert cls_ids is not None and len(cls_ids) == len(scores)
        cls_ids = cls_ids.cpu().numpy()

    boxes = boxes.cpu().numpy()
    if boxes.shape[1] == 7:
        loc_xy = boxes[:, :2]
        dim_lw = boxes[:, 5:3:-1]
        yaw = boxes[:, -1:]
        det_bbox3d = np.concatenate([loc_xy, dim_lw, -yaw], axis=1)
    else:
        assert boxes.shape[1] == 5
        det_bbox3d = boxes

    scores = scores.cpu().numpy()
    order = np.argsort(scores)
    keep = np.zeros(det_bbox3d.shape[0])

    while order.size > 0:
        index = order[-1]
        keep[index] = 1
        current_box = np.expand_dims(det_bbox3d[index], axis=0)
        left_boxes = det_bbox3d[order[:-1]]
        rotate_3d_ious = rotate_iou(current_box, left_boxes)
        rotate_3d_ious = np.squeeze(rotate_3d_ious)
        left = rotate_3d_ious < thresh
        if agnostic:
            left = np.where(left)
        else:
            current_id = cls_ids[index]
            left_ids = cls_ids[order[:-1]]
            left = np.where(np.logical_or(left, left_ids != current_id))
        order = order[left]
    return keep.astype("bool")


def points_cam2img(points_3d, proj_mat, with_depth=False):
    """Project camera-frame points to image coordinates."""
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1

    assert len(proj_mat.shape) == 2, (
        "The dimension of the projection"
        f" matrix should be 2 instead of {len(proj_mat.shape)}."
    )
    d1, d2 = proj_mat.shape[:2]
    assert (
        (d1 == 3 and d2 == 3) or (d1 == 3 and d2 == 4) or (d1 == 4 and d2 == 4)
    ), ("The shape of the projection matrix" f" ({d1}*{d2}) is not supported.")
    if d1 == 3:
        proj_mat_expanded = torch.eye(
            4, device=proj_mat.device, dtype=proj_mat.dtype
        )
        proj_mat_expanded[:d1, :d2] = proj_mat
        proj_mat = proj_mat_expanded

    points_4 = torch.cat([points_3d, points_3d.new_ones(points_shape)], dim=-1)
    point_2d = points_4 @ proj_mat.T
    depth = point_2d[..., 2:3]
    eps = torch.finfo(depth.dtype).eps
    depth_sign = torch.where(
        depth < 0, -torch.ones_like(depth), torch.ones_like(depth)
    )
    safe_depth = torch.where(depth.abs() < eps, depth_sign * eps, depth)
    point_2d_res = point_2d[..., :2] / safe_depth

    if with_depth:
        point_2d_res = torch.cat([point_2d_res, depth], dim=-1)

    return point_2d_res


def points_img2cam(points, cam2img):
    """Project image-frame points back to camera coordinates."""
    if not isinstance(cam2img, torch.Tensor):
        cam2img = torch.tensor(cam2img)
    assert cam2img.shape[0] <= 4
    assert cam2img.shape[1] <= 4
    assert points.shape[1] == 3

    xys = points[:, :2]
    depths = points[:, 2].view(-1, 1)
    unnormed_xys = torch.cat([xys * depths, depths], dim=1)

    pad_cam2img = torch.eye(4, dtype=xys.dtype, device=xys.device)
    pad_cam2img[: cam2img.shape[0], : cam2img.shape[1]] = cam2img
    inv_pad_cam2img = torch.inverse(pad_cam2img).transpose(0, 1)

    num_points = unnormed_xys.shape[0]
    homo_xys = torch.cat([unnormed_xys, xys.new_ones((num_points, 1))], dim=1)
    points3D = torch.mm(homo_xys, inv_pad_cam2img)[:, :3]
    return points3D


def xywhr2xyxyr(boxes_xywhr: torch.Tensor):
    """Convert rotated boxes from XYWHR format to XYXYR format."""
    boxes = torch.zeros_like(boxes_xywhr)
    half_w = boxes_xywhr[..., 2] / 2
    half_h = boxes_xywhr[..., 3] / 2

    boxes[..., 0] = boxes_xywhr[..., 0] - half_w
    boxes[..., 1] = boxes_xywhr[..., 1] - half_h
    boxes[..., 2] = boxes_xywhr[..., 0] + half_w
    boxes[..., 3] = boxes_xywhr[..., 1] + half_h
    boxes[..., 4] = boxes_xywhr[..., 4]
    return boxes


def xyxyr2xywhr(boxes_xyxyr: torch.Tensor) -> torch.Tensor:
    """Convert rotated boxes from XYXYR format to XYWHR format."""
    w = boxes_xyxyr[..., 2] - boxes_xyxyr[..., 0]
    h = boxes_xyxyr[..., 3] - boxes_xyxyr[..., 1]
    boxes = torch.zeros_like(boxes_xyxyr)
    boxes[..., 0] = boxes_xyxyr[..., 0] + w * 0.5
    boxes[..., 1] = boxes_xyxyr[..., 1] + h * 0.5
    boxes[..., 2] = w
    boxes[..., 3] = h
    boxes[..., 4] = boxes_xyxyr[..., 4]
    return boxes


def _stack_rotation_rows(*rows) -> torch.Tensor:
    return torch.stack([torch.stack(row) for row in rows])


def _normalize_rotation_axis(axis: int) -> int:
    if axis in (0, -3):
        return 0
    if axis in (1, -2):
        return 1
    if axis in (2, -1):
        return 2
    raise ValueError(f"axis should in range [-3, -2, -1, 0, 1, 2], got {axis}")


def _rotation_mat_transpose(
    rot_cos: torch.Tensor,
    rot_sin: torch.Tensor,
    ones: torch.Tensor,
    zeros: torch.Tensor,
    axis: int,
    point_dim: int,
) -> torch.Tensor:
    c, s, o, z = rot_cos, rot_sin, ones, zeros
    if point_dim == 2:
        return _stack_rotation_rows((c, s), (-s, c))

    axis_rows = (
        ((o, z, z), (z, c, s), (z, -s, c)),
        ((c, z, -s), (z, o, z), (s, z, c)),
        ((c, s, z), (-s, c, z), (z, z, o)),
    )
    return _stack_rotation_rows(*axis_rows[_normalize_rotation_axis(axis)])


def rotation_3d_in_axis(
    points: torch.Tensor,
    angles: torch.Tensor,
    axis: int = 0,
    return_mat: bool = False,
    clockwise: bool = False,
) -> Union[torch.Tensor, np.ndarray]:
    """Rotate points by angles according to axis."""
    batch_free = len(points.shape) == 2
    if batch_free:
        points = points[None]

    if isinstance(angles, float) or len(angles.shape) == 0:
        angles = torch.full(points.shape[:1], angles)

    assert (
        len(points.shape) == 3
        and len(angles.shape) == 1
        and points.shape[0] == angles.shape[0]
    ), (
        f"Incorrect shape of points " f"angles: {points.shape}, {angles.shape}"
    )

    point_dim = points.shape[-1]
    assert point_dim in (
        2,
        3,
    ), f"Points size should be 2 or 3 instead of {point_dim}"

    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)
    rot_mat_T = _rotation_mat_transpose(
        rot_cos, rot_sin, ones, zeros, axis, point_dim
    )

    if clockwise:
        rot_mat_T = rot_mat_T.transpose(0, 1)

    if points.shape[0] == 0:
        points_new = points
    else:
        points_new = torch.einsum("aij,jka->aik", points, rot_mat_T)

    if batch_free:
        points_new = points_new.squeeze(0)

    if return_mat:
        rot_mat_T = torch.einsum("jka->ajk", rot_mat_T)
        if batch_free:
            rot_mat_T = rot_mat_T.squeeze(0)
        return points_new, rot_mat_T
    return points_new


def yaw2local(yaw: torch.Tensor, loc: torch.Tensor):
    """Transform global yaw to local yaw in camera coordinates."""
    local_yaw = yaw - torch.atan2(loc[:, 0], loc[:, 2])
    larger_idx = (local_yaw > np.pi).nonzero(as_tuple=False)
    small_idx = (local_yaw < -np.pi).nonzero(as_tuple=False)
    if len(larger_idx) != 0:
        local_yaw[larger_idx] -= 2 * np.pi
    if len(small_idx) != 0:
        local_yaw[small_idx] += 2 * np.pi
    return local_yaw


def bbox3d2result(bboxes, scores, labels, attrs=None):
    result_dict = {
        "boxes_3d": bboxes.to("cpu"),
        "scores_3d": scores.cpu(),
        "labels_3d": labels.cpu(),
    }
    if attrs is not None:
        result_dict["attrs_3d"] = attrs.cpu()
    return result_dict
