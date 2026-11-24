# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import math

import torch


def _torch_string_type_to_class(ttype):
    type_map = {
        "torch.HalfTensor": torch.HalfTensor,
        "torch.FloatTensor": torch.FloatTensor,
        "torch.DoubleTensor": torch.DoubleTensor,
        "torch.IntTensor": torch.IntTensor,
        "torch.LongTensor": torch.LongTensor,
        "torch.ByteTensor": torch.ByteTensor,
        "torch.cuda.HalfTensor": torch.cuda.HalfTensor,
        "torch.cuda.FloatTensor": torch.cuda.FloatTensor,
        "torch.cuda.DoubleTensor": torch.cuda.DoubleTensor,
        "torch.cuda.IntTensor": torch.cuda.IntTensor,
        "torch.cuda.LongTensor": torch.cuda.LongTensor,
        "torch.cuda.ByteTensor": torch.cuda.ByteTensor,
    }
    return type_map[ttype]


def get_tensor_class(tensor):
    return _torch_string_type_to_class(tensor.type())


def rotation_points_single_angle(points, angle, axis=0):
    # points: [N, 3]
    rot_sin = math.sin(angle)
    rot_cos = math.cos(angle)
    point_type = get_tensor_class(points)
    if axis == 1:
        rot_mat_T = torch.stack(
            [
                point_type([rot_cos, 0, -rot_sin]),
                point_type([0, 1, 0]),
                point_type([rot_sin, 0, rot_cos]),
            ]
        )
    elif axis == 2 or axis == -1:
        rot_mat_T = torch.stack(
            [
                point_type([rot_cos, -rot_sin, 0]),
                point_type([rot_sin, rot_cos, 0]),
                point_type([0, 0, 1]),
            ]
        )
    elif axis == 0:
        rot_mat_T = torch.stack(
            [
                point_type([1, 0, 0]),
                point_type([0, rot_cos, -rot_sin]),
                point_type([0, rot_sin, rot_cos]),
            ]
        )
    else:
        raise ValueError("axis should in range")
    return points @ rot_mat_T


def corner_to_standup_nd(boxes_corner):
    ndim = boxes_corner.shape[2]
    standup_boxes = [
        torch.min(boxes_corner[:, :, i], dim=1)[0] for i in range(ndim)
    ]
    standup_boxes.extend(
        torch.max(boxes_corner[:, :, i], dim=1)[0] for i in range(ndim)
    )
    return torch.stack(standup_boxes, dim=1)
