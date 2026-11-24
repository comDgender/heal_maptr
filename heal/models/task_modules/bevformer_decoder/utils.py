# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

"""3D box encode/decode for BEVFormer detection head.

Training uses ``normalize_bbox`` (log size, sin/cos yaw); inference uses
``denormalize_bbox`` to recover metric boxes for NMS-free post-process.
"""

import torch


def normalize_bbox(bboxes: torch.Tensor) -> torch.Tensor:
    """Encode metric 3D box to network target: log(w,l,h), sin/cos(yaw), vx, vy."""  # noqa: E501
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    bl = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()
    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, w, bl, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1
        )
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, w, bl, cz, h, rot.sin(), rot.cos()), dim=-1
        )
    return normalized_bboxes


def denormalize_bbox(normalized_bboxes: torch.Tensor) -> torch.Tensor:
    """Decode network output to metric box: exp sizes, atan2 for yaw."""
    rot_sine = normalized_bboxes[..., 6:7]
    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    w = normalized_bboxes[..., 2:3]
    bl = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]

    w = w.exp()
    bl = bl.exp()
    h = h.exp()
    if normalized_bboxes.size(-1) > 8:
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat(
            [cx, cy, cz, w, bl, h, rot, vx, vy], dim=-1
        )
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, bl, h, rot], dim=-1)
    return denormalized_bboxes


def normalize_bbox_2d(bboxes: torch.Tensor) -> torch.Tensor:
    """Normalize bbox."""
    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    w = bboxes[..., 2:3].log()
    bl = bboxes[..., 3:4].log()
    if bboxes.size(-1) == 5:
        rot = bboxes[..., 4:5]
        normalized_bboxes = torch.cat(
            (cx, cy, w, bl, rot.sin(), rot.cos()), dim=-1
        )
    elif bboxes.size(-1) == 4:
        normalized_bboxes = torch.cat((cx, cy, w, bl), dim=-1)
    else:
        rot = bboxes[..., 4:5]
        vx = bboxes[..., 5:6]
        vy = bboxes[..., 6:7]
        normalized_bboxes = torch.cat(
            (cx, cy, w, bl, rot.sin(), rot.cos(), vx, vy), dim=-1
        )
    return normalized_bboxes
