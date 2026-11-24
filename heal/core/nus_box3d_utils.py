# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import copy
import logging
from typing import List, Tuple

import numpy as np
import torch
from hat_engine.utils.package_helper import require_packages

logger = logging.getLogger(__name__)

try:
    from nuscenes.eval.common.utils import Quaternion
except ImportError:
    Quaternion = None


def get_min_max_coords(
    bev_size: Tuple[float],
) -> Tuple[float, float, float, float]:
    min_x = -bev_size[1] + bev_size[2] / 2
    max_x = bev_size[1] - bev_size[2] / 2
    min_y = -bev_size[0] + bev_size[2] / 2
    max_y = bev_size[0] - bev_size[2] / 2
    return min_x, max_x, min_y, max_y


def adjust_coords(coords: torch.Tensor, grid_size: Tuple[int]) -> torch.Tensor:
    w = grid_size[0]
    h = grid_size[1]

    bev_x = (torch.linspace(0, w - 1, w).reshape((1, w)).repeat(h, 1)).float()
    bev_y = (torch.linspace(0, h - 1, h).reshape((h, 1)).repeat(1, w)).float()
    bev_coords = torch.stack([bev_x, bev_y], axis=-1).to(device=coords.device)
    return coords - bev_coords


@require_packages(
    "nuscenes", raise_msg="Please `pip3 install nuscenes-devkit`"
)
def bbox_to_corner(
    bboxes: torch.Tensor, score_thresh: float = 0.0
) -> Tuple[List[np.array], List[float], List[float]]:
    decoded_bbox = []
    scores = []
    cat_ids = []
    for bbox in bboxes:
        if isinstance(bbox, torch.Tensor):
            bbox = bbox.cpu().numpy()

        score = bbox[9]
        if score < score_thresh:
            continue
        w, l, h = bbox[3:6]
        x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))
        yaw = bbox[6]
        rot = Quaternion(axis=[0, 0, 1], radians=yaw)
        corners = np.dot(rot.rotation_matrix, corners)
        x, y, z = bbox[:3]
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z
        corners = np.transpose(corners, (1, 0))
        decoded_bbox.append(corners)
        if len(bbox) == 11:
            scores.append(bbox[9])
            cat_ids.append(bbox[10])
        else:
            scores.append(1.0)
            cat_ids.append(bbox[9])

    return decoded_bbox, scores, cat_ids


def bbox_bev2ego(bboxes: torch.Tensor, bev_size: Tuple[float]) -> torch.Tensor:
    ego_bboxes = []
    min_x, max_x, min_y, max_y = get_min_max_coords(bev_size)
    for bbox in bboxes:
        bbox = copy.deepcopy(bbox)
        bbox[:6] *= bev_size[2]
        bbox[0] -= max_x
        bbox[1] -= max_y
        bbox[7:9] *= bev_size[2]
        ego_bboxes.append(bbox)
    return ego_bboxes


def bbox_ego2bev(bboxes: torch.Tensor, bev_size: Tuple[float]) -> torch.Tensor:
    bev_bboxes = []
    min_x, max_x, min_y, max_y = get_min_max_coords(bev_size)
    for bbox in bboxes:
        bbox = copy.deepcopy(bbox)
        bbox[0] += max_x
        bbox[1] += max_y
        bbox[:6] /= bev_size[2]
        bbox[7:9] /= bev_size[2]
        bev_bboxes.append(bbox)
    return bev_bboxes


def bbox_ego2img(
    bboxes: torch.Tensor,
    ego2img: np.array,
    img_size: Tuple[int],
    score_thresh: float = 0.0,
):
    def _is_visible(corners_2d, corners_3d, imsize):
        visible = np.logical_and(
            corners_2d[:, 0] > 0, corners_2d[:, 0] < imsize[1]
        )
        visible = np.logical_and(visible, corners_2d[:, 1] < imsize[0])
        visible = np.logical_and(visible, corners_2d[:, 1] > 0)
        visible = np.logical_and(visible, corners_3d[:, 2] > 1)
        return any(visible)

    decoded_corners_3d, _, _ = bbox_to_corner(bboxes, score_thresh)
    img_bboxes = []
    for corners_3d in decoded_corners_3d:
        nbr_points = corners_3d.shape[0]
        corners_3d = np.transpose(corners_3d, (1, 0))
        corners_3d = np.concatenate((corners_3d, np.ones((1, nbr_points))))
        corners_3d = np.dot(ego2img, corners_3d)
        corners_2d = corners_3d[:3, :]
        corners_2d = corners_2d / corners_2d[2:3, :].repeat(3, 0).reshape(
            3, nbr_points
        )
        corners_3d = np.transpose(corners_3d, (1, 0))
        corners_2d = np.transpose(corners_2d[:2, :], (1, 0))
        if not _is_visible(corners_2d, corners_3d, (img_size)):
            continue
        corners_2d[..., 0] = np.clip(corners_2d[..., 0], 0, img_size[1] - 1)
        corners_2d[..., 1] = np.clip(corners_2d[..., 1], 0, img_size[0] - 1)
        img_bboxes.append(corners_2d)
    return img_bboxes
