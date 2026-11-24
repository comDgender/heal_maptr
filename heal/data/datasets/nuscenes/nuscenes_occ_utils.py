# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
"""Occ3D ego/lidar coordinate helpers (aligned with HAT NuscenesSample)."""

from typing import Dict

import numpy as np

__all__ = ["get_occ_ego2lidar", "get_lidar_occ_info"]


def get_occ_ego2lidar(
    voxel_semantics: np.ndarray,
    mask_lidar: np.ndarray,
    mask_camera: np.ndarray,
    transform_matrix: np.ndarray,
    pc_range=(-40, -40, -1, 40, 40, 5.4),
    lidar_pc_range=(-40.0, -40.0, -5.0, 40.0, 40.0, 3.0),
    ego_resolution=(0.4, 0.4, 0.4),
):
    """Remap occ3d voxels from ego frame to lidar frame."""
    h, w, z = voxel_semantics.shape
    occ_trans = np.full((200, 200, 16), 17, dtype=np.uint8)
    mask_lidar_trans = np.full((200, 200, 16), 0, dtype=np.uint8)
    mask_camera_trans = np.full((200, 200, 16), 0, dtype=np.uint8)

    x = np.linspace(0.5, w - 0.5, w)
    y = np.linspace(0.5, h - 0.5, h)
    z_coords = np.linspace(0.5, z - 0.5, z)
    xs, ys, zs = np.meshgrid(x / w, y / h, z_coords / z, indexing="ij")
    x_grid, y_grid, z_grid = np.meshgrid(x, y, z_coords, indexing="ij")
    lidar_coords = np.stack([xs, ys, zs, np.ones_like(xs)], axis=-1)
    lidar_coords[..., 0:1] = (
        lidar_coords[..., 0:1] * (lidar_pc_range[3] - lidar_pc_range[0])
        + lidar_pc_range[0]
    )
    lidar_coords[..., 1:2] = (
        lidar_coords[..., 1:2] * (lidar_pc_range[4] - lidar_pc_range[1])
        + lidar_pc_range[1]
    )
    lidar_coords[..., 2:3] = (
        lidar_coords[..., 2:3] * (lidar_pc_range[5] - lidar_pc_range[2])
        + lidar_pc_range[2]
    )

    transformed_coords = lidar_coords @ transform_matrix.T

    x_ego = (
        (transformed_coords[:, :, :, 0] - pc_range[0]) / ego_resolution[0]
    ).astype(np.int32)
    y_ego = (
        (transformed_coords[:, :, :, 1] - pc_range[1]) / ego_resolution[1]
    ).astype(np.int32)
    z_ego = (
        (transformed_coords[:, :, :, 2] - pc_range[2]) / ego_resolution[2]
    ).astype(np.int32)

    valid_mask = (
        (x_ego >= 0)
        & (x_ego < 200)
        & (y_ego >= 0)
        & (y_ego < 200)
        & (z_ego >= 0)
        & (z_ego < 16)
    )

    x_lidar = x_grid[valid_mask].astype(np.int32)
    y_lidar = y_grid[valid_mask].astype(np.int32)
    z_lidar = z_grid[valid_mask].astype(np.int32)
    x_ego = x_ego[valid_mask]
    y_ego = y_ego[valid_mask]
    z_ego = z_ego[valid_mask]

    occ_trans[x_lidar, y_lidar, z_lidar] = voxel_semantics[x_ego, y_ego, z_ego]
    mask_lidar_trans[x_lidar, y_lidar, z_lidar] = mask_lidar[
        x_ego, y_ego, z_ego
    ]
    mask_camera_trans[x_lidar, y_lidar, z_lidar] = mask_camera[
        x_ego, y_ego, z_ego
    ]

    return occ_trans, mask_lidar_trans, mask_camera_trans


def get_lidar_occ_info(
    gt_occ_info: Dict[str, np.ndarray],
    lidar2global: np.ndarray,
    cam_ego2global: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Transform ego-frame occ GT to lidar frame (matches HAT _get_lidar_occ)."""  # noqa: E501
    voxel_semantics = np.asarray(gt_occ_info["voxel_semantics"])
    mask_lidar = np.asarray(gt_occ_info["mask_lidar"])
    mask_camera = np.asarray(gt_occ_info["mask_camera"])

    global2camego = np.linalg.inv(cam_ego2global)
    lidar2camego = global2camego @ lidar2global
    lidar_voxel_semantics, mask_lidar, mask_camera = get_occ_ego2lidar(
        voxel_semantics, mask_lidar, mask_camera, lidar2camego
    )
    return {
        "ego_voxel_semantics": voxel_semantics,
        "voxel_semantics": lidar_voxel_semantics,
        "mask_lidar": mask_lidar,
        "mask_camera": mask_camera,
        "lidar2camego": lidar2camego,
    }
