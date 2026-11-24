# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import Dict

import numpy as np
import torch
from hat_engine.registry import OBJECT_REGISTRY

__all__ = ["BevFeatureAug"]


@OBJECT_REGISTRY.register
class BevFeatureAug(object):
    """Augment bev feature.

    Args:
    bda_aug_conf: a dict including augmentation transform.
        ex. bda_aug_conf = dict(
            rot_lim=(-0.0, 0.0),
            scale_lim=(1.0, 1.0),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5,
            )
        rot_lim: Random rotation angle range.
        scale_lim: The range of random scaling, in [0-1].
        flip_dx_ratio: Probability for horizontal.
        flip_dy_ratio: Probability for vertical.
    """

    def __init__(
        self,
        bda_aug_conf: Dict,
        is_train: bool = True,
    ):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train

    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""
        if not self.is_train:
            return 0, 1.0, False, False

        conf = self.bda_aug_conf
        rotate_bda = np.random.uniform(*conf["rot_lim"])
        scale_bda = np.random.uniform(*conf["scale_lim"])
        flip_dx = np.random.uniform() < conf["flip_dx_ratio"]
        flip_dy = np.random.uniform() < conf["flip_dy_ratio"]
        return rotate_bda, scale_bda, flip_dx, flip_dy

    def bev_transform(self, rotate_angle, scale_ratio, flip_dx, flip_dy):
        angle = torch.tensor(rotate_angle / 180 * np.pi)
        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        rot_mat = torch.Tensor(
            [
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1],
            ]
        )

        # Uniform scaling along all three axes.
        scale_mat = torch.eye(3) * scale_ratio

        # Axis flips are diagonal sign changes: negate x for a
        # horizontal flip and y for a vertical flip.
        flip_mat = torch.eye(3)
        if flip_dx:
            flip_mat[0, 0] = -1
        if flip_dy:
            flip_mat[1, 1] = -1

        return flip_mat @ (scale_mat @ rot_mat)

    def __call__(self, data):
        semantics = torch.tensor(
            data["gt_occ_info"]["voxel_semantics"],
            dtype=torch.uint8,
        )
        mask_lidar = torch.tensor(
            data["gt_occ_info"]["mask_lidar"],
            dtype=torch.uint8,
        )
        mask_camera = torch.tensor(
            data["gt_occ_info"]["mask_camera"],
            dtype=torch.uint8,
        )
        (
            rotate_bda,
            scale_bda,
            flip_dx,
            flip_dy,
        ) = self.sample_bda_augmentation()

        bda_mat = np.zeros((4, 4))
        bda_mat[3, 3] = 1
        bda_rot = self.bev_transform(rotate_bda, scale_bda, flip_dx, flip_dy)
        bda_mat[:3, :3] = bda_rot
        if "ego2img" in data:
            ego2img = data["ego2img"]
            for i in range(len(ego2img)):
                ego2img[i] = ego2img[i] @ bda_mat

        if flip_dx:
            semantics = torch.flip(semantics, [0])
            mask_lidar = torch.flip(mask_lidar, [0])
            mask_camera = torch.flip(mask_camera, [0])

        if flip_dy:
            semantics = torch.flip(semantics, [1])
            mask_lidar = torch.flip(mask_lidar, [1])
            mask_camera = torch.flip(mask_camera, [1])

        data["ego2img"] = ego2img
        data["gt_occ_info"]["voxel_semantics"] = semantics
        data["gt_occ_info"]["mask_lidar"] = mask_lidar
        data["gt_occ_info"]["mask_camera"] = mask_camera

        return data
