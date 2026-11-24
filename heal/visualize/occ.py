# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import logging
import os
from typing import Tuple, Union

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

try:
    import mayavi
except Exception:
    mayavi = None
from hat_engine.registry import OBJECT_REGISTRY

from heal.visualize.utils import plot_image

__all__ = ["OccViz"]

logger = logging.getLogger(__name__)


colors_map = np.array(
    [
        [0, 0, 0, 255],  # 0 undefined
        [112, 128, 144, 255],  # 1 car  orange
        [220, 20, 60, 255],  # 2 pedestrian  Blue
        [255, 127, 80, 255],  # 3 sign  Darkslategrey
        [255, 158, 0, 255],  # 4 CYCLIST  Crimson
        [233, 150, 70, 255],  # 5 traiffic_light  Orangered
        [255, 61, 99, 255],  # 6 pole  Darkorange
        [0, 0, 230, 255],  # 7 construction_cone  Darksalmon
        [47, 79, 79, 255],  # 8 bycycle  Red
        [255, 140, 0, 255],  # 9 motorcycle  Slategrey
        [255, 99, 71, 255],  # 10 building Burlywood
        [0, 207, 191, 255],  # 11 vegetation  Green
        [175, 0, 75, 255],  # 12 trunk  nuTonomy green
        [75, 0, 75, 255],  # 13 curb, road, lane_marker
        [112, 180, 60, 255],  # 14 walkable, sidewalk
        [222, 184, 135, 255],  # 15 unobsrvd
        [0, 175, 0, 255],  # 16 undefined
        [0, 0, 0, 255],  # 17 undefined
    ]
)


@OBJECT_REGISTRY.register
class OccViz(object):
    """The visiualize method of occ result.

    Args:
        ignore_idx: the idx will ignore.
        vcs_range: vcs range.
        voxel_size: voxel size.
    """

    def __init__(
        self,
        ignore_idx: int = -1,
        vcs_range: Tuple[float] = (-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
        voxel_size: Tuple[float] = (0.4, 0.4, 0.4),
        vis_bev_2d: bool = False,
        vis_occ_3d: bool = False,
    ):
        self.ignore_idx = ignore_idx
        self.vcs_range = vcs_range
        self.voxel_size = voxel_size
        self.vis_bev_2d = vis_bev_2d
        self.vis_occ_3d = vis_occ_3d

    def vis_occ(self, semantics):
        """Project the voxel grid to a top-down BEV color image.

        For every BEV cell keep the class of the highest occupied
        voxel (label 17 denotes empty space) and map it to a color.
        """
        height, width, depth = semantics.shape

        # Weight each voxel by its height index while masking out empty
        # cells, so the per-column maximum lands on the topmost
        # occupied voxel; empty columns fall back to height index 0.
        occupied = (semantics != 17).astype(np.float32)
        height_idx = np.arange(depth, dtype=np.float32).reshape(1, 1, depth)
        top_idx = np.argmax(height_idx * occupied, axis=2)

        occ_bev = np.take_along_axis(semantics, top_idx[..., None], axis=2)
        occ_bev = occ_bev.reshape(-1).astype(np.int32)

        occ_bev_vis = colors_map[occ_bev].astype(np.uint8)
        occ_bev_vis = occ_bev_vis.reshape(height, width, 4)[::-1, ::-1, :3]
        occ_bev_vis = cv2.resize(occ_bev_vis, (400, 400))
        return occ_bev_vis

    def prepare_for_3d_vis(self, occ_result):
        """Save the results as an npz file for visualization."""
        mask = (occ_result != 0) & (occ_result != self.ignore_idx)
        H, W, D = occ_result.shape
        g_xx = np.arange(0, H)
        g_yy = np.arange(0, W)
        g_zz = np.arange(0, D)

        vcs_range = torch.tensor(np.array(self.vcs_range)).float()
        resolution = (vcs_range[3:] - vcs_range[:3]) / torch.tensor([H, W, D])

        vox_origin = np.array((vcs_range[3], vcs_range[4], vcs_range[2]))
        xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz, indexing="ij")
        coords_grid = np.array([xx[mask], yy[mask], -zz[mask]]).T
        coords_grid = coords_grid.astype(np.float32)
        resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])
        coords_grid = coords_grid * (-resolution) + resolution / 2
        coords_grid_xyz = coords_grid + np.array(
            vox_origin, dtype=np.float32
        ).reshape([1, 3])
        coords_grid_xyz_seg = np.vstack(
            [
                coords_grid_xyz.T,
                occ_result[mask].reshape(-1),
            ]
        ).T

        return coords_grid_xyz_seg

    def show_occ_remap(self, fov_grid_coords):
        """Depicting 3D semantic space through rendering."""
        fov_voxels = fov_grid_coords[
            (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 17)
        ]

        figure = mayavi.mlab.figure(size=(1000, 600), bgcolor=(1, 1, 1))

        voxel_size = sum(self.voxel_size) / 3
        plt_plot_fov = mayavi.mlab.points3d(
            fov_voxels[:, 0],
            fov_voxels[:, 1],
            fov_voxels[:, 2],
            fov_voxels[:, 3],
            colormap="viridis",
            scale_factor=voxel_size,
            mode="cube",
            opacity=1.0,
            vmin=0,
            vmax=16,
        )

        colors = colors_map.astype(np.uint8)[:-1, :]

        plt_plot_fov.glyph.scale_mode = "scale_by_vector"
        plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors

        axes = np.array(
            [
                [4.0, 0.2, 1.0],
                [0.0, 1.5, 1.0],
                [0.0, 0.0, 2.5],
            ],
            dtype=np.float64,
        )
        mayavi.mlab.plot3d(
            [0, axes[0, 0]],
            [0, axes[0, 1]],
            [1, axes[0, 2]],
            color=(1, 0, 0),
            tube_radius=None,
            figure=figure,
        )
        mayavi.mlab.plot3d(
            [0, axes[1, 0]],
            [0, axes[1, 1]],
            [1, axes[1, 2]],
            color=(0, 1, 0),
            tube_radius=None,
            figure=figure,
        )
        mayavi.mlab.plot3d(
            [0, axes[2, 0]],
            [0, axes[2, 1]],
            [1, axes[2, 2]],
            color=(0, 0, 1),
            tube_radius=None,
            figure=figure,
        )

        scene = figure.scene
        scene.camera.position = [-7.08337438, 0, 7.51378558]
        scene.camera.focal_point = [-6.21734897, 0, 7.11378558]
        scene.camera.view_angle = 60.0
        scene.camera.view_up = [1.0, 0.0, 0.0]
        scene.camera.clipping_range = [0.01, 400.0]
        scene.render()

    def __call__(
        self,
        occ_pred: Union[torch.Tensor, np.ndarray],
        save_path: str = None,
    ):
        if self.vis_bev_2d:
            vis_result = self.vis_occ(occ_pred)
            plot_image(vis_result, reverse_rgb=False)
            if save_path is not None:
                pred_save_path = os.path.join(save_path, "bev_occ_2d.png")
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(pred_save_path)
                logger.info("Saving visualization to %s", pred_save_path)
            plt.close()
        if self.vis_occ_3d:
            assert mayavi is not None, (
                "Vis 3d occ must install mayavi==4.8.2 "
                "and support x display"
            )
            from mayavi import mlab

            mlab.options.offscreen = True
            fov_grid_coords = self.prepare_for_3d_vis(occ_pred)
            self.show_occ_remap(fov_grid_coords)
            if save_path is not None:
                filename = os.path.join(save_path, "occ_3d.png")
                mayavi.mlab.savefig(filename)
                mayavi.mlab.close()
            else:
                logger.info("OCC 3D VIS ONLY SUPPORT OFFLINE!")
