# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
import copy
import logging
import os
from typing import Optional, Sequence, Tuple, Type

import numpy as np
import torch
from hat_engine.registry import OBJECT_REGISTRY
from matplotlib import pyplot as plt

from heal.core.nus_box3d_utils import (
    bbox_ego2bev,
    bbox_ego2img,
    bbox_to_corner,
)

__all__ = ["NuscenesViz", "NuscenesMultitaskViz"]

logger = logging.getLogger(__name__)


def _as_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _image_hw(img_entry):
    image = img_entry["img"]
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    else:
        image = np.asarray(image)
    image = image.squeeze()
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    return image.shape[0], image.shape[1]


@OBJECT_REGISTRY.register
class NuscenesViz(object):
    """
    The visiualize method of Nuscenes result.

    Args:
        bev_size: Size of Bev size.
        pc_range: Range of the point cloud.
        bev_resolution: Resolution of Bev.
        score_thresh: Score thresh for filtering box in plot.
        is_plot: Whether to plot image.
        use_bce: Whether use bce as segmentation result.
        cols: Number of subplot columns.
        dpi: Output PNG resolution; figsize is derived from image size and dpi.
        fig_scale: Extra scale factor applied to auto-computed figsize.
    """

    def __init__(
        self,
        bev_size: Tuple[float, float, float] = None,
        pc_range: Optional[Sequence[float]] = None,
        bev_resolution: float = None,
        score_thresh: float = 0.4,
        is_plot: bool = True,
        use_bce: bool = False,
        cols: int = 3,
        dpi: int = 150,
        fig_scale: float = 1.0,
    ):
        self.bev_size = bev_size
        self.score_thresh = score_thresh
        self.is_plot = is_plot
        self.use_bce = use_bce
        self.cols = cols
        self.dpi = dpi
        self.fig_scale = fig_scale
        self.sample_idx = 0
        self.pc_range = pc_range
        self.bev_resolution = bev_resolution

    def _cell_size_inches(self, img_h, img_w):
        cell_w = img_w / self.dpi * self.fig_scale
        cell_h = img_h / self.dpi * self.fig_scale
        return cell_w, cell_h

    def _create_figure(self, n_cam, n_extra_panels, img_h, img_w):
        """Camera grid + optional extra row(s) with correct aspect."""
        cell_w, cell_h = self._cell_size_inches(img_h, img_w)
        cam_rows = int(np.ceil(n_cam / self.cols))
        extra_rows = (
            int(np.ceil(n_extra_panels / self.cols)) if n_extra_panels else 0
        )
        fig_w = self.cols * cell_w
        # Extra panels (BEV / mask): row height equals one cell width.
        extra_row_h = cell_w
        total_h = cam_rows * cell_h + extra_rows * extra_row_h

        height_ratios = [1.0] * cam_rows
        if extra_rows:
            height_ratios.extend([cell_w / cell_h] * extra_rows)

        fig = plt.figure(figsize=(fig_w, total_h))
        gs = fig.add_gridspec(
            cam_rows + extra_rows,
            self.cols,
            height_ratios=height_ratios,
            hspace=0.12,
            wspace=0.06,
        )
        cam_axes = [
            fig.add_subplot(gs[divmod(i, self.cols)]) for i in range(n_cam)
        ]
        extra_axes = []
        for j in range(n_extra_panels):
            row = cam_rows + j // self.cols
            col = j % self.cols
            extra_axes.append(fig.add_subplot(gs[row, col]))
        return fig, cam_axes, extra_axes

    def _save_figure(self, fig, result_path):
        fig.subplots_adjust(
            left=0.02,
            right=0.98,
            top=0.96,
            bottom=0.02,
            hspace=0.12,
            wspace=0.06,
        )
        fig.savefig(result_path, dpi=self.dpi, pad_inches=0)

    @staticmethod
    def result_filename(sample_idx, stage=None):
        if stage:
            return f"nusc_pred_{stage}_{sample_idx}.png"
        return f"nusc_pred_{sample_idx}.png"

    def draw_bboxes(self, ax, bboxes, colors=("b", "r", "k"), linewidth=2):
        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                # corner = selected_corners[:, i]
                ax.plot(
                    [prev[0], corner[0]],
                    [prev[1], corner[1]],
                    color=color,
                    linewidth=linewidth,
                )
                prev = corner

        for bbox in bboxes:
            # Draw the sides
            for i in range(4):
                ax.plot(
                    [bbox[i][0], bbox[i + 4][0]],
                    [bbox[i][1], bbox[i + 4][1]],
                    color=colors[2],
                    linewidth=linewidth,
                )
            draw_rect(bbox[:4], colors[0])
            draw_rect(bbox[4:], colors[1])

            center_bottom_forward = np.mean(bbox[2:4], axis=0)
            center_bottom = np.mean(bbox[[2, 3, 7, 6]], axis=0)
            ax.plot(
                [center_bottom[0], center_bottom_forward[0]],
                [center_bottom[1], center_bottom_forward[1]],
                color=colors[0],
                linewidth=linewidth,
            )

    def draw_cam_img(self, axes, imgs, homos, bboxes):
        for i, img in enumerate(imgs):
            name = img["name"]
            image = img["img"]
            ax = axes[i]
            ax.set_title(name, fontsize=9)
            ax.grid(False)
            ax.set_aspect("equal")
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().numpy()
            else:
                image = np.asarray(image)
            image = image.squeeze()
            if image.ndim == 3 and image.shape[0] in (1, 3):
                image = np.transpose(image, (1, 2, 0))
            image = image.astype(np.uint8)
            img_bbox = bbox_ego2img(
                bboxes, homos[i], image.shape[:2], self.score_thresh
            )
            ax.imshow(image, aspect="equal")
            self.draw_bboxes(ax, img_bbox)
            ax.axis("off")

    def create_pascal_label_colormap(self):
        colormap = np.zeros((256, 3), dtype=int)
        ind = np.arange(256, dtype=int)

        for shift in reversed(range(8)):
            for channel in range(3):
                colormap[:, channel] |= ((ind >> channel) & 1) << shift
            ind >>= 3

        return colormap

    def label_to_color_image(self, label):
        if label.ndim != 2:
            raise ValueError("Expect 2-D input label")

        colormap = self.create_pascal_label_colormap()
        if np.max(label) >= len(colormap):
            raise ValueError("label value too large.")

        return colormap[label]

    def draw_mask(self, ax, mask):
        mask = mask.cpu().numpy()
        mask_img = self.label_to_color_image(mask)
        ax.grid(False)

        ax.imshow(mask_img, origin="lower")

    def draw_bev_bboxes(self, ax, bboxes):
        if self.bev_size is None:
            self.draw_bboxes(ax, bboxes)
            ax.set_ylim(ymin=self.pc_range[1], ymax=self.pc_range[4])
            ax.set_xlim(xmin=self.pc_range[0], xmax=self.pc_range[3])
            ax.grid(False)
        else:
            W = int(self.bev_size[1] * 2 / self.bev_size[2])
            H = int(self.bev_size[0] * 2 / self.bev_size[2])

            self.draw_bboxes(ax, bboxes)
            ax.set_ylim(ymin=0, ymax=H)
            ax.set_xlim(xmin=0, xmax=W)
            ax.grid(False)

    def draw_ego_bboxes(self, ax, bboxes):
        self.draw_bboxes(ax, bboxes)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(False)

    def __call__(self, imgs, preds, meta, save_path=None, stage=None):
        num_imgs = len(imgs)

        if "bev_det" in preds:
            bboxes = preds["bev_det"][0]
        elif "ego_det" in preds:
            bboxes = preds["ego_det"][0]
        elif "lidar_det" in preds:
            bboxes = preds["lidar_det"][0]
        else:
            bboxes = []

        if "bev_seg" in preds:
            mask = preds["bev_seg"][0]
            if self.use_bce:
                mask += 1
        else:
            mask = None

        if "ego2img" in meta:
            homos = meta["ego2img"]
        else:
            homos = meta["lidar2img"]
        if len(bboxes) != 0:
            bboxes = _as_numpy(bboxes)

        n_extra = int(len(bboxes) != 0) + int(mask is not None)
        img_h, img_w = _image_hw(imgs[0])
        fig, cam_axes, extra_axes = self._create_figure(
            num_imgs, n_extra, img_h, img_w
        )
        self.draw_cam_img(cam_axes, imgs, homos, bboxes)

        extra_idx = 0
        if len(bboxes) != 0:
            ax = extra_axes[extra_idx]
            extra_idx += 1
            if "bev_det" in preds:
                if self.pc_range is None:
                    bev_bboxes = bbox_ego2bev(bboxes, self.bev_size)
                else:
                    bev_bboxes = bbox_ego2bev(
                        bboxes, None, self.bev_resolution
                    )
                bev_bboxes, _, _ = bbox_to_corner(
                    bev_bboxes, self.score_thresh
                )
                self.draw_bev_bboxes(ax, bev_bboxes)
            else:
                ego_bboxes, _, _ = bbox_to_corner(bboxes, self.score_thresh)
                self.draw_ego_bboxes(ax, ego_bboxes)
        if mask is not None:
            self.draw_mask(extra_axes[extra_idx], mask)

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            result_path = os.path.join(
                save_path,
                self.result_filename(self.sample_idx, stage),
            )
            self._save_figure(fig, result_path)
            plt.close(fig)
            logger.info("Saved visualization to %s", result_path)
        elif self.is_plot:
            plt.show()
            plt.close(fig)
        else:
            plt.close(fig)
        self.sample_idx += 1


@OBJECT_REGISTRY.register
class NuscenesMultitaskViz(NuscenesViz):
    """
    The visiualize method of Nuscenes for multitasks.

    Args:
        occ_viz: The occ visiualize module.
        bev_size: Size of Bev size.
        score_thresh: Score thresh for filtering box in plot.
        is_plot: Whether to plot image.
        use_bce: Whether use bce as segmentation result.
    """

    def __init__(
        self,
        occ_viz: Optional[Type] = None,
        om_viz: Optional[Type] = None,
        bev_size: Tuple[float, float, float] = None,
        score_thresh: float = 0.4,
        is_plot: bool = True,
        use_bce: bool = False,
    ):
        self.occ_viz = occ_viz
        self.om_viz = om_viz
        super().__init__(bev_size, score_thresh, is_plot=True, use_bce=use_bce)

    def plot_points(self, ax, points: np.ndarray, reverse: bool = False):
        ax.set_aspect("equal")
        if reverse:
            ax.plot(
                points[:, 1],
                -points[:, 0],
                "orchid",
                alpha=0.3,
                linestyle="none",
                marker=".",
                markersize=1,
            )
        else:
            ax.plot(
                -points[:, 1],
                points[:, 0],
                "orchid",
                alpha=0.3,
                linestyle="none",
                marker=".",
                markersize=1,
            )

    def __call__(
        self,
        imgs,
        points=None,
        preds=None,
        meta=None,
        save_path=None,
        stage=None,
    ):
        num_imgs = len(imgs)
        if "bev_det" in preds:
            bboxes = preds["bev_det"][0]
        elif "ego_det" in preds:
            bboxes = preds["ego_det"][0]
        elif "lidar_det" in preds:
            bboxes = preds["lidar_det"][0]
        else:
            bboxes = []
        if len(bboxes) != 0:
            bboxes = _as_numpy(bboxes)
        homos = meta["lidar2img"]

        if "bev_seg" in preds:
            mask = preds["bev_seg"][0]
            if self.use_bce:
                mask += 1
        else:
            mask = None

        n_extra = int(len(bboxes) != 0) + int(mask is not None)
        n_extra += int(points is not None)
        n_extra += int("occ_det" in preds)
        n_extra += int("om_det" in preds)

        img_h, img_w = _image_hw(imgs[0])
        fig, cam_axes, extra_axes = self._create_figure(
            num_imgs, n_extra, img_h, img_w
        )
        new_bboxes = copy.deepcopy(bboxes)
        new_bboxes[:, 3] = bboxes[:, 4]
        new_bboxes[:, 4] = bboxes[:, 3]
        bboxes = new_bboxes
        self.draw_cam_img(cam_axes, imgs, homos, bboxes)

        extra_idx = 0
        ego_bboxes = None
        if len(bboxes) != 0:
            ego_bboxes, _, _ = bbox_to_corner(bboxes, self.score_thresh)
            for box in ego_bboxes:
                x, y = copy.deepcopy(box[:, 0]), copy.deepcopy(box[:, 1])
                box[:, 1], box[:, 0] = -x, y
            self.draw_ego_bboxes(extra_axes[extra_idx], ego_bboxes)
            extra_idx += 1
        if mask is not None:
            self.draw_mask(extra_axes[extra_idx], mask)
            extra_idx += 1
        if points is not None:
            ax = extra_axes[extra_idx]
            self.plot_points(ax, points[0], reverse=True)
            if ego_bboxes is not None:
                self.draw_ego_bboxes(ax, ego_bboxes)
            extra_idx += 1

        if "occ_det" in preds:
            occ_pred = preds["occ_det"]
            vis_result = self.occ_viz.vis_occ(occ_pred)
            vis_result = np.rot90(vis_result, k=2)
            extra_axes[extra_idx].imshow(
                vis_result.astype(np.uint8), aspect="equal"
            )
            extra_idx += 1

        if "om_det" in preds:
            om_pred = preds["om_det"]
            pred_labels = om_pred[0]["labels_3d"]
            pred_ins = om_pred[0]["pts_3d"]
            self.om_viz.draw_ego_lines(
                extra_axes[extra_idx],
                pred_ins,
                pred_labels,
                self.om_viz.pred_name,
            )
            extra_idx += 1

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            result_path = os.path.join(
                save_path,
                self.result_filename(self.sample_idx, stage),
            )
            self._save_figure(fig, result_path)
            plt.close(fig)
            logger.info("Saved visualization to %s", result_path)
        elif self.is_plot:
            plt.show()
            plt.close(fig)
        else:
            plt.close(fig)
        self.sample_idx += 1
