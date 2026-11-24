# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
import copy
import os
from typing import Optional, Tuple, Type

import cv2
import numpy as np
from hat_engine.registry import OBJECT_REGISTRY
from matplotlib import pyplot as plt

from heal.core.nus_box3d_utils import (
    bbox_ego2bev,
    bbox_ego2img,
    bbox_to_corner,
)
from heal.visualize.nuscenes import NuscenesViz

__all__ = ["MultitaskViz"]


@OBJECT_REGISTRY.register
class MultitaskViz(NuscenesViz):
    """
    The visiualize method of perception multitasks.

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
        num_pics: Optional[int] = None,
        pc_range: Tuple[float, float, float] = None,
        bev_resolution: Optional[float] = None,
    ):
        self.occ_viz = occ_viz
        self.om_viz = om_viz
        self.num_pics = num_pics
        self.bev_size = bev_size
        self.pc_range = pc_range
        self.bev_resolution = bev_resolution

        super().__init__(
            bev_size=bev_size,
            bev_resolution=bev_resolution,
            score_thresh=score_thresh,
            pc_range=pc_range,
            is_plot=True,
            use_bce=use_bce,
        )

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

    def draw_bbox_on_cam_img(
        self,
        image,
        homo,
        bboxes,
        colors=((255, 0, 0), (0, 0, 255), (0, 0, 0)),
        linewidth=2,
    ):
        def draw_rect(img, selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                pt1 = tuple(prev.astype(int))
                pt2 = tuple(corner.astype(int))
                cv2.line(img, pt1, pt2, color, linewidth)
                prev = corner

        bboxes = bbox_ego2img(bboxes, homo, image.shape[:2], self.score_thresh)

        img = image.copy()
        for bbox in bboxes:
            # 1. 画竖边
            for i in range(4):
                pt1 = tuple(bbox[i].astype(int))
                pt2 = tuple(bbox[i + 4].astype(int))
                cv2.line(img, pt1, pt2, colors[2], linewidth)

            # 2. 上下面
            draw_rect(img, bbox[:4], colors[0])
            draw_rect(img, bbox[4:], colors[1])

            # 3. 朝向线
            center_bottom_forward = np.mean(bbox[2:4], axis=0)
            center_bottom = np.mean(bbox[[2, 3, 7, 6]], axis=0)

            pt1 = tuple(center_bottom.astype(int))
            pt2 = tuple(center_bottom_forward.astype(int))
            cv2.line(img, pt1, pt2, colors[0], linewidth)

        return img

    def draw_bev(self, bboxes, axes, cur_index, tile_name=None):
        bboxes_np = (
            bboxes if isinstance(bboxes, np.ndarray) else bboxes.cpu().numpy()
        )
        if self.pc_range is None:
            bboxes_np = bbox_ego2bev(bboxes_np, self.bev_size)
        else:
            bboxes_np = bbox_ego2bev(bboxes_np, None, self.bev_resolution)
        ego_bboxes, _, _ = bbox_to_corner(bboxes_np)
        # for box in ego_bboxes:
        #     x, y = copy.deepcopy(box[:, 0]), copy.deepcopy(box[:, 1])
        #     box[:, 1], box[:, 0] = -x, y
        axes[cur_index].set_aspect("equal")
        if self.pc_range is not None:
            axes[cur_index].set_ylim(
                ymin=self.pc_range[1], ymax=self.pc_range[4]
            )
            axes[cur_index].set_xlim(
                xmin=self.pc_range[0], xmax=self.pc_range[3]
            )
            axes[cur_index].grid(False)
        else:
            W = int(self.bev_size[1] * 2 / self.bev_size[2])
            H = int(self.bev_size[0] * 2 / self.bev_size[2])
            axes[cur_index].set_ylim(ymin=0, ymax=H)
            axes[cur_index].set_xlim(xmin=0, xmax=W)
            axes[cur_index].grid(False)
        if tile_name is not None:
            axes[cur_index].set_title(tile_name)
        self.draw_ego_bboxes(axes[cur_index], ego_bboxes)

    @staticmethod
    def result_filename(sample_idx, stage=None):
        if stage:
            return f"nusc_pred_{stage}_{sample_idx}.png"
        return f"nusc_pred_{sample_idx}.png"

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
        n = num_imgs + 1
        if "bev_det" in preds:
            bboxes = preds["bev_det"][0]
        elif "ego_det" in preds:
            bboxes = preds["ego_det"][0]
        elif "lidar_det" in preds:
            bboxes = preds["lidar_det"][0]
        else:
            bboxes = []
        cols = 3
        homos = meta["lidar2img"]

        if "bev_seg" in preds:
            mask = preds["bev_seg"][0]
            if self.use_bce:
                mask += 1
            n += 1
        else:
            mask = None

        if self.num_pics is not None:
            n = self.num_pics

        fig, axes = plt.subplots(
            int(np.ceil(n / cols)),
            cols,
            figsize=(int(np.ceil(n / cols)) * 10, cols * 10),
        )
        axes = axes.flatten()
        cur_index = num_imgs
        if bboxes != []:
            new_bboxes = copy.deepcopy(bboxes)
            new_bboxes[:, 3] = bboxes[:, 4]
            new_bboxes[:, 4] = bboxes[:, 3]
            bboxes = new_bboxes
            self.draw_cam_img(axes[:num_imgs], imgs, homos, bboxes)

            if len(bboxes) != 0:
                self.draw_bev(bboxes, axes, cur_index)
                cur_index += 1
        else:
            names, images = [], []
            for _, img in enumerate(imgs):
                names.append(img["name"])
                cur_img = img["img"].numpy().squeeze()
                cur_img = np.transpose(cur_img, (1, 2, 0)).astype(np.uint8)
                images.append(cur_img)
            if "dynamic_objects_det" in preds:
                dynamic_bboxes = preds["dynamic_objects_det"][0]
                tmp_images = []
                for i, (_, cur_img) in enumerate(zip(names, images)):
                    cur_img = self.draw_bbox_on_cam_img(
                        cur_img,
                        homos[i],
                        dynamic_bboxes,
                        colors=((255, 128, 0), (255, 0, 0), (128, 0, 0)),
                    )
                    tmp_images.append(cur_img)
                images = tmp_images
                if len(dynamic_bboxes) != 0:
                    self.draw_bev(
                        dynamic_bboxes, axes, cur_index, "dynamic_objects_det"
                    )
                    cur_index += 1
            if "static_objects_det" in preds:
                static_bboxes = preds["static_objects_det"][0]
                tmp_images = []
                for i, (_, cur_img) in enumerate(zip(names, images)):
                    cur_img = self.draw_bbox_on_cam_img(
                        cur_img,
                        homos[i],
                        static_bboxes,
                        colors=((0, 255, 128), (0, 255, 0), (0, 128, 0)),
                    )
                    tmp_images.append(cur_img)
                images = tmp_images
                if len(static_bboxes) != 0:
                    self.draw_bev(
                        static_bboxes, axes, cur_index, "static_objects_det"
                    )
                    cur_index += 1
            if "discrete_map_det" in preds:
                discrete_map_bboxes = preds["discrete_map_det"][0]
                tmp_images = []
                for i, (_, cur_img) in enumerate(zip(names, images)):
                    cur_img = self.draw_bbox_on_cam_img(
                        cur_img,
                        homos[i],
                        discrete_map_bboxes,
                        colors=((0, 255, 128), (0, 255, 0), (0, 128, 0)),
                    )
                    tmp_images.append(cur_img)
                images = tmp_images
                if len(discrete_map_bboxes) != 0:
                    self.draw_bev(
                        discrete_map_bboxes,
                        axes,
                        cur_index,
                        "discrete_map_det",
                    )
                    cur_index += 1
            for i, (name, cur_img) in enumerate(zip(names, images)):
                ax = axes[i]
                ax.set_title(name)
                ax.grid(False)
                ax.imshow(cur_img)

        if mask is not None:
            self.draw_mask(axes[cur_index], mask)
            cur_index += 1
        if points is not None:
            self.plot_points(axes[cur_index], points[0], reverse=True)
            # self.draw_ego_bboxes(axes[cur_index], ego_bboxes)
            cur_index += 1

        if "occ_det" in preds:
            occ_pred = preds["occ_det"]
            vis_result = self.occ_viz.vis_occ(occ_pred)
            vis_result = np.rot90(vis_result, k=2)
            axes[cur_index].imshow(vis_result.astype(np.uint8))
            cur_index += 1

        if "om_det" in preds:
            om_pred = preds["om_det"]
            pred_labels = om_pred[0]["labels_3d"]
            pred_ins = om_pred[0]["pts_3d"]
            self.om_viz.draw_ego_lines(
                axes[cur_index], pred_ins, pred_labels, self.om_viz.pred_name
            )

            cur_index += 1

        if self.is_plot:
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                result_path = os.path.join(
                    save_path,
                    self.result_filename(self.sample_idx, stage),
                )
                plt.savefig(result_path)
            else:
                plt.show()
        self.sample_idx += 1
