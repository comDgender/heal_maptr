# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
import logging
import os
from typing import Tuple

import numpy as np
from hat_engine.registry import OBJECT_REGISTRY
from matplotlib import pyplot as plt
from PIL import Image

__all__ = ["NuscenesMapViz"]

logger = logging.getLogger(__name__)

colors_plt = ["orange", "b", "r", "g"]


def _tensor_to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


@OBJECT_REGISTRY.register
class NuscenesMapViz(object):
    """Visualize NuScenes map prediction results."""

    def __init__(
        self,
        pc_range: Tuple[float, float, float] = None,
        score_thresh: float = 0.4,
        is_plot: bool = True,
        car_img_path: str = None,
        use_lidar: bool = True,
    ):
        self.pc_range = pc_range
        self.score_thresh = score_thresh
        self.is_plot = is_plot
        self.car_img = None
        if car_img_path:
            if os.path.isfile(car_img_path):
                self.car_img = Image.open(car_img_path)
            else:
                logger.warning(
                    "Map viz car overlay not found: %s. "
                    "Continuing without ego-car image.",
                    car_img_path,
                )
        self.gt_name = "GT_fixednum_pts_MAP"
        self.pred_name = "PRED_MAP_plot"
        self.use_lidar = use_lidar
        self.sample_idx = 0

    @staticmethod
    def result_filename(sample_idx, stage=None):
        if stage:
            return f"nusc_map_pred_{stage}_{sample_idx}.png"
        return f"nusc_map_pred_{sample_idx}.png"

    def draw_cam_img(self, axes, imgs):
        for i, img in enumerate(imgs):
            ax = axes[i]
            ax.set_title(img["name"])
            ax.grid(False)
            image = np.transpose(
                _tensor_to_numpy(img["img"]).squeeze(), (1, 2, 0)
            ).astype(np.uint8)
            ax.imshow(image)

    def draw_ego_lines(self, ax, instances, labels, name):
        ax.set_xlim(self.pc_range[0], self.pc_range[3])
        ax.set_ylim(self.pc_range[1], self.pc_range[4])
        ax.grid(False)
        ax.set_title(name)
        for points, label in zip(instances, labels):
            pts = _tensor_to_numpy(points)
            x = np.array([pt[0] for pt in pts])
            y = np.array([pt[1] for pt in pts])
            ax.plot(
                x,
                y,
                color=colors_plt[label],
                linewidth=1,
                alpha=0.8,
                zorder=-1,
            )
            ax.scatter(
                x, y, color=colors_plt[label], s=2, alpha=0.8, zorder=-1
            )
        if self.car_img is not None:
            ax.imshow(self.car_img, extent=[-1.2, 1.2, -1.5, 1.5])

    def __call__(self, imgs, preds=None, gts=None, save_path=None, stage=None):
        num_imgs = len(imgs)
        n = num_imgs + (1 if preds is not None else 0)
        fig, axes = plt.subplots(int(np.ceil(n / 3)), 3, figsize=(16, 24))
        axes = axes.flatten()
        self.draw_cam_img(axes[:num_imgs], imgs)
        if preds is not None:
            self.draw_ego_lines(
                axes[num_imgs],
                preds[0]["pts_3d"],
                preds[0]["labels_3d"],
                self.pred_name,
            )
        if self.is_plot:
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                out_png = os.path.join(
                    save_path,
                    self.result_filename(self.sample_idx, stage),
                )
                plt.savefig(out_png)
                logger.info("Saving map visualization to %s", out_png)
                plt.close(fig)
            else:
                plt.show()
        self.sample_idx += 1
