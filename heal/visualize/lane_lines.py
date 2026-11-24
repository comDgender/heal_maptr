# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import os
from typing import Any, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from hat_engine.registry import OBJECT_REGISTRY
from matplotlib import pyplot as plt
from PIL import Image

from heal.visualize.utils import plot_image

__all__ = ["LanelineViz"]


@OBJECT_REGISTRY.register
class LanelineViz(object):
    """
    The visualize method of Laneline detection result.

    Args:
        is_plot: Whether to plot image.
        thickness: The thickness of lane line.
        color: The color of lane line.
        save_path: Directory to save the visualization image. If None,
            the image is shown via ``plt.show`` (or not saved). Injected by
            ``tools/inference.py`` into ``__init__`` (HEAL visualize_processors
            chain), unlike HAT which passes it via ``partial`` at call time.
    """

    def __init__(
        self,
        is_plot: bool = True,
        thickness: int = 10,
        color: Tuple[int, int, int] = (255, 0, 0),
        save_path: Optional[str] = None,
    ):
        self.is_plot = is_plot
        self.thickness = thickness
        self.color = color
        self.save_path = save_path

    @staticmethod
    def _image_to_numpy(
        image: Union[torch.Tensor, np.ndarray, Image.Image],
    ) -> np.ndarray:
        if isinstance(image, torch.Tensor):
            image = image.squeeze().cpu().numpy()
        if isinstance(image, Image.Image):
            image = np.array(image)
        if (
            image.ndim == 3
            and image.shape[0] in (1, 3)
            and image.shape[-1] not in (1, 3)
        ):
            image = np.transpose(image, (1, 2, 0))
        return image

    def __call__(self, preds: Any, inputs: dict) -> Any:
        """Draw predicted lane lines on the input image.

        Args:
            preds: Model outputs — a list of lanes, each lane is an
                ``(N, 2)`` array of points.
            inputs: Post-processed input dict, must include ``img`` or
                ``ori_img``.
        """
        vis_img = inputs.get("ori_img", inputs.get("img"))
        if isinstance(vis_img, list):
            vis_img = vis_img[0]
        image = self._image_to_numpy(vis_img)

        lines = preds
        # GaNetDecoder returns list per batch sample; take the first (batch=1
        # for inference). Each lane is an (N, 2) array/tensor.
        if (
            isinstance(lines, list)
            and len(lines) == 1
            and isinstance(lines[0], list)
        ):
            lines = lines[0]
        elif (
            isinstance(lines, list)
            and len(lines)
            and isinstance(lines[0], (list, tuple))
        ):
            lines = lines[0]
        if not isinstance(lines, (list, tuple)):
            lines = [lines]

        for line in lines:
            if isinstance(line, torch.Tensor):
                line = line.detach().cpu().numpy()
            line = np.around(line).astype(np.int64)
            points_num = len(line)
            for i in range(points_num - 1):
                x1 = line[i][0]
                y1 = line[i][1]
                x2 = line[i + 1][0]
                y2 = line[i + 1][1]
                cv2.line(
                    image,
                    (x1, y1),
                    (x2, y2),
                    color=self.color,
                    thickness=self.thickness,
                )

        plot_image(image)

        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            name = inputs.get("image_name", inputs.get("img_name"))
            if isinstance(name, list):
                name = name[0] if name else None
            stem = (
                os.path.splitext(os.path.basename(str(name)))[0]
                if name
                else "sample"
            )
            result_path = os.path.join(self.save_path, "%s.png" % stem)
            plt.savefig(result_path)
        elif self.is_plot:
            plt.show()
        return preds
