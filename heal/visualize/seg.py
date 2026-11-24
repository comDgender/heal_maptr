# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
import logging
import os
from typing import Any, Callable, Optional, Union

import cv2
import numpy
import numpy as np
import torch
from hat_engine.registry import OBJECT_REGISTRY
from matplotlib import pyplot as plt
from PIL import Image
from torch import Tensor

from heal.visualize.utils import colormap, plot_image, show_images

logger = logging.getLogger(__name__)

__all__ = ["SegViz"]

_cityscapes_colormap = torch.tensor(
    [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
        [0, 0, 0],
    ]
)


def _default_image_process(image: Tensor, prefix):
    show_images(image, prefix)


def _seg_target_process(target: Tensor, prefix):
    if target.size(1) > 1:
        one_hot = torch.argmax(target, dim=1, keepdim=True)
    else:
        one_hot = target
    one_hot = colormap(one_hot, _cityscapes_colormap)
    show_images(one_hot, prefix, "hwc")


@OBJECT_REGISTRY.register
class SegViz(object):
    """
    The visualize method of segmentation result.

    Args:
        image_process (Callable): Process of image.
        label_process (Callable): Process of label.
    """

    def __init__(
        self,
        is_plot: bool,
        num_classes: int = 19,
        image_process: Callable = _default_image_process,
        label_process: Callable = _seg_target_process,
        save_path: Optional[str] = None,
    ):
        self.is_plot = is_plot
        self.num_classes = num_classes
        self.image_process = image_process
        self.label_process = label_process
        self.save_path = save_path
        self.sample_idx = 0

    @staticmethod
    def _image_to_numpy(
        image: Union[torch.Tensor, np.ndarray, Image.Image],
    ) -> np.ndarray:
        if isinstance(image, torch.Tensor):
            image = image.squeeze().detach().cpu().numpy()
        if isinstance(image, Image.Image):
            image = np.array(image)
        if (
            image.ndim == 3
            and image.shape[0] in (1, 3)
            and image.shape[-1]
            not in (
                1,
                3,
            )
        ):
            image = np.transpose(image, (1, 2, 0))
        return image

    @classmethod
    def _extract_seg_pred(cls, preds: Any, inputs: dict) -> torch.Tensor:
        if isinstance(inputs, dict) and "pred" in inputs:
            seg_pred = inputs["pred"]
        else:
            seg_pred = preds
        if isinstance(seg_pred, (list, tuple)):
            seg_pred = seg_pred[0]
            if isinstance(seg_pred, (list, tuple)):
                seg_pred = seg_pred[0]
        if not isinstance(seg_pred, torch.Tensor):
            seg_pred = torch.as_tensor(seg_pred)
        if seg_pred.ndim == 4:
            seg_pred = torch.argmax(seg_pred, dim=1, keepdim=False)
        if seg_pred.ndim == 3 and seg_pred.shape[0] == 1:
            seg_pred = seg_pred.squeeze(0)
        return seg_pred

    def __call__(self, preds: Any, inputs: dict):
        """Visualize segmentation predictions for ``Inference`` runners.

        Args:
            preds: Raw model outputs.
            inputs: Input dict with ``img`` and optional ``pred``.
        """
        seg_pred = self._extract_seg_pred(preds, inputs)
        max_cls = self.num_classes - 1
        seg_pred = torch.clamp(seg_pred, 0, max_cls)
        seg_vis = colormap(seg_pred, _cityscapes_colormap)
        seg_vis = seg_vis.detach().cpu().numpy().astype(np.uint8)

        image = None
        if isinstance(inputs, dict) and inputs.get("img") is not None:
            image = self._image_to_numpy(inputs["img"])

        fig = plt.figure(figsize=(12, 6), facecolor="white")
        if image is not None:
            ax1 = fig.add_subplot(1, 2, 1)
            plot_image(image, ax=ax1)
            ax1.set_title("input")
            ax2 = fig.add_subplot(1, 2, 2)
            plot_image(seg_vis, ax=ax2)
            ax2.set_title("pred")
        else:
            ax = fig.add_subplot(1, 1, 1)
            plot_image(seg_vis, ax=ax)

        if self.is_plot:
            if self.save_path is not None:
                os.makedirs(self.save_path, exist_ok=True)
                stage = (
                    inputs.get("stage") if isinstance(inputs, dict) else None
                )
                file_name = (
                    f"seg_pred_{stage}_{self.sample_idx}.png"
                    if stage
                    else f"seg_pred_{self.sample_idx}.png"
                )
                result_path = os.path.join(self.save_path, file_name)
                logger.info(
                    "Saving segmentation visualization to %s", result_path
                )
                fig.savefig(
                    result_path,
                    facecolor=fig.get_facecolor(),
                    bbox_inches="tight",
                    pad_inches=0,
                )
            else:
                plt.show()
        plt.close(fig)
        self.sample_idx += 1
        return seg_pred


def colorize(img, colors):
    """Colorize img based on colors."""
    assert img.ndim == 2
    palette = np.asarray(colors, dtype=np.float64)
    seg = img.astype(np.int64, copy=False)
    seg = np.where(seg == 255, 0, seg)
    return palette[seg]


def semantic_segmentation(
    _seg,
    colors,
    image=None,
    hwc=False,
    alpha=0.5,
    keep_shape_as_image=False,
):

    if torch and isinstance(_seg, torch.Tensor):
        _seg = _seg.cpu().numpy()

    seg = _seg.copy()
    assert isinstance(seg, np.ndarray)

    if seg.ndim == 2:
        seg = seg
    elif seg.ndim == 3:
        if hwc:
            seg = np.transpose(seg, (2, 0, 1))
        seg = np.argmax(seg, axis=0)
    else:
        raise ValueError("bad input ndim: %d" % seg.ndim)
    h, w = seg.shape

    mask_color = colorize(seg, colors)

    if image is not None:
        if keep_shape_as_image:
            image_h, image_w = image.shape[:2]
            if image_h != h or image_w != w:
                mask_color = cv2.resize(mask_color, (image_w, image_h))
            h, w = image_h, image_w
        if image.shape[0] != h:
            image_resize = cv2.resize(image, (w, h))
        else:
            image_resize = image
        blended = image_resize * alpha + mask_color * (1 - alpha)
    else:
        blended = mask_color
    return blended
