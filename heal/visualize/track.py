# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import os
from typing import Union

import cv2
import numpy as np
import torch
from hat_engine.registry import OBJECT_REGISTRY
from matplotlib import pyplot as plt

from heal.visualize.utils import plot_image

__all__ = ["TrackViz"]


def _rgb_from_hex(packed: int):
    """Unpack ``0xRRGGBB`` into an ``(R, G, B)`` tuple."""
    return ((packed >> 16) & 0xFF, (packed >> 8) & 0xFF, packed & 0xFF)


# Compact track-id palette (same colors as before, packed as 0xRRGGBB).
_ID_COLOR_HEX = (
    0x90EE90,
    0xB22222,
    0xDDA0DD,
    0x00FF00,
    0x008000,
    0xD2691E,
    0xDC143C,
    0xC0C0C0,
    0xFFE4C4,
    0x32CD32,
    0x8B008B,
    0x6495ED,
    0x8A2BE2,
    0xEE82EE,
    0xFF00FF,
    0x006400,
    0x7FFF00,
    0xFF00FF,
    0x0000CD,
    0xFF8C00,
    0xFFEFD5,
    0xC71585,
    0x7CFC00,
    0x9370DB,
    0x6A5ACD,
    0xB0C4DE,
    0x4169E1,
    0xADFF2F,
    0xFF1493,
    0xDB7093,
    0xBA55D3,
    0xC71585,
    0x9400D3,
    0xFF6347,
    0x90EE90,
    0xFFFF00,
    0xE6E6FA,
    0x0000FF,
    0x808000,
    0xBDB76B,
    0xFFFFE0,
    0x808080,
    0x696969,
    0x40E0D0,
    0xCD853F,
    0x008080,
    0x48D1CC,
    0x8B4513,
    0xFFF5EE,
    0xFAF0E6,
    0x98FB98,
    0x00FFFF,
    0x87CEEB,
    0x00BFFF,
    0xB0E0E6,
    0x00FA9A,
    0xF5FFFA,
    0xF0E68C,
    0xF5DEB3,
    0x008B8B,
    0x8FBC8F,
    0xFF0000,
    0xF08080,
    0x66CDAA,
    0x3CB371,
    0x2E8B57,
    0xA52A2A,
    0xB22222,
    0xAFEEEE,
    0xFFF8DC,
    0xDAA520,
    0xFFFAF0,
    0xFDF5E6,
    0xF4A460,
    0xD2691E,
)

ID_COLORS = tuple(_rgb_from_hex(c) for c in _ID_COLOR_HEX)


@OBJECT_REGISTRY.register
class TrackViz(object):
    """Visualize object track result on the input image.

    Adapted to HEAL visualize_processors chain. Unlike HAT which calls
    ``viz_func(vis_inputs, preds)`` with save_path as a partial arg, HEAL
    injects save_path into ``__init__`` and calls ``viz(preds, inputs)``.

    Args:
        is_plot (bool): Whether to plot image.
        thickness (int): Box border thickness.
        save_path (str): Directory to save the visualization image.
            Injected by ``tools/inference.py``.
    """

    def __init__(
        self,
        is_plot: bool = True,
        thickness: int = 2,
        save_path: str = None,
    ):
        self.is_plot = is_plot
        self.thickness = thickness
        self.save_path = save_path
        self.sample_idx = 0

    def _image_to_numpy(
        self, image: Union[torch.Tensor, np.ndarray]
    ) -> np.ndarray:
        if isinstance(image, torch.Tensor):
            image = image.squeeze().cpu().numpy()
        if (
            image.ndim == 3
            and image.shape[0] in (1, 3)
            and image.shape[-1] not in (1, 3)
        ):
            image = np.transpose(image, (1, 2, 0))
        return image

    def __call__(self, preds, inputs):
        """Draw track boxes on image.

        Args:
            preds: Model outputs — a dict keyed by img_name, each value
                is a dict with ``pred_boxes`` and ``obj_idxes``.
            inputs: Post-processed input dict, must include ``img`` or
                ``ori_img``.
        """
        vis_img = inputs.get("ori_img", inputs.get("img"))
        if isinstance(vis_img, list):
            vis_img = vis_img[0]
        image = self._image_to_numpy(vis_img)

        for _, pred in preds.items():

            pred_boxes = pred["pred_boxes"].cpu().numpy()
            obj_idxes = pred["obj_idxes"].cpu().numpy()

            for i in range(len(pred_boxes)):
                boxx = pred_boxes[i].astype(np.int32)
                idxss = obj_idxes[i]
                coll = ID_COLORS[idxss % len(ID_COLORS)]
                cv2.rectangle(
                    image,
                    (boxx[0], boxx[1]),
                    (boxx[2], boxx[3]),
                    coll,
                    self.thickness,
                )
                t_size = cv2.getTextSize(
                    "{}".format(idxss),
                    0,
                    fontScale=2 / 3,
                    thickness=1,
                )[0]
                c2 = boxx[0] + t_size[0], boxx[1] - t_size[1] - 3
                cv2.rectangle(image, (boxx[0], boxx[1]), c2, coll, -1)
                cv2.putText(
                    image,
                    "{}".format(idxss),
                    (boxx[0], boxx[1] - 2),
                    0,
                    2 / 3,
                    [225, 255, 255],
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

            if self.is_plot:
                plot_image(image, reverse_rgb=False)
            if self.save_path is not None:
                os.makedirs(self.save_path, exist_ok=True)
                result_path = os.path.join(
                    self.save_path,
                    f"track_pred_{self.sample_idx}.png",
                )
                # Save with cv2 directly to preserve original image
                # resolution. matplotlib savefig compresses to figure size
                # (default 640x480), making box labels unreadable.
                cv2.imwrite(
                    result_path,
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                )
            elif self.is_plot:
                plt.show()
            self.sample_idx += 1
        return preds
