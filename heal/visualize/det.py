# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
import logging
import os
from typing import Any, List, Optional, Union

import numpy as np
import torch
from hat_engine.registry import OBJECT_REGISTRY
from matplotlib import pyplot as plt
from PIL import Image

from heal.visualize.utils import plot_bbox

__all__ = ["DetViz"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class DetViz(object):
    """Visualize detection predictions for ``Inference`` runners.

    Args:
        score_thresh: Score threshold for drawing boxes.
        absolute_coord: Whether box coordinates are absolute pixels.
        is_plot: Whether to render and save/show the figure.
        save_path: Directory to save visualization images.
        class_names: Class names indexed by label id.
    """

    def __init__(
        self,
        score_thresh: float = 0.05,
        absolute_coord: bool = True,
        is_plot: bool = True,
        save_path: Optional[str] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.score_thresh = score_thresh
        self.absolute_coord = absolute_coord
        self.is_plot = is_plot
        self.save_path = save_path
        self.class_names = class_names
        self.sample_idx = 0

    @staticmethod
    def _extract_detections(preds: Any) -> torch.Tensor:
        if isinstance(preds, dict):
            if "pred_bboxes" in preds:
                detections = preds["pred_bboxes"]
            else:
                raise KeyError(
                    "Detection visualize expects 'pred_bboxes' in output."
                )
        else:
            detections = preds

        if isinstance(detections, (list, tuple)):
            detections = detections[0]
        if not isinstance(detections, torch.Tensor):
            detections = torch.as_tensor(detections)
        return detections

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
            and image.shape[-1]
            not in (
                1,
                3,
            )
        ):
            image = np.transpose(image, (1, 2, 0))
        return image

    def __call__(self, preds: Any, inputs: dict):
        """Draw predicted boxes on the input image.

        Args:
            preds: Model outputs, usually a dict containing ``pred_bboxes``.
            inputs: Post-processed input dict, must include key ``img``.
        """
        vis_img = inputs.get("ori_img", inputs["img"])
        if isinstance(vis_img, list):
            vis_img = vis_img[0]
        image = self._image_to_numpy(vis_img)
        detections = self._extract_detections(preds)
        det = detections.detach().cpu().numpy()

        pred_label = None
        pred_score = None
        pred_bbox = np.zeros((0, 4))
        if det.size > 0:
            pred_label = det[:, -1]
            pred_score = det[:, -2]
            pred_bbox = det[:, 0:4]
            num_vis = int((pred_score >= self.score_thresh).sum())
            logger.info(
                "detections: %d total, %d with score >= %.3f",
                len(det),
                num_vis,
                self.score_thresh,
            )
        else:
            logger.info("No detections to visualize.")

        img_h, img_w = image.shape[:2]
        fig_w = max(8.0, img_w / 100.0)
        fig_h = max(6.0, img_h / 100.0)
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
        ax = fig.add_subplot(1, 1, 1)
        plot_bbox(
            img=image,
            bboxes=pred_bbox,
            labels=pred_label,
            scores=pred_score,
            thresh=self.score_thresh,
            class_names=self.class_names,
            absolute_coordinates=self.absolute_coord,
            ax=ax,
        )

        if self.is_plot:
            if self.save_path is not None:
                os.makedirs(self.save_path, exist_ok=True)
                stage = inputs.get("stage")
                file_name = (
                    f"det_pred_{stage}_{self.sample_idx}.png"
                    if stage
                    else f"det_pred_{self.sample_idx}.png"
                )
                result_path = os.path.join(self.save_path, file_name)
                logger.info(
                    "Saving detection visualization to %s", result_path
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
        return detections
