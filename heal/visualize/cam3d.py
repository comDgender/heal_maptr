# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

"""Visualize 3D detection results by projecting 3D bboxes onto 2D images."""

import copy
import logging
import os
from typing import Any, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from hat_engine.registry import OBJECT_REGISTRY
from PIL import Image

from heal.core.box3d_utils import points_cam2img
from heal.core.cam_box3d import CameraInstance3DBoxes

logger = logging.getLogger(__name__)

# Default class names for NuScenes 3D detection.
DEFAULT_CLASS_NAMES: Tuple[str, ...] = (
    "car",
    "truck",
    "trailer",
    "bus",
    "constr_veh",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
)

# Unified color for 3D box lines and labels (RGB blue).
_BOX_COLOR: Tuple[int, int, int] = (72, 101, 241)

# 12 edges of a 3D bounding box (corner index pairs).
_LINE_INDICES = (
    (0, 1),
    (0, 3),
    (0, 4),
    (1, 2),
    (1, 5),
    (3, 2),
    (3, 7),
    (4, 5),
    (4, 7),
    (2, 6),
    (5, 6),
    (6, 7),
)


def plot_rect3d_on_img(
    img: np.ndarray,
    num_rects: int,
    rect_corners: np.ndarray,
    color: Tuple[int] = (0, 255, 0),
    thickness: int = 1,
) -> np.ndarray:
    """Plot the boundary lines of 3D rectangular on 2D images."""
    for i in range(num_rects):
        corners = rect_corners[i].astype(np.int32)
        for start, end in _LINE_INDICES:
            cv2.line(
                img,
                (corners[start, 0], corners[start, 1]),
                (corners[end, 0], corners[end, 1]),
                color,
                thickness,
                cv2.LINE_AA,
            )
    return img.astype(np.uint8)


def _project_corners(
    bboxes3d: CameraInstance3DBoxes,
    cam2img: np.ndarray,
) -> np.ndarray:
    """Project 3D box corners to 2D image coordinates."""
    cam2img_t = copy.deepcopy(cam2img)
    if not isinstance(cam2img_t, torch.Tensor):
        cam2img_t = torch.from_numpy(np.array(cam2img_t))
    assert cam2img_t.shape == torch.Size(
        [3, 3]
    ) or cam2img_t.shape == torch.Size(
        [4, 4]
    )  # noqa: E501
    cam2img_t = cam2img_t.float().cpu()

    corners_3d = bboxes3d.corners
    num_bbox = corners_3d.shape[0]
    points_3d = corners_3d.reshape(-1, 3)
    uv_origin = points_cam2img(points_3d, cam2img_t)
    uv_origin = (uv_origin - 1).round()
    return uv_origin[..., :2].reshape(num_bbox, 8, 2).numpy()


def draw_camera_bbox3d_on_img(
    bboxes3d: CameraInstance3DBoxes,
    raw_img: np.ndarray,
    cam2img: np.ndarray,
    color: Tuple[int] = (0, 255, 0),
    thickness: int = 1,
) -> np.ndarray:
    """Project the 3D bbox on 2D plane and draw on input image."""
    img = raw_img.copy()
    if bboxes3d.tensor.shape[0] == 0:
        return img
    imgfov_pts_2d = _project_corners(bboxes3d, cam2img)
    return plot_rect3d_on_img(
        img, imgfov_pts_2d.shape[0], imgfov_pts_2d, color, thickness
    )


def draw_labeled_bbox3d_on_img(
    bboxes3d: CameraInstance3DBoxes,
    labels: np.ndarray,
    scores: np.ndarray,
    raw_img: np.ndarray,
    cam2img: np.ndarray,
    class_names: Tuple[str, ...] = DEFAULT_CLASS_NAMES,
    thickness: int = 2,
) -> np.ndarray:
    """Project 3D bboxes, draw with unified color and label text."""
    img = raw_img.copy()
    num_bbox = bboxes3d.tensor.shape[0]
    if num_bbox == 0:
        return img
    imgfov_pts_2d = _project_corners(bboxes3d, cam2img)

    for i in range(num_bbox):
        cls_id = int(labels[i])
        corners = imgfov_pts_2d[i].astype(np.int32)
        for start, end in _LINE_INDICES:
            cv2.line(
                img,
                (corners[start, 0], corners[start, 1]),
                (corners[end, 0], corners[end, 1]),
                _BOX_COLOR,
                thickness,
                cv2.LINE_AA,
            )
        name = (
            class_names[cls_id]
            if cls_id < len(class_names)
            else f"cls{cls_id}"
        )
        text = f"{name} {scores[i]:.2f}"
        top_x = int(corners[:, 0].min())
        top_y = int(corners[:, 1].min()) - 5
        if top_y < 12:
            top_y = int(corners[:, 1].max()) + 18
        cv2.putText(
            img,
            text,
            (top_x, top_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            _BOX_COLOR,
            1,
            cv2.LINE_AA,
        )
    return img.astype(np.uint8)


@OBJECT_REGISTRY.register
class Cam3dViz(object):
    """Visualize 3D camera detection results for ``Inference`` runners.

    Args:
        score_thresh: Score threshold for visualization filtering.
            Defaults to 0.3 (matching HAT's visual threshold).
            The model post_process uses a lower threshold (0.05)
            internally; this only controls what is drawn.
        is_plot: Whether to save the visualization image.
        save_path: Directory to save visualization images.
        class_names: Tuple of class name strings.
    """

    def __init__(
        self,
        score_thresh: float = 0.3,
        is_plot: bool = True,
        save_path: Optional[str] = None,
        class_names: Tuple[str, ...] = DEFAULT_CLASS_NAMES,
    ):
        self.score_thresh = score_thresh
        self.is_plot = is_plot
        self.save_path = save_path
        self.class_names = class_names
        self.sample_idx = 0

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
        """Draw predicted 3D boxes projected onto the input image.

        Args:
            preds: Model outputs — a dict containing ``boxes_3d``,
                ``scores_3d``, ``labels_3d`` (and optionally ``attrs_3d``).
            inputs: Post-processed input dict, must include ``img`` or
                ``ori_img``, and ``cam2img``.
        """
        vis_img = inputs.get("ori_img", inputs.get("img"))
        if isinstance(vis_img, list):
            vis_img = vis_img[0]
        image = self._image_to_numpy(vis_img)

        cam2img = inputs.get("ori_cam2img", inputs.get("cam2img"))
        if cam2img is None:
            logger.warning("cam2img not found in inputs, skipping 3D viz.")
            return preds
        if isinstance(cam2img, list):
            cam2img = cam2img[0]
        cam2img = np.array(cam2img)

        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        if not isinstance(preds, dict):
            logger.info("No 3D boxes in predictions, skipping visualization.")
            return preds

        # Support both HAT-style keys (bboxes/scores) and HEAL-style keys
        # (boxes_3d/scores_3d).
        boxes_3d = preds.get("boxes_3d", preds.get("bboxes"))
        if boxes_3d is None:
            logger.info("No 3D boxes in predictions, skipping visualization.")
            return preds
        scores_3d = preds.get("scores_3d", preds.get("scores"))
        labels_3d = preds.get("labels_3d", preds.get("labels"))

        if isinstance(boxes_3d, CameraInstance3DBoxes):
            bboxes3d = boxes_3d
        else:
            if isinstance(boxes_3d, (list, tuple)):
                boxes_3d = boxes_3d[0]
            if not isinstance(boxes_3d, torch.Tensor):
                boxes_3d = torch.as_tensor(boxes_3d)
            if boxes_3d.numel() == 0:
                logger.info("No 3D detections to visualize.")
                return preds
            bboxes3d = CameraInstance3DBoxes(
                boxes_3d.cpu(),
                box_dim=boxes_3d.shape[-1],
            )

        if scores_3d is not None:
            if isinstance(scores_3d, (list, tuple)):
                scores_3d = scores_3d[0]
            if isinstance(scores_3d, torch.Tensor):
                scores_3d = scores_3d.cpu()
            keep = scores_3d > self.score_thresh
            _s = (
                scores_3d.numpy()
                if isinstance(scores_3d, torch.Tensor)
                else np.array(scores_3d)
            )
            if labels_3d is not None:
                _l = labels_3d
                if isinstance(_l, (list, tuple)):
                    _l = _l[0]
                _l = (
                    _l.cpu().numpy()
                    if isinstance(_l, torch.Tensor)
                    else np.array(_l)
                )
            if keep.any():
                if labels_3d is not None:
                    _l_kept = _l[keep]
                    u, c = np.unique(_l_kept, return_counts=True)
                    logger.info(
                        "Detections above %.2f: %d, per-class: %s",
                        self.score_thresh,
                        int(keep.sum()),
                        ", ".join(
                            "%s:%d"
                            % (
                                (
                                    self.class_names[int(ui)]
                                    if int(ui) < len(self.class_names)
                                    else int(ui)
                                ),
                                ci,
                            )
                            for ui, ci in zip(u, c)
                        ),
                    )
                    labels_3d = _l_kept
                bboxes3d = CameraInstance3DBoxes(
                    bboxes3d.tensor[keep],
                    box_dim=bboxes3d.box_dim,
                )
                scores_3d = _s[keep]

        num_det = bboxes3d.tensor.shape[0]
        if num_det == 0:
            logger.info(
                "No 3D detections above %.2f to visualize.",
                self.score_thresh,
            )
            self.sample_idx += 1
            return preds

        stage = inputs.get("stage", "unknown")
        if self.is_plot and self.save_path:
            save_dir = (  # noqa: E501
                os.path.join(self.save_path, stage)
                if stage
                else self.save_path  # noqa: E501
            )
            filename = f"cam3d_pred_{self.sample_idx}.png"

            if labels_3d is not None and scores_3d is not None:
                pred_img = draw_labeled_bbox3d_on_img(
                    bboxes3d,
                    labels_3d,
                    scores_3d,
                    image,
                    cam2img,
                    class_names=self.class_names,
                )
            else:
                pred_img = draw_camera_bbox3d_on_img(bboxes3d, image, cam2img)

            os.makedirs(save_dir, exist_ok=True)
            pil_img = Image.fromarray(pred_img)
            pil_img.save(os.path.join(save_dir, filename))
            logger.info("Saved 3D detection visualization to %s", save_dir)

        self.sample_idx += 1
        return preds
