# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
import logging
import os
from typing import Optional

import torch
from hat_engine.registry import OBJECT_REGISTRY
from matplotlib import pyplot as plt

__all__ = ["ClsViz"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class ClsViz(object):
    """Visualize classification predictions.

    Args:
        save_path: Directory used to save rendered images.
    """

    def __init__(self, save_path: Optional[str] = None):
        self.save_path = save_path
        self.sample_idx = 0

    @staticmethod
    def _normalize_preds(preds):
        if isinstance(preds, (tuple, list)):
            preds = preds[0]
        if not isinstance(preds, torch.Tensor):
            preds = torch.as_tensor(preds)
        if preds.ndim > 1:
            preds = preds.reshape(preds.shape[0], -1)[0]
        return preds

    def __call__(
        self,
        preds: torch.Tensor,
        inputs: dict,
    ):
        """Visualize predictions and return the predicted class index.

        Args:
            preds: Prediction tensor.
            inputs: Input sample dict.

        Returns:
            int: Predicted class index.
        """
        preds = self._normalize_preds(preds)
        image = inputs["img"]
        if isinstance(image, torch.Tensor):
            image = image.squeeze().cpu().numpy()

        topk_entries = []
        if preds.numel() > 1:
            probs = torch.softmax(preds, dim=-1)
            topk = min(5, probs.shape[-1])
            topk_scores, topk_indices = torch.topk(probs, k=topk, dim=-1)
            if topk_scores.ndim > 1:
                topk_scores = topk_scores[0]
                topk_indices = topk_indices[0]

            class_to_desc = inputs.get("classes")
            for idx, score in zip(topk_indices.tolist(), topk_scores.tolist()):
                if class_to_desc is not None:
                    desc = class_to_desc.get(idx, str(idx))
                else:
                    desc = str(idx)
                topk_entries.append((idx, desc, score))
        else:
            pred_value = preds.cpu().item()
            topk_entries.append(
                (pred_value, str(pred_value), float(pred_value))
            )

        pred_value, pred_desc, pred_score = topk_entries[0]
        topk_log = ", ".join(
            [f"{idx}:{desc}({score:.4f})" for idx, desc, score in topk_entries]
        )
        logger.info(f"pred top-{len(topk_entries)}: {topk_log}")

        if self.save_path is not None:
            panel_width = 0.16
            image_width = 1.0 - panel_width
            if hasattr(image, "shape"):
                img_h, img_w = image.shape[:2]
            else:
                # PIL.Image uses size=(width, height)
                img_w, img_h = image.size
            fig_h = 6
            fig_w = max(6, fig_h * (img_w / img_h) / image_width)

            fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
            image_ax = fig.add_axes([0.0, 0.0, image_width, 1.0])
            text_ax = fig.add_axes([image_width, 0.0, panel_width, 1.0])

            image_ax.imshow(image, aspect="equal")
            image_ax.set_anchor("C")
            image_ax.axis("off")

            text_ax.set_facecolor("white")
            text_ax.text(
                0.02,
                0.98,
                f"top-{len(topk_entries)}:\n"
                + "\n".join(
                    [
                        f"{rank}. {desc} ({score:.4f})"
                        for rank, (_, desc, score) in enumerate(
                            topk_entries, 1
                        )
                    ]
                ),
                transform=text_ax.transAxes,
                va="top",
                ha="left",
                color="black",
                fontsize=9,
                linespacing=1.1,
            )
            text_ax.axis("off")
            os.makedirs(self.save_path, exist_ok=True)
            stage = inputs.get("stage")
            file_name = (
                f"cls_pred_{stage}_{self.sample_idx}.png"
                if stage
                else f"cls_pred_{self.sample_idx}.png"
            )
            result_path = os.path.join(
                self.save_path,
                file_name,
            )
            logger.info(f"Saving image to {result_path}")
            fig.savefig(
                result_path,
                facecolor=fig.get_facecolor(),
                bbox_inches="tight",
                pad_inches=0,
            )
            plt.close(fig)
        self.sample_idx += 1
        return pred_value
