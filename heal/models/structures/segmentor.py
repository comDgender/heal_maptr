# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import logging
from inspect import signature

import numpy as np
import torch
import torch.nn as nn
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.model_helpers import fx_wrap

__all__ = ["Segmentor", "SegmentorIrInfer"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class Segmentor(nn.Module):
    """
    The basic structure of segmentor.

    Args:
        backbone (torch.nn.Module): Backbone module.
        neck (torch.nn.Module): Neck module.
        head (torch.nn.Module): Head module.
        losses (torch.nn.Module): Losses module.
    """

    def __init__(self, backbone, neck, head, losses=None):
        super(Segmentor, self).__init__()
        self.backbone = None
        self.neck = None
        self.head = None
        self.losses = None

        self.backbone = backbone
        self.neck = neck
        self.head = head

        self.losses = losses

    @fx_wrap()
    def _post_process(self, preds, target):
        """Post-process predictions for training or inference.

        Args:
            preds: Raw segmentation predictions.
            target: Optional segmentation target tensor.

        Returns:
            Predictions in inference mode, or predictions with computed losses
            in training mode.
        """
        if target is None:
            return preds

        if not self.training or self.losses is None:
            return preds, target

        losses = self.losses(preds, target)
        return preds, losses

    def forward(self, data: dict):
        """Run segmentation model forward.

        Args:
            data: Input batch dict containing ``img`` and optional ``gt_seg``.

        Returns:
            Model predictions, or predictions with losses during training.
        """
        image = data["img"]
        target = data.get("gt_seg", None)
        if "uv_map" in signature(self.backbone.forward).parameters:
            features = self.backbone(image, uv_map=data.get("uv_map", None))
        else:
            features = self.backbone(image)
        preds = self.neck(features)
        preds = self.head(preds)

        return self._post_process(preds, target)

    def fuse_model(self):
        """Fuse quantization-friendly child modules."""
        for module in [self.backbone, self.neck, self.head, self.losses]:
            if module is not None and hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        """Set QAT qconfig for model parts; disable loss quantization."""
        from hat_engine.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        for module in [self.backbone, self.neck, self.head]:
            if module is not None:
                if hasattr(module, "set_qconfig"):
                    module.set_qconfig()
        if self.losses is not None:
            self.losses.qconfig = None


@OBJECT_REGISTRY.register
class SegmentorIrInfer(nn.Module):
    """
    The basic structure of SegmentorIrInfer.

    Args:
        ir_model: The ir model.
    """

    def __init__(
        self,
        ir_model,
        use_nv12=False,
    ):
        super().__init__()
        self.ir_model = ir_model
        self.use_nv12 = use_nv12

    def nv12_runtime(self, data):
        """Convert RGB image tensor to NV12 input planes.

        Args:
            data: Input image tensor in NCHW or CHW layout.

        Returns:
            dict[str, torch.Tensor]: ``img_y`` and ``img_uv`` NV12 planes.
        """
        import cv2

        def img2nv12(input_image):
            image = input_image
            image = image.squeeze(0)
            image = np.transpose(image, (1, 2, 0))
            height, width = image.shape[0], image.shape[1]
            image = image.astype(np.uint8)
            yuv420p = cv2.cvtColor(image, cv2.COLOR_RGB2YUV_I420).reshape(
                (height * width * 3 // 2,)
            )
            y = yuv420p[: height * width]
            uv_planar = yuv420p[height * width :].reshape(
                (2, height * width // 4)
            )
            uv_packed = uv_planar.transpose((1, 0)).reshape(
                (height * width // 2,)
            )
            return torch.tensor(
                y.reshape(1, height, width, 1), dtype=torch.uint8
            ), torch.tensor(
                uv_packed.reshape(1, height // 2, width // 2, 2),
                dtype=torch.uint8,
            )

        dict_data = {}
        dict_data["img_y"], dict_data["img_uv"] = img2nv12(data.cpu().numpy())
        return dict_data

    def forward(self, data):
        """Run IR inference and return optional target for evaluation.

        Args:
            data: Input batch dict containing ``img`` and optional ``gt_seg``.

        Returns:
            IR model outputs, optionally paired with target when present.
        """
        target = data.get("gt_seg", None)

        if self.use_nv12 is True:
            nv12_img = self.nv12_runtime(data["img"])
            data.update(nv12_img)

        outputs = self.ir_model(data)

        if target is not None:
            return outputs, target
        return outputs
