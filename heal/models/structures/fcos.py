# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import logging
import re
from collections.abc import Mapping, Sequence
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

try:
    import cv2
except ImportError:
    cv2 = None
try:
    import numpy as np
except ImportError:
    np = None

from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.model_helpers import fx_wrap

logger = logging.getLogger(__name__)

YUV420_BYTES_NUMERATOR = 3
YUV420_BYTES_DENOMINATOR = 2
YUV420_UV_AREA_DIVISOR = 4
NV12_UV_PACKED_FACTOR = 2

__all__ = ["FCOS", "FCOSIrInfer"]


@OBJECT_REGISTRY.register
class FCOS(nn.Module):
    """FCOS detector with train/infer unified forward.

    It composes backbone, neck, head, optional descriptor and post-processing
    modules, and computes losses through target assignment during training.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module = None,
        head: nn.Module = None,
        targets: nn.Module = None,
        desc: nn.Module = None,
        post_process: nn.Module = None,
        loss_cls: nn.Module = None,
        loss_reg: nn.Module = None,
        loss_centerness: nn.Module = None,
    ):
        """Initialize FCOS detector.

        Args:
            backbone: Backbone feature extractor.
            neck: Optional neck module for multi-scale feature fusion.
            head: Detection head that predicts cls/reg/centerness.
            targets: Target assigner that builds loss inputs during training.
            desc: Optional prediction descriptor/adapter module.
            post_process: Optional inference post-process decoder.
            loss_cls: Classification loss module.
            loss_reg: Regression loss module.
            loss_centerness: Centerness loss module.
        """
        super(FCOS, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.loss_cls = loss_cls
        self.loss_reg = loss_reg
        self.loss_centerness = loss_centerness
        self.desc = desc
        self.targets = targets
        self.post_process = post_process

    def extract_feat(self, img):
        """Extract detector features from input image tensor.

        Args:
            img: Input image tensor.

        Returns:
            Feature tensor/list consumed by detection head.
        """
        x = self.backbone(img)
        if self.neck is not None:
            x = self.neck(x)
        return x

    @fx_wrap()
    def _post_process(self, data, preds):
        """Compute training losses or decode predictions for inference.

        Args:
            data: Input batch dict containing labels/meta info.
            preds: Raw predictions from detection head (or desc module).

        Returns:
            Training: merged loss dict.
            Inference: decoded outputs or raw preds when post-process is None.
        """
        if self.training and self.targets is not None:
            cls_targets, reg_targets, centerness_targets = self.targets(
                data, preds
            )
            cls_loss = self.loss_cls(**cls_targets)
            reg_loss = self.loss_reg(**reg_targets)
            centerness_loss = self.loss_centerness(**centerness_targets)
            return dict(**cls_loss, **reg_loss, **centerness_loss)
        if self.post_process is None:
            return preds
        return self.post_process(preds, data)

    def forward(self, data: Dict):
        """Run FCOS forward for either training or inference.

        Args:
            data: Input batch dict, must include key `img`.

        Returns:
            Loss dict in training mode, otherwise inference predictions.
        """
        imgs = data["img"]
        feats = self.extract_feat(imgs)
        preds = self.head(feats)
        if self.desc is not None:
            preds = self.desc(preds)
        return self._post_process(data, preds)


@OBJECT_REGISTRY.register
class FCOSIrInfer(nn.Module):
    """FCOS IR inference wrapper: ir_model + post_process."""

    def __init__(
        self,
        ir_model: nn.Module,
        num_class: int = 80,
        post_process: nn.Module = None,
        is_nv12_type: bool = False,
    ):
        """Initialize FCOS IR inference wrapper.

        Args:
            ir_model: Exported IR model used for forward inference.
            num_class: Number of classes, kept for compatibility.
            post_process: Optional decoder/post-process module.
            is_nv12_type: Whether the model expects NV12 inputs.
        """
        super().__init__()
        self.ir_model = ir_model
        self.post_process = post_process
        self.num_class = num_class
        self.is_nv12_type = is_nv12_type

    def nv12_runtime(self, data):
        """Convert uint8 RGB image to NV12 planes for pyramid HBM input."""
        if cv2 is None or np is None:
            raise ImportError(
                "FCOSIrInfer nv12_runtime requires cv2 and numpy."
            )

        img = data["img"]
        if img.dim() == 4:
            if img.size(0) != 1:
                raise ValueError("NV12 runtime only supports batch_size=1.")
            img = img[0]

        color_space = data.get("color_space", "rgb")
        if isinstance(color_space, (list, tuple)):
            color_space = color_space[0]
        cvt_code = (
            cv2.COLOR_RGB2YUV_I420
            if str(color_space).lower() == "rgb"
            else cv2.COLOR_BGR2YUV_I420
        )

        image = img.detach().cpu().numpy()
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        height, width = image.shape[0], image.shape[1]
        yuv420p = cv2.cvtColor(image, cvt_code).reshape(
            (
                height
                * width
                * YUV420_BYTES_NUMERATOR
                // YUV420_BYTES_DENOMINATOR,
            )
        )
        y = yuv420p[: height * width]
        uv_planar = yuv420p[height * width :].reshape(
            (NV12_UV_PACKED_FACTOR, height * width // YUV420_UV_AREA_DIVISOR)
        )
        uv_packed = uv_planar.transpose((1, 0)).reshape((height * width // 2,))
        device = img.device
        return {
            "img_y": torch.tensor(
                y.reshape(1, height, width, 1),
                dtype=torch.uint8,
                device=device,
            ),
            "img_uv": torch.tensor(
                uv_packed.reshape(1, height // 2, width // 2, 2),
                dtype=torch.uint8,
                device=device,
            ),
        }

    @staticmethod
    def _output_sort_key(name):
        """Sort IR output names by numeric suffix (not lexicographic)."""
        if isinstance(name, str):
            match = re.search(r"(\d+)$", name)
            if match:
                return int(match.group(1))
        return name

    @staticmethod
    def _reformat_outputs(
        outputs: Union[Sequence, Mapping],
    ) -> Union[Tuple, Sequence]:
        """Group flat IR outputs into FCOS head layout.

        Exported models emit per-level tensors in order:
        cls (N levels), bbox (N levels), centerness (N levels).
        ``FCOSDecoder`` expects ``(cls_scores, bbox_preds, centernesses)``.
        """
        if isinstance(outputs, Mapping):
            keys = sorted(outputs.keys(), key=FCOSIrInfer._output_sort_key)
            return FCOSIrInfer._reformat_outputs([outputs[k] for k in keys])

        if not isinstance(outputs, (list, tuple)):
            return outputs

        if len(outputs) == 3 and all(
            isinstance(item, (list, tuple)) for item in outputs
        ):
            return outputs

        if len(outputs) % 3 != 0:
            return outputs

        num_levels = len(outputs) // 3
        cls_scores = list(outputs[:num_levels])
        bbox_preds = list(outputs[num_levels : 2 * num_levels])
        centernesses = list(outputs[2 * num_levels :])
        return cls_scores, bbox_preds, centernesses

    def forward(self, data):
        """Run IR model inference and optional post-processing.

        Args:
            data: Input data passed to IR model.

        Returns:
            Raw IR outputs or post-processed outputs.
        """
        if self.is_nv12_type:
            input_data = self.nv12_runtime(data)
        else:
            input_data = data
        outputs = self.ir_model(input_data)
        outputs = self._reformat_outputs(outputs)
        if self.post_process is not None:
            outputs = self.post_process(outputs, data)
        return outputs
