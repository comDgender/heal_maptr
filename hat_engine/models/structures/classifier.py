# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import logging
import os

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

__all__ = ["Classifier", "ClassifierIrInfer"]

logger = logging.getLogger(__name__)

YUV420_BYTES_NUMERATOR = 3
YUV420_BYTES_DENOMINATOR = 2
YUV420_UV_AREA_DIVISOR = 4
NV12_UV_PACKED_FACTOR = 2


@OBJECT_REGISTRY.register
class Classifier(nn.Module):
    """
    The basic structure of classifier.

    Args:
        backbone: Backbone module.
        losses: Losses module.
        make_backbone_graph: whether to use cuda_graph in backbone.
        num_warmup_iters: Num of iters for warmup of cuda_graph.

    """

    def __init__(
        self,
        backbone,
        losses=None,
        make_backbone_graph=False,
        num_warmup_iters=3,
    ):
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.losses = losses

        self.make_backbone_graph = make_backbone_graph
        self.num_warmup_iters = num_warmup_iters
        self.graph_backbone = None
        self.stream = None

        if self.make_backbone_graph:
            assert (
                os.environ.get("HAT_USE_CUDAGRAPH", "0") == "1"
            ), "Please set HAT_USE_CUDAGRAPH=1 while use cuda-graph."

    @fx_wrap()
    def _post_process(self, preds, target):
        if target is None:
            return preds

        if not self.training or self.losses is None:
            return preds, target

        losses = self.losses(preds, target)
        return preds, losses

    @fx_wrap()
    def _pre_process(self, data, target=None):
        if isinstance(data, torch.Tensor):
            image = data
        else:
            image = data["img"]
            target = data.get("labels", None)
        return image, target

    def forward(self, data, target=None):
        image, target = self._pre_process(data, target)

        if self.make_backbone_graph:
            if self.graph_backbone is None:
                self.graph_backbone = torch.cuda.make_graphed_callables(
                    self.backbone,
                    (image,),
                    num_warmup_iters=self.num_warmup_iters,
                )
                preds = self.graph_backbone(image)
            else:
                preds = self.graph_backbone(image)
        else:
            preds = self.backbone(image)

        return self._post_process(preds, target)

    def fuse_model(self):
        for module in [self.backbone, self.losses]:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat_engine.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        if self.backbone is not None:
            if hasattr(self.backbone, "set_qconfig"):
                self.backbone.set_qconfig()

        if self.losses is not None:
            self.losses.qconfig = None

    def set_calibration_qconfig(self):
        from hat_engine.utils import qconfig_manager

        self.calibration_qconfig = (
            qconfig_manager.get_default_calibration_qconfig()
        )
        if self.losses is not None:
            self.losses.qconfig = None


@OBJECT_REGISTRY.register
class ClassifierIrInfer(nn.Module):
    """
    The basic structure of ClassifierIrInfer.

    Args:
        ir_model: Loaded IR model module.
        is_nv12_type: Whether the model expects NV12 inputs.
    """

    def __init__(
        self,
        ir_model: nn.Module,
        is_nv12_type: bool = False,
    ):
        super().__init__()
        self.ir_model = ir_model
        self.is_nv12_type = is_nv12_type

    def nv12_runtime(self, data):
        """Convert the input image tensor to NV12 layout.

        Args:
            data: Input sample dict containing key ``img``.

        Returns:
            dict: Dictionary containing NV12 image planes.
        """
        if cv2 is None or np is None:
            raise ImportError(
                "ClassifierIrInfer nv12_runtime requires cv2 and numpy."
            )

        def img2nv12(input_image, device):
            image = input_image.astype(np.uint8)
            image = image.squeeze(0)
            image = np.transpose(image, (1, 2, 0))
            height, width = image.shape[0], image.shape[1]
            yuv420p = cv2.cvtColor(image, cv2.COLOR_RGB2YUV_I420).reshape(
                (
                    height
                    * width
                    * YUV420_BYTES_NUMERATOR
                    // YUV420_BYTES_DENOMINATOR,
                )
            )
            y = yuv420p[: height * width]
            uv_planar = yuv420p[height * width :].reshape(
                (
                    NV12_UV_PACKED_FACTOR,
                    height * width // YUV420_UV_AREA_DIVISOR,
                )
            )
            uv_packed = uv_planar.transpose((1, 0)).reshape(
                (
                    height
                    * width
                    // (YUV420_UV_AREA_DIVISOR // NV12_UV_PACKED_FACTOR),
                )
            )
            return torch.tensor(
                y.reshape(1, height, width, 1),
                dtype=torch.uint8,
                device=device,
            ), torch.tensor(
                uv_packed.reshape(1, height // 2, width // 2, 2),
                dtype=torch.uint8,
                device=device,
            )

        dict_data = {}
        if len(data["img"].shape) == 4:
            dict_data["img_y"], dict_data["img_uv"] = img2nv12(
                data["img"].cpu().numpy(), device=data["img"].device
            )
        return dict_data

    def forward(self, data):
        if self.is_nv12_type:
            input_data = self.nv12_runtime(data)
        else:
            input_data = data
        outputs = self.ir_model(input_data).squeeze(-1).squeeze(-1)
        target = data.get("labels", None)

        if target is not None:
            return outputs, target
        return outputs
