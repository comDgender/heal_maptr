# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

"""Sparse MapTR model structure.

End-to-end pipeline::

    img -> HENet backbone -> FPN neck -> SparseMapPerceptionDecoder
         -> SparseMapHead (sparse queries + deformable aggregation)
         -> map vector predictions (class + polyline points)

Unlike dense MapTR / BEVFormer, there is no BEV encoder: map elements are
represented as a fixed set of learnable anchor queries that directly sample
multi-view image features via ``DeformableFeatureAggregationOEv2``.

Quantization: calibration-only (no QAT training). QuantStub / DeQuantStub mark
export boundaries; per-op qconfig is managed by ``QconfigSetter`` in the task
config (``configs/.../qconfig.py``), not by ``fuse_model`` / ``set_qconfig``
hooks on the model structure.
"""

import logging
from typing import Dict, List, Sequence

import numpy as np
import torch
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.apply_func import _as_list
from hat_engine.utils.model_helpers import fx_wrap
from torch import Tensor, nn

from heal.models.task_modules.sparse_maptr.head import SparseMapHead

logger = logging.getLogger(__name__)

__all__ = ["SparseMapTR", "SparseMapIrInfer"]


@OBJECT_REGISTRY.register
class SparseMapTR(nn.Module):
    """Sparse MapTR detector: image FPN features + sparse query decoder.

    Args:
        backbone: Image encoder (e.g. HENet).
        out_indices: Which backbone stage outputs feed the neck.
        neck: Multi-scale fusion neck (e.g. FPN).
        bev_decoders: Perception decoders; name kept for HAT config compat.
            Sparse MapTR passes ``None`` as BEV embed and uses FPN feats only.
    """

    def __init__(
        self,
        backbone: nn.Module,
        out_indices: Sequence[int] = (1, 2, 3, 4),
        neck: nn.Module = None,
        bev_decoders: List[nn.Module] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.out_indices = out_indices
        self.neck = neck
        self.bev_decoders = nn.ModuleList(bev_decoders)

    def extract_feat(self, img: Tensor) -> List[Tensor]:
        """Backbone + neck; returns multi-level FPN features per camera."""
        x = self.backbone(img)
        x = [x[i] for i in self.out_indices]
        if self.neck is not None:
            x = self.neck(x)
        return x

    def forward(self, data: Dict) -> List:
        """Run all perception decoders and merge their outputs."""
        feats = self.extract_feat(data["img"])
        results = None
        for decoder in self.bev_decoders:
            # bev_embed is unused; sparse decoder reads mlvl_feats directly.
            result = decoder(None, feats, data)
            results = self._update_res(result, results)
        return results

    @fx_wrap()
    def _update_res(self, result, results):
        if results is None:
            results = _as_list(result)
        else:
            results.extend(_as_list(result))
        return results


@OBJECT_REGISTRY.register
class SparseMapIrInfer(nn.Module):
    """HBIR/BPU inference wrapper for sparse MapTR.

    Runs compiled ``quantized.bc`` on device, then reuses the PyTorch decoder's
    ``get_outputs`` / ``post_process`` for metric eval and visualization.
    """

    def __init__(
        self,
        ir_model: nn.Module,
        test_model: nn.Module,
        model_convert_pipeline: List[callable],
        projection_mat_key: str = "lidar2img",
        use_nv12: bool = False,
    ):
        super().__init__()
        self.ir_model = ir_model
        self.projection_mat_key = projection_mat_key
        self.use_nv12 = use_nv12
        self.test_model = model_convert_pipeline(test_model)
        self.test_model.eval()

    def nv12_runtime(self, data: Tensor) -> Dict[str, Tensor]:
        import cv2

        def img2nv12(input_image):
            image = input_image.squeeze(0)
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
        if data.shape[0] == 1:
            dict_data["img_y"], dict_data["img_uv"] = img2nv12(
                data.cpu().numpy()
            )
        else:
            for i in range(data.shape[0]):
                y, uv = img2nv12(data[i : i + 1, :, :, :].cpu().numpy())
                dict_data[f"img_{i}_y"] = y
                dict_data[f"img_{i}_uv"] = uv
        return dict_data

    def process_input(self, data):
        """Pack runtime inputs into the same layout used at export time."""
        projection_mat = SparseMapHead.gen_projection_mat(
            self.projection_mat_key, data
        ).float()
        inputs = {"projection_mat": projection_mat}
        if self.use_nv12:
            inputs.update(self.nv12_runtime(data["img"]))
        else:
            inputs["img"] = data["img"]
        return inputs

    def forward(self, data):
        inputs = self.process_input(data)
        device = (
            inputs["projection_mat"].device
            if self.use_nv12
            else inputs["img"].device
        )
        outputs = self.ir_model(inputs)

        if isinstance(outputs, (tuple, list)) and isinstance(
            outputs[0], Tensor
        ):
            outputs_classes, reference_out = outputs
        elif isinstance(outputs, (tuple, list)) and isinstance(
            outputs[0], dict
        ):
            outputs_classes = outputs[0]["classification"]
            reference_out = outputs[0]["prediction"]
        elif isinstance(outputs, dict):
            outputs_classes = outputs["classification"]
            reference_out = outputs["prediction"]
        else:
            raise TypeError("Unexpected output type from ir_model.")

        outputs_classes = outputs_classes.to(device)
        reference_out = reference_out.to(device)
        # HBIR outputs raw head tensors; decode to MapTR prediction format.
        outputs = self.test_model.bev_decoders[0].get_outputs(
            outputs_classes=[outputs_classes],
            reference_out=[reference_out],
        )
        return self.test_model.bev_decoders[0].post_process(outputs, data)


# Backward-compatible aliases for legacy HAT config names.
OBJECT_REGISTRY.alias("MapTROE")(SparseMapTR)
OBJECT_REGISTRY.alias("MapTROEIrInfer")(SparseMapIrInfer)
