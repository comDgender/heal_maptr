# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

"""BEVFormer model structure.

End-to-end pipeline::

    img (multi-cam)
      -> backbone + neck (FPN)
      -> BevFormerViewTransformer (BEV encoder: TSA + SCA)
      -> BEVFormerDetDecoder (DETR-style 3D detection head)
      -> BevFormerProcess (post-process / NMS)

Temporal fusion lives in ``BevFormerViewTransformer``: previous-frame BEV is
ego-motion aligned and fused via Temporal Self-Attention (TSA) inside
``BEVFormerEncoder``. Spatial Cross-Attention (SCA) samples multi-view image
features into BEV queries.

Deploy / HBIR inference uses ``BevFormerIrInfer``: assembles ``prev_bev``,
reference-point grids, runs ``quantized.bc``, then reuses the PyTorch decoder
``get_outputs`` / ``post_process`` for eval and visualization.
"""

import logging
from typing import Dict, List, Optional, Sequence

import torch
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.apply_func import _as_list
from hat_engine.utils.model_helpers import fx_wrap
from torch import Tensor, nn

logger = logging.getLogger(__name__)

__all__ = ["BevFormer", "BevFormerIrInfer"]


@OBJECT_REGISTRY.register
class BevFormer(nn.Module):
    """BEVFormer detector: Backbone + Neck + ViewTransformer + TaskDecoder.

    ``view_transformer`` lifts multi-view image features to a dense BEV grid;
    each entry in ``bev_decoders`` consumes that BEV tensor (e.g. 3D det head).

    Args:
        backbone: Image encoder (e.g. ResNet).
        neck: Multi-scale fusion neck (e.g. FPN); optional.
        view_transformer: BEV encoder (``BevFormerViewTransformer``).
        out_indices: Which backbone stage outputs feed the neck.
        bev_decoders: Task heads operating on BEV features.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: Optional[nn.Module] = None,
        view_transformer: nn.Module = None,
        out_indices: Sequence[int] = (5,),
        bev_decoders: List[nn.Module] = None,
    ):
        super(BevFormer, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.view_transformer = view_transformer
        self.out_indices = out_indices
        self.bev_decoders = nn.ModuleList(bev_decoders)

    def extract_feat(self, img: Tensor) -> List[Tensor]:
        """Backbone + neck; returns multi-level features for all cameras."""
        x = self.backbone(img)
        x = [x[i] for i in self.out_indices]
        if self.neck is not None:
            x = self.neck(x)
        return x

    def forward(self, data: Dict) -> List:
        """Image features -> BEV encoder -> task decoders; merge all outputs."""  # noqa: E501
        imgs = data["img"]
        feats = self.extract_feat(imgs)
        # View transformer: multi-cam FPN feats -> dense BEV embedding.
        bev_feat = self.view_transformer(feats, data)

        results = None
        for bev_decoder in self.bev_decoders:
            result = bev_decoder(bev_feat, data)
            results = self._update_res(result, results)
        return results

    @fx_wrap()
    def _update_res(self, result: List, results: List) -> List:
        if results is None:
            results = _as_list(result)
        else:
            results.extend(_as_list(result))
        return results

    def export_reference_points(self, data: Dict) -> Dict:
        """Export refrence points.

        Args:
            data: A dictionary containing the input data.
            feat_wh: View transformer input shape
                     for generationg reference points.

        Returns:
            The Reference points.
        """

        return self.view_transformer.get_reference_points_cam(data)


@OBJECT_REGISTRY.register
class BevFormerIrInfer(nn.Module):
    """HBIR/BPU inference wrapper for BEVFormer.

    Maintains ``prev_frame_info`` for temporal BEV across frames. ``process_input``  # noqa: E501
    packs the same tensors used at export (reference grids, prev BEV, ego warp).  # noqa: E501
    ``ir_model`` runs the compiled graph; decoder post-process stays in PyTorch.  # noqa: E501
    """

    def __init__(
        self,
        ir_model: nn.Module,
        deploy_model: nn.Module,
        model_convert_pipeline: List[callable],
        post_process: nn.Module = None,
        is_nv12_type: bool = False,
    ):
        super().__init__()
        self.ir_model = ir_model
        self.is_nv12_type = is_nv12_type
        self.deploy_model = model_convert_pipeline(deploy_model)
        self.deploy_model.eval()
        self.bev_decoder_postprocess = post_process
        pose_key = self.deploy_model.view_transformer.global_pose_key
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            pose_key: None,
        }

    def nv12_runtime(self, data):
        import cv2
        import numpy as np

        def img2nv12(input_image, device):
            image = input_image.astype(np.uint8)
            image = image.squeeze(0)
            image = np.transpose(image, (1, 2, 0))
            height, width = image.shape[0], image.shape[1]
            # image = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)
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
                y.reshape(1, height, width, 1),
                dtype=torch.uint8,
                device=device,
            ), torch.tensor(
                uv_packed.reshape(1, height // 2, width // 2, 2),
                dtype=torch.uint8,
                device=device,
            )

        dict_data = {}
        for key in data.keys():
            if key == "img":
                for i in range(data["img"].shape[0]):
                    (
                        dict_data[f"img_{i}_y"],
                        dict_data[f"img_{i}_uv"],
                    ) = img2nv12(
                        data["img"][i : i + 1, :, :, :].cpu().numpy(),
                        device=data["img"].device,
                    )
            else:
                dict_data[key] = data[key]
        return dict_data

    def export_reference_points(self, data: Dict) -> Dict:

        return self.deploy_model.export_reference_points(data)

    def process_input(self, data):
        """Pack runtime inputs into the layout expected by the exported HBIR graph."""  # noqa: E501
        image = data["img"]
        # SCA sampling grids (fixed per camera geometry at export time).
        ref_inputs = self.deploy_model.export_reference_points(data)

        # Temporal state: previous BEV + ego-motion normalized coords.
        temporal_inputs = self.deploy_model.view_transformer.get_deploy_prev(
            data, self.prev_frame_info
        )

        outputs = {
            "img": image,
            "prev_bev": temporal_inputs[0],
            "prev_bev_ref": temporal_inputs[1],
            "queries_rebatch_grid": ref_inputs[0],
            "restore_bev_grid": ref_inputs[1],
            "reference_points_rebatch": ref_inputs[2],
            "bev_pillar_counts": ref_inputs[3],
        }

        return outputs

    def forward(self, data):
        inputs = self.process_input(data)
        if self.is_nv12_type:
            input_data = self.nv12_runtime(inputs)
        else:
            input_data = inputs
        # Compiled BEV encoder + detection head (last-layer outputs only).
        outputs = self.ir_model(input_data)
        if isinstance(outputs, dict):
            bev_embed = outputs["_output_0"]
            outputs_classes = outputs["_output_1"]
            reference_out = outputs["_output_2"]
            outputs_coords = outputs["_output_3"]
        else:
            bev_embed, outputs_classes, reference_out, outputs_coords = outputs
        outputs = self.deploy_model.bev_decoders[0].get_outputs(
            [outputs_classes], [reference_out], [outputs_coords], bev_embed
        )
        results = None
        result = self.deploy_model.bev_decoders[0].post_process(outputs)
        results = self._update_res(result, results)
        self.update_pre_info(data, bev_embed)
        return results

    @fx_wrap()
    def _update_res(self, result: List, results: List) -> List:
        if results is None:
            results = _as_list(result)
        else:
            results.extend(_as_list(result))
        return results

    def update_pre_info(
        self,
        data,
        bev_embed,
    ) -> List[List[Tensor]]:
        """Update prev bev info."""
        self.prev_frame_info["scene_token"] = data["seq_meta"][0]["meta"][0][
            "scene"
        ]
        pose_key = self.deploy_model.view_transformer.global_pose_key
        self.prev_frame_info[pose_key] = data["seq_meta"][0][pose_key]
        self.prev_frame_info["prev_bev"] = bev_embed
