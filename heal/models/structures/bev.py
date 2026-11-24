# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
from typing import Dict, List, Sequence, Tuple

import horizon_plugin_pytorch.nn as hnn
import numpy as np
import torch
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.model_helpers import fx_wrap
from torch import Tensor, nn

__all__ = [
    "Bev",
    "BevIrInfer",
]


@OBJECT_REGISTRY.register
class Bev(nn.Module):
    """BEV multi-view fusion model for ViewTransformer v2."""

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        view_transformer: nn.Module,
        bev_encoder: nn.Module,
        bev_decoders: List[nn.Module],
        bev_feat_index: int = 0,
        bev_upscale: int = 2,
        compile_model: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.view_transformer = view_transformer
        self.bev_encoder = bev_encoder
        self.bev_decoders = nn.ModuleList(bev_decoders)
        self.bev_feat_index = bev_feat_index
        self.bev_upscale = bev_upscale
        if bev_upscale > 1:
            self.resize = hnn.Interpolate(
                scale_factor=bev_upscale,
                align_corners=None,
                recompute_scale_factor=True,
            )
        self.compile_model = compile_model

    def img_encode(self, img: Tensor) -> Tensor:
        feats = self.backbone(img)
        return self.neck(feats)

    def forward(self, data: Dict) -> Tuple:
        img = data["img"]
        feat = self.img_encode(img)[self.bev_feat_index]
        bev_feat, _ = self.view_transformer(feat, data, self.compile_model)

        if self.bev_upscale > 1:
            bev_feat = self.resize(bev_feat)

        bev_feat = self.bev_encoder(bev_feat, data)
        if not isinstance(bev_feat, Sequence):
            bev_feat = [bev_feat]

        preds = None
        results = {}
        for bev_decoder in self.bev_decoders:
            pred, result = bev_decoder(bev_feat, data)
            preds, results = self._update_res(pred, result, preds, results)
        if self.compile_model:
            return preds
        return preds, results

    @fx_wrap()
    def _update_res(
        self, pred: List, result: Dict, preds: Dict, results: Dict
    ) -> Tuple[Dict, Dict]:
        if preds is None:
            preds = pred
        else:
            preds.extend(pred)
        if result:
            results.update(result)
        return preds, results

    def export_reference_points(
        self, data: Dict, feat_wh: Tuple[int, int]
    ) -> Dict:
        return self.view_transformer.export_reference_points(data, feat_wh)

    def fuse_model(self) -> None:
        for module in [
            self.backbone,
            self.neck,
            self.view_transformer,
            self.bev_encoder,
            *self.bev_decoders,
        ]:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self) -> None:
        from hat_engine.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        for module in [
            self.backbone,
            self.neck,
            self.view_transformer,
            self.bev_encoder,
            *self.bev_decoders,
        ]:
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()


@OBJECT_REGISTRY.register
class BevIrInfer(nn.Module):
    """HBIR inference wrapper for Bev."""

    def __init__(
        self,
        ir_model: nn.Module,
        deploy_model: nn.Module,
        model_convert_pipeline: List[callable],
        vt_input_hw: List[int],
        bev_decoder_infers: List[nn.Module],
        use_nv12: bool = False,
    ):
        super().__init__()
        self.ir_model = ir_model
        self.deploy_model = model_convert_pipeline(deploy_model)
        self.deploy_model.eval()
        self.bev_decoder_infers = bev_decoder_infers
        self.vt_input_hw = vt_input_hw
        self.use_nv12 = use_nv12

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
                dict_data["img_%d_y" % i] = y
                dict_data["img_%d_uv" % i] = uv
        return dict_data

    def process_input(self, data: Dict) -> Dict:
        ref_p = self.deploy_model.export_reference_points(
            data, self.vt_input_hw
        )
        data.update(ref_p)
        if self.use_nv12:
            data.update(self.nv12_runtime(data["img"]))
        return data

    def forward(self, data: Dict):
        inputs = self.process_input(data)
        outputs = self.ir_model(inputs)
        if isinstance(outputs, dict):
            outputs = list(outputs.values())

        idx = 0
        results = {}
        for decoder_infer in self.bev_decoder_infers:
            num = decoder_infer.input_num()
            decoder_input = outputs[idx : idx + num]
            _, ret = decoder_infer(decoder_input, data)
            results.update(ret)
            idx += num
        return None, results
