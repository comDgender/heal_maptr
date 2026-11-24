# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
import logging
from typing import Dict

import torch
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.model_helpers import fx_wrap
from torch import nn

logger = logging.getLogger(__name__)

__all__ = ["GaNet", "GaNetIrInfer"]


@OBJECT_REGISTRY.register
class GaNet(nn.Module):
    """The basic structure of GaNet.

    Args:
        backbone: Backbone module.
        neck: Neck module.
        head: Head module.
        targets: Target module.
        post_process: Post process module.
        losses: Loss module.

    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module = None,
        head: nn.Module = None,
        targets: nn.Module = None,
        post_process: nn.Module = None,
        losses: nn.Module = None,
    ):
        super(GaNet, self).__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.losses = losses
        self.targets = targets
        self.post_process = post_process

    @fx_wrap()
    def _post_process(self, data, kpts_hm, pts_offset, int_offset):
        if self.post_process is None:
            return kpts_hm, pts_offset, int_offset

        # 不进入 decoder 的两种情形：
        # 1. jit-strip trace：GaNetDecoder 内含 .cpu().numpy()、按 batch 的
        #    Python 循环、变长 lane 列表等数据依赖控制流，不可 trace。
        #    @fx_wrap 在 jit-strip 模式下不生效，故用 Tracer.is_tracing() 判断。
        # 2. 输入不带 scale_factor/crop_offset：decoder 靠它们把坐标映射回原图，
        #    缺了就无法工作。prepare() 会把同一份 example_inputs 转手交给
        #    check_qat_model 跑一次普通 forward（check_model.py:620），此时
        #    is_tracing() 已为 False，而 deploy_inputs 只有 img，会 KeyError。
        from horizon_plugin_pytorch.fx.jit_scheme import Tracer

        if Tracer.is_tracing() or "scale_factor" not in data:
            return kpts_hm, pts_offset, int_offset

        if self.training:
            ganet_target = self.targets(data)
            outputs = self.losses(
                kpts_hm, pts_offset, int_offset, ganet_target
            )
            return outputs

        else:
            results = self.post_process(kpts_hm, pts_offset, int_offset, data)
            return results

    def forward(self, data: Dict):

        feats = self.backbone(data["img"])

        if self.neck is not None:
            feats = self.neck(feats)

        kpts_hm, pts_offset, int_offset = self.head(feats[0])

        return self._post_process(data, kpts_hm, pts_offset, int_offset)

    def fuse_model(self):
        for module in [self.backbone, self.neck, self.head]:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        from hat_engine.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        for module in [self.backbone, self.neck, self.head]:
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()
        if self.losses is not None:
            self.losses.qconfig = None


@OBJECT_REGISTRY.register
class GaNetIrInfer(nn.Module):
    """
    The basic structure of GaNetIrInfer.

    Args:
        ir_model: The ir model.
        post_process: Postprocess module.
        use_nv12: Whether transform the input image to nv12 (alias of
            nv12_input; kept for backward compatibility with HAT configs).
        nv12_input: Whether transform the input image to nv12.
        hbm_infer: Whether the ir_model is an HbmModule (board-side hbm).
    """

    def __init__(
        self,
        ir_model: nn.Module,
        post_process: nn.Module,
        use_nv12: bool = False,
        nv12_input: bool = False,
        hbm_infer: bool = False,
    ):
        super().__init__()
        self.ir_model = ir_model
        self.post_process = post_process
        # nv12_input takes precedence if explicitly set; else fall back to
        # use_nv12 for backward compatibility.
        self.use_nv12 = nv12_input or use_nv12
        self.hbm_infer = hbm_infer

    def nv12_runtime(self, data):
        import cv2
        import numpy as np

        def img2nv12(input_image, device):
            # print(input_image.shape)
            image = input_image.astype(np.uint8)
            image = image.squeeze(0)
            image = np.transpose(image, (1, 2, 0))
            # print(image.shape)
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
        if len(data["img"].shape) == 4:
            dict_data["img_y"], dict_data["img_uv"] = img2nv12(
                data["img"].cpu().numpy(), device=data["img"].device
            )
        return dict_data

    def forward(self, data):
        if self.use_nv12 and self.hbm_infer:
            ir_input = {"img": data["img"]}
            ir_input = self.nv12_runtime(ir_input)
            outputs = self.ir_model(ir_input)
        else:
            if self.use_nv12:
                img = self.nv12_runtime(data)
                data.update(img)
            outputs = self.ir_model(data)
        if isinstance(outputs, dict):
            outputs = list(outputs.values())
        kpts_hm, pts_offset, int_offset = outputs
        results = self.post_process(kpts_hm, pts_offset, int_offset, data)
        return results
