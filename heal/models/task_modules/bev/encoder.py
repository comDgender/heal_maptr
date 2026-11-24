# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import Dict

from hat_engine.registry import OBJECT_REGISTRY
from torch import Tensor, nn

__all__ = ["BevEncoder"]


@OBJECT_REGISTRY.register
class BevEncoder(nn.Module):
    """BEV feature encoder with optional backbone and neck."""

    def __init__(self, backbone: nn.Module, neck: nn.Module = None):
        super().__init__()
        self.backbone = backbone
        self.neck = neck

    def forward(self, feat: Tensor, meta: Dict) -> Tensor:
        feat = self.backbone(feat)
        if self.neck is not None:
            feat = self.neck(feat)
        return feat

    def fuse_model(self) -> None:
        modules = [self.backbone]
        if self.neck is not None:
            modules.append(self.neck)
        for module in modules:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self) -> None:
        from hat_engine.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()
        modules = [self.backbone]
        if self.neck is not None:
            modules.append(self.neck)
        for module in modules:
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()
