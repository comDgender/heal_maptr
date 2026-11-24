# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from abc import ABC
from typing import Iterable

import torch.nn as nn
from hat_engine.registry import OBJECT_REGISTRY

__all__ = ["ExtSequential"]


class QatSequentialMixin(ABC):
    """Sequential qat interface."""

    def fuse_model(self):
        for module in self:
            if hasattr(module, "fuse_model"):
                module.fuse_model()

    def set_qconfig(self):
        for module in self:
            if hasattr(module, "set_qconfig"):
                module.set_qconfig()


@OBJECT_REGISTRY.register
class ExtSequential(nn.Sequential, QatSequentialMixin):
    """A sequential container extends nn.Sequential for dict or nn.Module.

    Same as nn.Sequential, ExtSequential can only forward one input argument:

        input -> module1 -> input -> module2 -> input ...

    Args:
        modules: list/tuple of nn.Module instance.
    """

    def __init__(self, modules: Iterable[nn.Module]):
        assert isinstance(modules, (list, tuple)), (
            f"`modules` should be list/tuple of nn.Module "
            f"instance, but get {type(modules)}"
        )
        args = []
        for i, m in enumerate(modules):
            assert isinstance(m, nn.Module), (
                f"the {i}th element of `modules` should be a config dict "
                f"or an nn.Module instance, but get %s" % type(m)
            )
            args.append(m)

        super(ExtSequential, self).__init__(*args)
