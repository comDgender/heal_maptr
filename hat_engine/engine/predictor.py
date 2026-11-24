# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
# type: ignore

import copy
import gc
import logging
from typing import Dict, Iterable, List, Optional, Sequence, Union

import torch
import torch.nn as nn

from hat_engine.callbacks import CallbackMixin
from hat_engine.models.ir_modules.hbm_module import close_hbm_servers
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.apply_func import _as_list
from .ddp_trainer import launch
from .launcher import register_launcher
from .loop_base import LoopBase, LoopType
from .processors import BatchProcessorMixin

try:
    from horizon_plugin_pytorch.quantization import (
        FakeQuantState,
        set_fake_quantize,
    )
except (RuntimeError, ImportError):
    set_fake_quantize = None
    FakeQuantState = None


__all__ = ["Predictor"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register_module
class Predictor(LoopBase):
    """Predictor is a tool for predict.

    The abundant callbacks in trainer is also supported.

    Predictor supports to launch multi-process on single gpu.
    Predictor supports multi dataloaders.

    Args:
        model: `nn.Module` instance.
        data_loader: Validation data loader.
        batch_processor: Batch processor config.
        model_convert_pipeline: Define the process of model convert.
            e.g. convert float model to qat model, convert qat model
            to quantize model.
        callbacks: Callbacks.
        metrics: Metrics on predict data.
        profiler: To profile individual steps during predicting and
            assist in identifying bottlenecks.
        log_interval: Logging output frequency.
        share_callbacks: Whether to share callbacks on different dataloader.
        compiler: Converter of `torch.compile`.
    """

    def __init__(
        self,
        model: nn.Module,
        data_loader: Iterable,
        batch_processor: BatchProcessorMixin,
        device: Optional[int] = None,
        num_epochs: int = 1,
        model_convert_pipeline: Optional[Union[Dict, List]] = None,
        callbacks: Optional[Sequence[Union[dict, CallbackMixin]]] = None,
        metrics: Optional[dict] = None,
        profiler: Optional[dict] = None,
        log_interval: int = 0,
        share_callbacks: bool = True,
        compiler: Optional[dict] = None,
        **kwargs,
    ):
        if callbacks is None:
            callbacks = []
        if not isinstance(data_loader, (list, tuple)):
            callbacks = [callbacks]
        elif share_callbacks:
            callbacks = [callbacks for _ in range(len(data_loader))]
        else:
            assert len(data_loader) == len(callbacks)
        self.data_loaders = _as_list(data_loader)
        self.multi_callbacks = callbacks

        super().__init__(
            model=model,
            data_loader=data_loader,
            optimizer=None,
            batch_processor=batch_processor,
            model_convert_pipeline=model_convert_pipeline,
            device=device,
            num_epochs=num_epochs,
            callbacks=callbacks,
            train_metrics=metrics,
            profiler=profiler,
            log_interval=log_interval,
            compiler=compiler,
            loop_type=LoopType.VALIDATION,
            **kwargs,
        )

        assert batch_processor.need_grad_update is False
        self.profiler.setup(stage="validation")

    def on_epoch_begin(self, **kwargs):
        self.model.eval()
        if set_fake_quantize is not None:
            set_fake_quantize(self.model, FakeQuantState.VALIDATION)
        super(Predictor, self).on_epoch_begin(**kwargs)

    @torch.no_grad()
    def fit(self):
        for data_loader, callbacks in zip(
            self.data_loaders, self.multi_callbacks
        ):

            try:
                tmp_loader = copy.deepcopy(data_loader)
            except Exception as e:
                logger.warning(
                    f"Predictor: deepcopy `data_loader` failed: {e}, will"
                    f"use `data_loader` itself and not clear memory cache."
                )
                tmp_loader = None

            self.data_loader = (
                data_loader if tmp_loader is None else tmp_loader
            )
            self.set_callbacks(callbacks)
            super().fit()

            # empty memory cache
            if tmp_loader is not None:
                del tmp_loader
                gc.collect()

        close_hbm_servers(self.model)


register_launcher("Predictor", launch)
