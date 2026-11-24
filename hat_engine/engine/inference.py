# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

# copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.
import logging
from typing import Any, Callable, List, Optional, Union

try:
    import horizon_plugin_pytorch as horizon
except (RuntimeError, ImportError):
    horizon = None
import torch
import torch.nn as nn

try:
    from horizon_plugin_pytorch.quantization import (
        FakeQuantState,
        set_fake_quantize,
    )
except (RuntimeError, ImportError):
    FakeQuantState = None  # type: ignore
    set_fake_quantize = None  # type: ignore

from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.apply_func import _as_list, to_cuda
from hat_engine.utils.package_helper import require_packages

__all__ = ["Inference"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register_module
class Inference(object):
    """Basic Inference Runner.

    Args:
        device: Device ids used for inference.
        model: Model for inference.
        march: March of model convert.
        model_convert_pipeline: Pipeline of model convert.
        pre_processors: Transforms before model inference.
        post_processors: Inverse transforms after model inference.
        visualize_processors: Optional visualizers after post-processing.
        fake_quant_state: ``FakeQuantState`` to apply after model convert.
            Defaults to ``FakeQuantState.VALIDATION``.

    """

    @require_packages("horizon_plugin_pytorch")
    def __init__(
        self,
        device: Union[List[int], int],
        model: nn.Module,
        march: Optional[str] = None,
        model_convert_pipeline: Optional[List] = None,
        pre_processors: Optional[List[Callable]] = None,
        post_processors: Optional[List[Callable]] = None,
        visualize_processors: Optional[Union[Callable, List[Callable]]] = None,
        fake_quant_state: Optional["FakeQuantState"] = None,
    ):
        if march is None:
            logging.warning(
                "march not provided, please make sure it's expected"
            )
        else:
            horizon.march.set_march(march)

        self.model = model
        if model_convert_pipeline is not None:
            self.model = model_convert_pipeline(self.model)
        self.model.eval()

        device_ids = _as_list(device)
        if len(device_ids) > 1:
            raise NotImplementedError("Current only support single device.")
        if len(device_ids) != 0:
            device_ids[0] = (
                int(device_ids[0]) if device_ids[0] is not None else None
            )
            self.device = torch.device("cuda", device_ids[0])
        else:
            self.device = torch.device("cpu")

        self.set_device(self.device)
        self.preprocessors = (
            [] if pre_processors is None else _as_list(pre_processors)
        )
        self.postprocessors = (
            [] if post_processors is None else _as_list(post_processors)
        )
        self.visualize_processors = (
            []
            if visualize_processors is None
            else _as_list(visualize_processors)
        )
        if set_fake_quantize is not None and FakeQuantState is not None:
            if fake_quant_state is None:
                fake_quant_state = FakeQuantState.VALIDATION
            set_fake_quantize(self.model, fake_quant_state)

    def set_device(self, device: torch.device) -> None:
        self.device = device
        self.model.to(device)

    def preprocess(self, data: Any) -> Any:
        for processor in self.preprocessors:
            data = processor(data)
        return data

    def postprocess(self, pred: Any, data: Any) -> Any:
        """Run post-processors and return prediction only.

        Note:
            This method keeps backward-compatible return behavior.
            Use ``postprocess_with_data`` when both ``pred`` and ``data`` are
            needed by downstream consumers.
        """
        pred, _ = self.postprocess_with_data(pred, data)
        return pred

    def postprocess_with_data(self, pred: Any, data: Any) -> Any:
        """Run post-processors and return both prediction and data."""
        for processor in self.postprocessors:
            pred, data = processor(pred, data)
        return pred, data

    def vispostprocess(self, pred: Any, data: Any) -> None:
        for processor in self.visualize_processors:
            processor(pred, data)

    def forward(self, data: Any) -> Any:
        with torch.no_grad():
            data = self.preprocess(data)
            if self.device.type == "cuda":
                data = to_cuda(data, self.device, non_blocking=True)

            output = self.model(data)
            output, data = self.postprocess_with_data(output, data)
            if self.visualize_processors:
                self.vispostprocess(output, data)
        return output
