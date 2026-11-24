# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import Any, Dict, Optional, Tuple

import torch
from hat_engine.registry import OBJECT_REGISTRY
from horizon_plugin_pytorch.nn.quantized import FloatFunctional
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.quantization import (
    FixedScaleObserver,
    QuantStub,
    get_default_qat_qconfig,
)
from torch.quantization import DeQuantStub

from .blocks import QINT16_MAX
from .instance_bank import InstanceBank

__all__ = ["MemoryBank"]


@OBJECT_REGISTRY.register
class MemoryBank(InstanceBank):
    """
    Module for managing instance embeddings associated with anchors.

    Args:
        num_anchor : Number of anchors.
        embed_dims : Dimensionality of instance embeddings.
        anchor : Initial anchor points or path to anchor points.
        num_temp_instances : Number of temporal instances (default: 0).
        default_time_interval: Default time interval (default: 0.5).
        confidence_decay : Confidence decay rate (default: 0.6).
        anchor_grad : Whether anchor points are trainable
                      (default: True).
        max_time_interval : Maximum time interval (default: 2).
        projection_mat_key : Key for projection matrix
                             (default: "lidar2global").
        task_name: Task name. Default to None.
    """

    def __init__(
        self,
        num_memory_instances: int = 512,
        task_name: Optional[str] = None,
        **kwargs,
    ):
        super(MemoryBank, self).__init__(**kwargs)
        self.num_memory_instances = num_memory_instances
        self.prepare_quant_stub()
        self.task_name = task_name

    def prepare_quant_stub(self):
        self.anchor_quant_stub = QuantStub()
        self.instance_feature_quant_stub = QuantStub()
        self.temp_anchor_quant_stub = QuantStub()
        self.temp_instance_feature_quant_stub = QuantStub()
        self.feature_cat = FloatFunctional()
        self.anchor_cat = FloatFunctional()
        self.dequant = DeQuantStub()

    def update_temp(self):
        return False

    def get(
        self,
        batch_size: int,
        metas: Dict[str, Any],
        dn_metas: Dict[str, Any],
        compiler_model: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get instance features and anchors for the current batch.

        Args:
            batch_size: Batch size.
            metas: Metadata dictionary (default: None).
            dn_metas: Dense anchor metadata dictionary (default: None).
            compiler_model: Flag indicating if in compiler mode
                            (default: False).

        Returns:
            Tuple containing instance features and anchors.
        """
        instance_feature = torch.tile(
            self.instance_feature[None], (batch_size, 1, 1)
        )

        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))

        if compiler_model is True:
            if self.task_name is None:
                self.cached_anchor = metas["cached_anchor"]
                self.cached_feature = metas["cached_feature"]
            else:
                self.cached_anchor = metas[f"{self.task_name}_cached_anchor"]
                self.cached_feature = metas[f"{self.task_name}_cached_feature"]
        else:
            self.gen_cached_data(
                batch_size, metas, dn_metas, self.get_num_memory()
            )

            self.cached_feature = torch.where(
                self.mask[:, None, None],
                self.cached_feature,
                torch.zeros_like(self.cached_feature),
            )
            self.cached_anchor = torch.where(
                self.mask[:, None, None],
                self.cached_anchor,
                torch.zeros_like(self.cached_anchor),
            )
            if self.instance_id is not None:
                self.instance_id = torch.where(
                    self.mask[:, None],
                    self.instance_id,
                    self.instance_id.new_tensor(-1),
                )
        anchor = self.anchor_quant_stub(anchor)
        instance_feature = self.instance_feature_quant_stub(instance_feature)
        temp_anchor = self.anchor_quant_stub(self.cached_anchor)
        temp_feature = self.instance_feature_quant_stub(self.cached_feature)

        temp_anchor = torch.clamp(temp_anchor, min=-60, max=60)
        anchor = torch.clamp(anchor, min=-60, max=60)

        anchor = self.anchor_cat.cat(
            [temp_anchor[:, : self.get_num_temp()], anchor], dim=1
        )
        instance_feature = self.feature_cat.cat(
            [temp_feature[:, : self.get_num_temp()], instance_feature], dim=1
        )
        temp_anchor = temp_anchor[:, self.get_num_temp() :]
        temp_feature = temp_feature[:, self.get_num_temp() :]
        return (
            instance_feature,
            anchor,
            temp_feature,
            temp_anchor,
        )

    def get_num_memory(self) -> int:
        """Get number of temporal instances."""
        return self.num_memory_instances

    def set_qconfig(self) -> None:
        """Set the qconfig."""
        from hat_engine.utils import qconfig_manager
        from horizon_plugin_pytorch.dtype import qint16

        self.anchor_cat.qconfig = qconfig_manager.get_qconfig(
            activation_qat_qkwargs={"dtype": qint16, "averaging_constant": 0},
            activation_calibration_qkwargs={
                "dtype": qint16,
            },
            weight_qat_qkwargs={
                "averaging_constant": 1,
            },
        )
        self.anchor_quant_stub.qconfig = get_default_qat_qconfig(
            dtype="qint16",
            activation_qkwargs={
                "observer": FixedScaleObserver,
                "scale": 60 / QINT16_MAX,
            },
        )

    def cache(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        confidence: torch.Tensor,
        metas: Dict[str, Any],
        update_confidence: bool = False,
    ) -> None:
        if isinstance(instance_feature, QTensor):
            self.instance_feature_quant_stub.activation_post_process.reset_dtype(  # noqa
                instance_feature.dtype, False
            )
            self.instance_feature_quant_stub.activation_post_process.set_qparams(  # noqa
                instance_feature.q_scale()
            )
        anchor = self.dequant(anchor)
        instance_feature = self.dequant(instance_feature)
        num_temp_instances = self.get_num_temp()
        if update_confidence:
            (
                cached_confidence,
                cached_feature,
                cached_anchor,
                update_confidence,
            ) = InstanceBank.cache_instance(
                num_temp_instances,
                instance_feature,
                anchor,
                confidence,
                self.cached_confidence,
                self.confidence_decay,
                update_confidence,
            )
            self.temp_confidence = update_confidence
        else:
            (
                cached_confidence,
                cached_feature,
                cached_anchor,
            ) = InstanceBank.cache_instance(
                num_temp_instances,
                instance_feature,
                anchor,
                confidence,
                self.cached_confidence,
                self.confidence_decay,
            )
        self.metas = metas
        self.cached_confidence = torch.cat(
            [cached_confidence, self.cached_confidence], dim=1
        )[:, : self.get_num_memory()]
        self.cached_feature = torch.cat(
            [cached_feature, self.cached_feature], dim=1
        )[:, : self.get_num_memory()]
        self.cached_anchor = torch.cat(
            [cached_anchor, self.cached_anchor], dim=1
        )[:, : self.get_num_memory()]
