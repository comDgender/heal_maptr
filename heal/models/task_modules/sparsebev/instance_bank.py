# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import horizon_plugin_pytorch.nn as hnn
import numpy as np
import torch
import torch.nn.functional as F
from hat_engine.registry import OBJECT_REGISTRY
from horizon_plugin_pytorch.nn.quantized import FloatFunctional
from horizon_plugin_pytorch.qtensor import QTensor
from horizon_plugin_pytorch.quantization import QuantStub
from torch import nn
from torch.quantization import DeQuantStub

__all__ = ["InstanceBank"]


def topk(confidence, k, *inputs):
    """
    Perform top-k selection on confidence and gather corresponding inputs.

    Args:
        confidence: Confidence scores, shape (bs, N)
                    where bs is batch size and N is the number of items.
        k : Number of top-k elements to select.
        *inputs : Additional tensors to gather based on the indices of
                  top-k elements.

    Returns:
        A tuple containing:
            - confidence: Top-k confidence scores, shape (bs, k).
            - outputs : List of gathered tensors,
                        each of shape (bs, k, -1).

    """
    bs, N = confidence.shape[:2]
    confidence, indices = torch.topk(confidence, k, dim=1)
    indices = (
        indices + torch.arange(bs, device=indices.device)[:, None] * N
    ).reshape(-1)
    outputs = []
    for input in inputs:
        outputs.append(input.flatten(end_dim=1)[indices].reshape(bs, k, -1))
    return confidence, outputs


@OBJECT_REGISTRY.register
class InstanceBank(nn.Module):
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
        pc_range: Range of the point cloud.
    """

    def __init__(
        self,
        num_anchor: int,
        embed_dims: int,
        anchor: Union[str, list, tuple, np.ndarray],
        anchor_dims: int = 11,
        num_temp_instances: int = 0,
        default_time_interval: float = 0.5,
        confidence_decay: float = 0.6,
        anchor_grad: bool = True,
        max_time_interval: int = 2,
        projection_mat_key: str = "lidar2global",
        feat_grad: bool = False,
        pc_range: Optional[Sequence[float]] = None,
    ):
        super(InstanceBank, self).__init__()
        self.embed_dims = embed_dims
        self.num_temp_instances = num_temp_instances
        self.default_time_interval = default_time_interval
        self.confidence_decay = confidence_decay
        self.max_time_interval = max_time_interval
        self.projection_mat_key = projection_mat_key
        self.anchor_dims = anchor_dims

        if isinstance(anchor, str):
            anchor = np.load(anchor)
        elif isinstance(anchor, (list, tuple, np.ndarray)):
            anchor = np.array(anchor)
        elif anchor is None:
            if pc_range is None:
                anchor = np.zeros((num_anchor, self.anchor_dims))
            else:
                min_x, max_x = pc_range[0], pc_range[3]
                min_y, max_y = pc_range[1], pc_range[4]
                min_z, max_z = pc_range[2], pc_range[5]
                anchor = np.zeros((num_anchor, self.anchor_dims))
                anchor[:, :1] = np.random.uniform(
                    min_x, max_x, size=(num_anchor, 1)
                )
                anchor[:, 1:2] = np.random.uniform(
                    min_y, max_y, size=(num_anchor, 1)
                )
                anchor[:, 2:3] = np.random.uniform(
                    min_z, max_z, size=(num_anchor, 1)
                )
        if len(anchor.shape) == 3:  # for map
            anchor = anchor.reshape(anchor.shape[0], -1)
        self.num_anchor = min(len(anchor), num_anchor)
        anchor = anchor[:num_anchor]
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32),
            requires_grad=anchor_grad,
        )
        self.anchor_init = anchor
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]),
            requires_grad=feat_grad,
        )
        self.reset()

        self.prepare_quant_stub()

    def prepare_quant_stub(self):
        self.feature_cat = FloatFunctional()
        self.anchor_cat = FloatFunctional()

        self.feat_where = hnn.Where()
        self.anchor_where = hnn.Where()

        self.cls_quant_stub = QuantStub()
        self.anchor_quant_stub = QuantStub()
        self.instance_feature_quant_stub = QuantStub()
        self.mask_feature_quant_stub = QuantStub()
        self.dequant = DeQuantStub()

    def update_temp(self):
        return True

    def reset(self) -> None:
        """Reset instance features to initial values."""
        self.cached_feature = None
        self.cached_anchor = None
        self.cached_confidence = None
        self.mask = None
        self.metas = None
        self.temp_confidence = None
        self.instance_id = None
        self.prev_id = 0

    def init_weight(self) -> None:
        """Initialize anchor weights with the initial anchor points."""
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)

    @staticmethod
    def anchor_projection(
        anchor: torch.Tensor,
        T_src2dst: torch.Tensor,
        time_interval: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Project anchors to a new coordinate frame.

        Args:
            anchor : Anchor points.
            T_src2dst : Transformation matrix from source
                        to destination frame.
            time_interval : Time interval for velocity adjustment
                            (default: None).

        Returns:
            Projected anchor points in the destination frame.
        """
        dst_anchor = anchor.clone()
        vel = anchor[..., 8:]
        T_src2dst = torch.unsqueeze(T_src2dst, dim=1)
        center = dst_anchor[..., 0:3]
        if time_interval is not None:
            translation = vel.transpose(0, -1) * time_interval.to(
                dtype=vel.dtype
            )
            translation = translation.transpose(0, -1)
            center = center + translation
        dst_anchor[..., 0:3] = (
            torch.matmul(T_src2dst[..., :3, :3], center[..., None]).squeeze(
                dim=-1
            )
            + T_src2dst[..., :3, 3]
        )
        dst_anchor[..., [7, 6]] = torch.matmul(
            T_src2dst[..., :2, :2],
            dst_anchor[..., [7, 6], None],
        ).squeeze(-1)

        dst_anchor[..., 8:] = torch.matmul(
            T_src2dst[..., :3, :3], vel[..., None]
        ).squeeze(-1)
        return dst_anchor

    @staticmethod
    def gen_temp2cur(
        projection_mat_key: str,
        his_metas: Dict[str, Any],
        metas: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Generate transformation matrix from temporal to current frame.

        Args:
            projection_mat_key : Key for projection matrix in metadata.
            his_metas : Metadata from historical frame.
            metas : Metadata from current frame.

        Returns:
            Transformation matrix from temporal to current frame.
        """
        projection_mat = his_metas[projection_mat_key]
        src_projection_mat = projection_mat.cpu().numpy()
        dst_projection_mat = metas[projection_mat_key].cpu().numpy()
        temp2cur_numpy = np.linalg.inv(dst_projection_mat) @ src_projection_mat
        T_temp2cur = torch.tensor(temp2cur_numpy).float()
        return T_temp2cur

    def gen_cached_data(
        self,
        batch_size: int,
        metas: Dict[str, Any],
        dn_metas: Dict[str, Any],
        num_cached_instances: int,
    ) -> None:
        """
        Generate cached data.

        Args:
            batch_size: Batch size.
            metas: Metadata dictionary (default: None).
            dn_metas: Dense anchor metadata dictionary (default: None).
            num_cached_instances: Number of cached instances.
        """
        if (
            self.metas is None
            or self.cached_anchor is None
            or batch_size != self.cached_anchor.shape[0]
        ):
            self.reset()
            self.cached_feature = torch.zeros(
                (batch_size, num_cached_instances, self.embed_dims)
            ).to(device=self.instance_feature.device)
            self.cached_anchor = torch.zeros(
                (batch_size, num_cached_instances, self.anchor_dims)
            ).to(device=self.instance_feature.device)
            self.mask = (
                torch.zeros((batch_size))
                .bool()
                .to(device=self.instance_feature.device)
            )
            self.cached_confidence = torch.zeros(
                (batch_size, num_cached_instances)
            ).to(device=self.instance_feature.device)

        else:
            history_time = self.metas["timestamp"]
            time_interval = metas["timestamp"] - history_time
            # time_interval = time_interval.to(dtype=self.instance_feature.dtype)  # noqa
            time_interval = time_interval.to(self.instance_feature.device)
            threshold = torch.as_tensor(
                self.max_time_interval,
                dtype=time_interval.dtype,
                device=time_interval.device,
            )
            self.mask = torch.abs(time_interval) <= threshold
            if "scene" in metas.keys():
                for i in range(len(metas["scene"])):
                    if metas["scene"][i] != self.metas["scene"][i]:
                        self.mask[i] = 0
            with torch.no_grad():
                T_temp2cur = InstanceBank.gen_temp2cur(
                    self.projection_mat_key, self.metas, metas
                ).to(device=self.instance_feature.device)
                self.cached_anchor = InstanceBank.anchor_projection(
                    self.cached_anchor,
                    T_temp2cur,
                    time_interval,
                )
                if (
                    dn_metas is not None
                    and batch_size == dn_metas["dn_anchor"].shape[0]
                ):
                    dn_anchor = dn_metas["dn_anchor"]
                    num_dn_group, num_dn = dn_anchor.shape[1:3]
                    dn_anchor = InstanceBank.anchor_projection(
                        dn_anchor.flatten(1, 2),
                        T_temp2cur,
                        time_interval=time_interval,
                    )
                    dn_metas["dn_anchor"] = dn_anchor.reshape(
                        batch_size, num_dn_group, num_dn, -1
                    )

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

        if self.get_num_temp() <= 0:
            return instance_feature, anchor, None, None
        if compiler_model is True:
            self.cached_anchor = metas["cached_anchor"]
            self.cached_feature = metas["cached_feature"]
            self.mask = metas["mask"]
        else:
            self.gen_cached_data(
                batch_size, metas, dn_metas, self.get_num_temp()
            )

        anchor = self.anchor_quant_stub(anchor)
        instance_feature = self.instance_feature_quant_stub(instance_feature)
        temp_anchor = self.anchor_quant_stub(self.cached_anchor)
        temp_feature = self.instance_feature_quant_stub(self.cached_feature)

        return (
            instance_feature,
            anchor,
            temp_feature,
            temp_anchor,
        )

    def get_num_temp(self) -> int:
        """Get number of temporal instances."""
        return self.num_temp_instances

    def update(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        confidence: torch.Tensor,
        temp_instance_feature: torch.Tensor,
        temp_anchor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update instance features and anchors based on the given inputs.

        Args:
            instance_feature : Instance features tensor.
            anchor : Anchor tensor.
            confidence : Confidence scores tensor.
            temp_instance_feature : Temporary instance features tensor.
            temp_anchor: Temporary anchor tensor.
        """
        n_temp = self.get_num_temp()
        n_dn = 0
        dn_feat = None
        dn_anc = None
        if instance_feature.shape[1] > self.num_anchor:
            n_dn = instance_feature.shape[1] - self.num_anchor
            dn_feat = instance_feature[:, -n_dn:]
            dn_anc = anchor[:, -n_dn:]
            instance_feature = instance_feature[:, : self.num_anchor]
            anchor = anchor[:, : self.num_anchor]
            confidence = confidence[:, : self.num_anchor]

        n_keep = self.num_anchor - n_temp
        score = confidence.max(dim=-1)[0]
        _, (picked_feat, picked_anc) = topk(
            score, n_keep, instance_feature, anchor
        )
        merged_feat = self.feature_cat.cat(
            [temp_instance_feature, picked_feat], dim=1
        )
        merged_anc = self.anchor_cat.cat([temp_anchor, picked_anc], dim=1)
        cond = self.mask[:, None, None]
        instance_feature = self.feat_where(cond, merged_feat, instance_feature)
        anchor = self.anchor_where(cond, merged_anc, anchor)

        if self.instance_id is not None:
            self.instance_id = torch.where(
                self.mask[:, None],
                self.instance_id,
                self.instance_id.new_tensor(-1),
            )
        if n_dn > 0:
            instance_feature = torch.cat([instance_feature, dn_feat], dim=1)
            anchor = torch.cat([anchor, dn_anc], dim=1)

        return instance_feature, anchor

    @staticmethod
    def cache_instance(
        num_temp_instances: int,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        confidence: torch.Tensor,
        cached_confidence,
        confidence_decay,
        return_confidence: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        instance_feature = instance_feature.detach()
        anchor = anchor.detach()
        confidence = confidence.detach()

        confidence = confidence.max(dim=-1)[0]
        confidence = confidence.sigmoid()
        confidence[:, :num_temp_instances] = torch.maximum(
            cached_confidence[:, :num_temp_instances] * confidence_decay,
            confidence[:, :num_temp_instances],
        )

        (
            cached_confidence,
            (cached_feature, cached_anchor),
        ) = topk(confidence, num_temp_instances, instance_feature, anchor)
        if return_confidence:
            return cached_confidence, cached_feature, cached_anchor, confidence
        else:
            return cached_confidence, cached_feature, cached_anchor

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
        self.cached_confidence = cached_confidence
        self.cached_feature = cached_feature
        self.cached_anchor = cached_anchor

    def get_instance_id(
        self, confidence: torch.Tensor, anchor: torch.Tensor, threshold: float
    ) -> torch.Tensor:
        score = confidence.max(dim=-1).values.sigmoid()
        instance_id = score.new_full(score.shape, -1).long()

        if (
            self.instance_id is not None
            and self.instance_id.shape[0] == instance_id.shape[0]
        ):
            n_prev = self.instance_id.shape[1]
            instance_id[:, :n_prev] = self.instance_id

        need_id = instance_id < 0
        if threshold is not None:
            need_id = need_id & (score >= threshold)
        n_new = int(need_id.sum().item())
        fresh = torch.arange(n_new, device=instance_id.device) + self.prev_id
        fresh = fresh.to(dtype=instance_id.dtype)
        instance_id[torch.where(need_id)] = fresh
        self.prev_id += n_new
        self.update_instance_id(instance_id, score)
        return instance_id

    def update_instance_id(
        self,
        instance_id: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None,
    ) -> None:
        if self.temp_confidence is None:
            if confidence.dim() == 3:
                temp_conf = confidence.max(dim=-1).values
            else:
                temp_conf = confidence
        else:
            temp_conf = self.temp_confidence
        kept_ids = topk(temp_conf, self.num_temp_instances, instance_id)[1][0]
        kept_ids = kept_ids.squeeze(dim=-1)
        pad_w = self.num_anchor - self.num_temp_instances
        self.instance_id = F.pad(kept_ids, (0, pad_w), value=-1)
