# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from hat_engine.registry import OBJECT_REGISTRY
from scipy.optimize import linear_sum_assignment

__all__ = ["SparseBEVTarget"]


@OBJECT_REGISTRY.register
class SparseBEVTarget(object):
    """
    Configuration and parameters for Sparse BEV Target generation.

    Args:
        cls_weight: Weight for classification loss. Defaults to 2.0.
        alpha: Focal loss alpha parameter. Defaults to 0.25.
        gamma: Focal loss gamma parameter. Defaults to 2.
        eps: Small epsilon value for numerical stability. Defaults to 1e-12.
        box_weight: Weight for regression loss. Defaults to 0.25.
        reg_weights: List of weights for each regression target.
                     Defaults to [1.0] * 8 + [0.0] * 2.
        cls_wise_reg_weights: List of class-wise regression weights.
                              Defaults to None.
        ignore_reg_weight: Weight for ignored regression targets.
                           Defaults to 0.0.
        ignore_cls_weight: Weight for ignored classification targets.
                           Defaults to 0.0.
        num_dn_groups: Number of groups for denoise sampling.
                       Defaults to 0.
        num_temp_dn_groups: Number of temporal groups for denosis sampling.
                            Defaults to 0.
        dn_noise_scale: Scale factor for denoise sampling .
                        Defaults to 0.5.
        max_dn_gt: Maximum number of ground truth samples for denoise sampling.
                   Defaults to 30.
        add_neg_dn: Whether to add negative denoise samples.
                    Defaults to True.
    """

    def __init__(
        self,
        cls_weight: float = 2.0,
        alpha: float = 0.25,
        gamma: float = 2,
        eps: float = 1e-12,
        box_weight: float = 0.25,
        reg_weights: List[float] = None,
        cls_wise_reg_weights: List[float] = None,
        ignore_reg_weight: float = 0.0,
        ignore_cls_weight: float = 0.0,
        num_dn_groups: int = 0,
        num_temp_dn_groups: int = 0,
        dn_noise_scale: float = 0.5,
        max_dn_gt: int = 30,
        add_neg_dn: bool = True,
    ):
        super(SparseBEVTarget, self).__init__()
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.reg_weights = reg_weights
        if self.reg_weights is None:
            self.reg_weights = [1.0] * 8 + [0.0] * 2
        self.cls_wise_reg_weights = cls_wise_reg_weights
        self.ignore_reg_weight = ignore_reg_weight
        self.ignore_cls_weight = ignore_cls_weight
        self.num_dn_groups = num_dn_groups
        self.num_temp_dn_groups = num_temp_dn_groups
        self.dn_noise_scale = dn_noise_scale
        self.max_dn_gt = max_dn_gt
        self.add_neg_dn = add_neg_dn
        self.dn_metas = None

    def encode_reg_target(
        self,
        box_target: List[torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> List[torch.Tensor]:
        """
        Encode regression targets into a standardized format.

        Args:
            box_target: List of tensors representing regression targets.
            device: Device to move the outputs to. Defaults to None.

        Returns:
            List of encoded regression targets.
        """

        outputs = []
        for box in box_target:
            if len(box) == 0:
                output = torch.zeros([1, 10])
            else:
                output = torch.cat(
                    [
                        box[..., 0:3],
                        box[..., 3:6].log(),
                        torch.sin(box[..., 6]).unsqueeze(-1),
                        torch.cos(box[..., 6]).unsqueeze(-1),
                        torch.where(
                            torch.isnan(box[..., 6 + 1 :]),
                            box.new_tensor(0),
                            box[..., 6 + 1 :],
                        ),
                    ],
                    dim=-1,
                )
            if device is not None:
                output = output.to(device=device)
            outputs.append(output)
        return outputs

    def __call__(
        self,
        cls_pred: torch.Tensor,
        box_pred: torch.Tensor,
        cls_target: torch.Tensor,
        box_target: List[torch.Tensor],
        gt_ignore: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor]:
        """
        Compute target components based on predicted and target values.

        Args:
            cls_pred: Predicted class scores, shape (bs, num_pred, num_cls).
            box_pred: Predicted box coordinates, shape (bs, num_pred, 4).
            cls_target: Target class labels, shape (bs, num_pred).
            box_target: List of target box coordinates tensors.
            gt_ignore: Ignored ground truth indicator, shape (bs, num_pred).

        Returns:
            Tuple of :
                - output_cls_target: Classification target.
                - output_box_target: Regression target.
                - output_cls_weights: Classification loss weight.
                - output_reg_weights: Regression loss weight.
        """

        bs, num_pred, num_cls = cls_pred.shape
        cls_cost = self._cls_cost(cls_pred, cls_target)

        box_target = self.encode_reg_target(box_target, box_pred.device)
        instance_reg_weights = []
        for bi, boxes in enumerate(box_target):
            w = (~boxes.isnan()).to(dtype=boxes.dtype)
            if self.cls_wise_reg_weights is not None:
                for cls_id, cls_w in self.cls_wise_reg_weights.items():
                    sel = (cls_target[bi] == cls_id)[:, None]
                    w = torch.where(sel, w.new_tensor(cls_w), w)
            instance_reg_weights.append(w)

        box_cost = self._box_cost(box_pred, box_target, instance_reg_weights)

        indices = []
        for bi in range(bs):
            cost_np = (cls_cost[bi] + box_cost[bi]).detach().cpu().numpy()
            cost_np = np.where(
                np.isneginf(cost_np) | np.isnan(cost_np), 1e8, cost_np
            )
            row_ind, col_ind = linear_sum_assignment(cost_np)
            indices.append(
                [
                    cls_pred.new_tensor(row_ind, dtype=torch.int64),
                    cls_pred.new_tensor(col_ind, dtype=torch.int64),
                ]
            )

        output_cls_target = cls_target[0].new_ones([bs, num_pred]) * num_cls
        output_box_target = box_pred.new_zeros(box_pred.shape)
        output_reg_weights = box_pred.new_zeros(box_pred.shape)
        ignore = box_pred.new_zeros([bs, num_pred])
        for bi, (pred_idx, target_idx) in enumerate(indices):
            if len(cls_target[bi]) == 0:
                continue
            output_cls_target[bi, pred_idx] = cls_target[bi][target_idx]
            output_box_target[bi, pred_idx] = box_target[bi][target_idx]
            output_reg_weights[bi, pred_idx] = instance_reg_weights[bi][
                target_idx
            ]
            if gt_ignore is not None and len(gt_ignore) != 0:
                ignore[bi, pred_idx] = gt_ignore[bi][target_idx].to(
                    dtype=ignore.dtype
                )

        output_reg_weights = torch.where(
            ignore[..., None] == 0,
            output_reg_weights,
            output_reg_weights * self.ignore_reg_weight,
        )
        output_cls_weights = ignore.new_ones(ignore.shape)
        output_cls_weights = torch.where(
            ignore == 0,
            output_cls_weights,
            output_cls_weights * self.ignore_cls_weight,
        )
        return (
            output_cls_target,
            output_box_target,
            output_cls_weights,
            output_reg_weights,
        )

    def _cls_cost(
        self, cls_pred: torch.Tensor, cls_target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute classification cost based on predicted and target class scores.

        Args:
            cls_pred: Predicted class scores, shape (bs, num_pred, num_cls).
            cls_target: Target class labels, shape (bs, num_pred).

        Returns:
            Classification cost.
        """
        bs = cls_pred.shape[0]
        cls_pred = cls_pred.sigmoid()
        cost = []
        for i in range(bs):
            neg_cost = (
                -(1 - cls_pred[i] + self.eps).log()
                * (1 - self.alpha)
                * cls_pred[i].pow(self.gamma)
            )
            pos_cost = (
                -(cls_pred[i] + self.eps).log()
                * self.alpha
                * (1 - cls_pred[i]).pow(self.gamma)
            )
            cost.append(
                (pos_cost[:, cls_target[i]] - neg_cost[:, cls_target[i]])
                * self.cls_weight
            )
        return cost

    def _box_cost(
        self,
        box_pred: torch.Tensor,
        box_target: List[torch.Tensor],
        instance_reg_weights: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute regression cost based on predicted and target box coordinates.

        Args:
            box_pred : Predicted box coordinates, shape (bs, num_pred, 4).
            box_target: List of target box coordinates tensors.
            instance_reg_weights: List of instance-wise regression weights.

        Returns:
            reg_cost : Regression cost.
        """
        bs = box_pred.shape[0]
        cost = []
        for i in range(bs):
            cost.append(
                torch.sum(
                    torch.abs(box_pred[i, :, None] - box_target[i][None])
                    * instance_reg_weights[i][None]
                    * box_pred.new_tensor(self.reg_weights),
                    dim=-1,
                )
                * self.box_weight
            )
        return cost

    def get_dn_anchors(
        self,
        cls_target: List[torch.Tensor],
        box_target: List[torch.Tensor],
        gt_instance_id: Optional[List[torch.Tensor]] = None,
    ) -> Optional[torch.Tensor]:
        """
        Generate denosis anchor for training based on ground truth targets.

        Args:
            cls_target : List of tensors containing target class labels.
            box_target : List of tensors containing target box coordinates.
            gt_instance_id : List of tensors containing ground truth
                             instance IDs.Defaults to None.

        Returns:
            Tensor containing denosis anchor points or None
            if num_dn_groups <= 0.
        """

        if self.num_dn_groups <= 0:
            return None

        if self.num_temp_dn_groups <= 0:
            gt_instance_id = None
        # put all non-ignore gt to front and delete the gt out of max_dn_gt
        if self.max_dn_gt > 0:
            cls_target = [x[: self.max_dn_gt] for x in cls_target]
            box_target = [x[: self.max_dn_gt] for x in box_target]
            if gt_instance_id is not None:
                gt_instance_id = [x[: self.max_dn_gt] for x in gt_instance_id]

        max_dn_gt = max([len(x) for x in cls_target])

        cls_target = torch.stack(
            [
                F.pad(x, (0, max_dn_gt - x.shape[0]), value=-1)
                for x in cls_target
            ]
        )
        box_target = torch.stack(
            self.encode_reg_target(
                [
                    F.pad(x, (0, 0, 0, max_dn_gt - x.shape[0]))
                    for x in box_target
                ],
                device=cls_target.device,
            )
        )
        # Zero-out padded GT slots before building DN queries.
        pad_slots = cls_target[..., None] == -1
        box_target = torch.where(
            pad_slots, box_target.new_tensor(0), box_target
        )
        if gt_instance_id is not None:
            gt_instance_id = torch.stack(
                [
                    F.pad(x, (0, max_dn_gt - x.shape[0]), value=-1)
                    for x in gt_instance_id
                ]
            )

        bs, num_gt, state_dims = box_target.shape
        n_groups = self.num_dn_groups

        if n_groups > 1:
            cls_target = cls_target.tile(n_groups, 1)
            box_target = box_target.tile(n_groups, 1, 1)
            if gt_instance_id is not None:
                gt_instance_id = gt_instance_id.tile(n_groups, 1)

        scale = box_target.new_tensor(self.dn_noise_scale)
        noise = (torch.rand_like(box_target) * 2 - 1) * scale
        dn_anchor = box_target + noise
        if self.add_neg_dn:
            # Push negatives outside the unit noise ball, then randomize sign.
            noise_neg = (torch.rand_like(box_target) + 1.0) * scale
            flip = torch.rand_like(box_target) > 0.5
            noise_neg = torch.where(flip, noise_neg, -noise_neg)
            dn_anchor = torch.cat([dn_anchor, box_target + noise_neg], dim=1)
            num_gt = num_gt * 2

        weights = torch.ones_like(box_target)
        box_cost = self._box_cost(dn_anchor, box_target, weights)
        dn_box_target = dn_anchor.new_zeros(dn_anchor.shape)
        dn_cls_target = torch.full_like(cls_target, -3)

        dn_id_target = (
            None
            if gt_instance_id is None
            else torch.full_like(gt_instance_id, -1)
        )
        if self.add_neg_dn:
            dn_cls_target = torch.cat([dn_cls_target, dn_cls_target], dim=1)
            if dn_id_target is not None:
                dn_id_target = torch.cat([dn_id_target, dn_id_target], dim=1)

        for bi in range(dn_anchor.shape[0]):
            rows, cols = linear_sum_assignment(box_cost[bi].cpu().numpy())
            rows = dn_anchor.new_tensor(rows, dtype=torch.int64)
            cols = dn_anchor.new_tensor(cols, dtype=torch.int64)
            dn_box_target[bi, rows] = box_target[bi, cols]
            dn_cls_target[bi, rows] = cls_target[bi, cols]
            if dn_id_target is not None:
                dn_id_target[bi, rows] = gt_instance_id[bi, cols]

        def _pack_groups(t: torch.Tensor, has_state: bool) -> torch.Tensor:
            if has_state:
                return (
                    t.reshape(n_groups, bs, num_gt, state_dims)
                    .permute(1, 0, 2, 3)
                    .flatten(1, 2)
                )
            return t.reshape(n_groups, bs, num_gt).permute(1, 0, 2).flatten(1)

        dn_anchor = _pack_groups(dn_anchor, True)
        dn_box_target = _pack_groups(dn_box_target, True)
        dn_cls_target = _pack_groups(dn_cls_target, False)
        if dn_id_target is not None:
            dn_id_target = _pack_groups(dn_id_target, False)

        valid_mask = dn_cls_target >= 0
        if self.add_neg_dn:
            cls_full = _pack_groups(
                torch.cat([cls_target, cls_target], dim=1), False
            )
            neg_keep = (cls_full >= 0) & (dn_cls_target == -3)
            valid_mask = valid_mask | neg_keep

        group_len = num_gt * n_groups
        attn_mask = dn_box_target.new_ones(group_len, group_len)
        for gi in range(n_groups):
            lo = num_gt * gi
            hi = lo + num_gt
            attn_mask[lo:hi, lo:hi] = 0
        attn_mask = attn_mask == 1
        return (
            dn_anchor,
            dn_box_target,
            dn_cls_target,
            attn_mask,
            valid_mask,
            dn_id_target,
        )

    def update_dn(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        dn_reg_target: torch.Tensor,
        dn_cls_target: torch.Tensor,
        valid_mask: torch.Tensor,
        dn_id_target: torch.Tensor,
        num_noraml_anchor: torch.Tensor,
        temporal_valid_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Update Denosis anchors with temporal denosis anchor.

        Args:
            instance_feature: Instance features, shape (bs, num_anchor, ...)
            anchor: Anchor points, shape (bs, num_anchor, ...)
            dn_reg_target: DN regression targets,
                           shape (bs, num_anchor, ...)
            dn_cls_target: DN class targets, shape (bs, num_anchor)
            valid_mask: Valid mask indicating active DN instances,
                        shape (bs, num_anchor)
            dn_id_target: Optional DN instance IDs,
                          shape (bs, num_anchor)
            num_normal_anchor: Number of normal anchors (non-DN) in each batch.
            temporal_valid_mask: Optional temporal valid mask.

        Returns:
            Updated instance_feature, anchor, dn_reg_target, dn_cls_target,
            valid_mask, dn_id_target
        """
        bs, num_anchor = instance_feature.shape[:2]
        if temporal_valid_mask is None:
            self.dn_metas = None
        if self.dn_metas is None or num_noraml_anchor >= num_anchor:
            return (
                instance_feature,
                anchor,
                dn_reg_target,
                dn_cls_target,
                valid_mask,
                dn_id_target,
            )

        n_normal = num_noraml_anchor
        n_dn_total = num_anchor - n_normal
        # Trailing tokens are DN; keep the leading normal queries.
        normal_feat = instance_feature[:, :n_normal]
        normal_anc = anchor[:, :n_normal]
        dn_feat_flat = instance_feature[:, n_normal:]
        dn_anc_flat = anchor[:, n_normal:]

        n_groups = self.num_dn_groups
        n_dn = n_dn_total // n_groups
        cur = [
            dn_feat_flat.reshape(bs, n_groups, n_dn, -1),
            dn_anc_flat.reshape(bs, n_groups, n_dn, -1),
            dn_reg_target.reshape(bs, n_groups, n_dn, -1),
            dn_cls_target.reshape(bs, n_groups, n_dn),
            valid_mask.reshape(bs, n_groups, n_dn),
            (
                None
                if dn_id_target is None
                else dn_id_target.reshape(bs, n_groups, n_dn)
            ),
        ]

        temp_feat = self.dn_metas["dn_instance_feature"]
        _, n_temp_groups, n_temp_dn = temp_feat.shape[:3]
        temp_id = self.dn_metas["dn_id_target"]
        hit = temp_id[..., None] == cur[5][:, :n_temp_groups, None]
        temp_reg = (hit[..., None] * cur[2][:, :n_temp_groups, None]).sum(
            dim=3
        )
        miss = ~hit.any(dim=-1)
        temp_cls = torch.where(
            miss,
            self.dn_metas["dn_cls_target"].new_tensor(-1),
            self.dn_metas["dn_cls_target"],
        )
        prev = [
            temp_feat,
            self.dn_metas["dn_anchor"],
            temp_reg,
            temp_cls,
            self.dn_metas["valid_mask"],
            temp_id,
        ]

        merged = []
        for prev_t, cur_t in zip(prev, cur):
            if n_temp_dn < n_dn:
                pad_size = (0, n_dn - n_temp_dn)
                if prev_t.dim() == 4:
                    pad_size = (0, 0) + pad_size
                else:
                    assert prev_t.dim() == 3
                prev_t = F.pad(prev_t, pad_size, value=0)
            else:
                prev_t = prev_t[:, :, :n_dn]
            gate = temporal_valid_mask[:, None, None]
            if cur_t.dim() == 4:
                gate = gate.unsqueeze(-1)
            chosen = torch.where(gate, prev_t, cur_t[:, :n_temp_groups])
            packed = torch.cat([chosen, cur_t[:, n_temp_groups:]], dim=1)
            merged.append(packed.flatten(1, 2))

        merged[0] = torch.cat([normal_feat, merged[0]], dim=1)
        merged[1] = torch.cat([normal_anc, merged[1]], dim=1)
        return merged

    def cache_dn(
        self,
        dn_instance_feature: torch.Tensor,
        dn_anchor: torch.Tensor,
        dn_cls_target: torch.Tensor,
        valid_mask: torch.Tensor,
        dn_id_target: torch.Tensor,
    ) -> None:
        """
        Cache Denosis features and anchors for future use.

        Args:
            dn_instance_feature: DN instance features,
                                 shape (bs, num_dn, ...)
            dn_anchor: DN anchor points, shape (bs, num_dn, ...)
            dn_cls_target: DN class targets, shape (bs, num_dn)
            valid_mask: Valid mask indicating active DN instances,
                        shape (bs, num_dn)
            dn_id_target: DN instance IDs, shape (bs, num_dn)
        """
        if self.num_temp_dn_groups <= 0 or dn_anchor.shape[1] == 0:
            return
        n_groups = self.num_dn_groups
        bs, num_dn = dn_instance_feature.shape[:2]
        n_per_group = num_dn // n_groups
        keep = torch.randperm(n_groups) < self.num_temp_dn_groups
        keep = keep.to(device=dn_anchor.device)

        dn_instance_feature = dn_instance_feature.detach().reshape(
            bs, n_groups, n_per_group, -1
        )[:, keep]
        dn_anchor = dn_anchor.detach().reshape(bs, n_groups, n_per_group, -1)[
            :, keep
        ]
        dn_cls_target = dn_cls_target.reshape(bs, n_groups, n_per_group)[
            :, keep
        ]

        valid_mask = valid_mask.reshape(bs, n_groups, n_per_group)[:, keep]
        if dn_id_target is not None:
            dn_id_target = dn_id_target.reshape(bs, n_groups, n_per_group)[
                :, keep
            ]
        self.dn_metas = {
            "dn_instance_feature": dn_instance_feature,
            "dn_anchor": dn_anchor,
            "dn_cls_target": dn_cls_target,
            "valid_mask": valid_mask,
            "dn_id_target": dn_id_target,
        }
