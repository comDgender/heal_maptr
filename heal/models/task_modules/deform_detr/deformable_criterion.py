# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.distributed import dist_initialized
from torch.distributed import get_world_size

from heal.models.task_modules.detr import DetrCriterion

__all__ = ["SetCriterion", "DeformableCriterion", "sigmoid_focal_loss"]


def sigmoid_focal_loss(
    inputs,
    targets,
    num_boxes,
    alpha: float = 0.25,
    gamma: float = 2,
    mean_in_dim1: bool = True,
):
    """Binary focal loss with logits (RetinaNet / DETR-style).

    ``inputs`` / ``targets`` share shape; ``targets`` are soft labels in
    ``[0, 1]``. The result is summed over positives and divided by
    ``num_boxes``. When ``mean_in_dim1`` is set, features are averaged
    along dim 1 before the sum.
    """
    probs = torch.sigmoid(inputs)
    bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # p_t is the predicted probability of the true class.
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    focal = bce * torch.pow(1.0 - p_t, gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        focal = focal * alpha_t

    if mean_in_dim1:
        return focal.mean(dim=1).sum() / num_boxes
    return focal.sum() / num_boxes


class SetCriterion(DetrCriterion):
    """Hungarian set loss for DETR-style detectors (cls + boxes).

    Matches predictions to GT with ``matcher``, then supervises each
    matched pair. Classification may use CE or sigmoid focal loss.
    """

    def __init__(
        self,
        num_classes: int,
        matcher: nn.Module,
        weight_dict: Dict[str, float],
        losses: List[str] = None,
        eos_coef: float = 0.1,
        loss_class_type: str = "focal_loss",
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super(DetrCriterion, self).__init__()
        if loss_class_type not in ("ce_loss", "focal_loss"):
            raise AssertionError(
                "only support ce loss and focal loss for computing cls loss"
            )
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = ["class", "boxes"] if losses is None else losses
        self.eos_coef = eos_coef
        self.loss_class_type = loss_class_type
        self.alpha = alpha
        self.gamma = gamma

        if loss_class_type == "ce_loss":
            # Down-weight the no-object class for CE.
            cls_weight = torch.ones(num_classes + 1)
            cls_weight[-1] = eos_coef
            self.register_buffer("empty_weight", cls_weight)

    def loss_labels(
        self,
        outputs: dict,
        targets: dict,
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: torch.Tensor,
    ):
        """Classification loss (CE or binary focal) on matched queries."""
        assert "pred_logits" in outputs
        logits = outputs["pred_logits"]
        matched = self._get_src_permutation_idx(indices)
        matched_labels = torch.cat(
            [
                labels[tgt_i]
                for labels, (_, tgt_i) in zip(targets["gt_classes"], indices)
            ]
        )

        # Default every query to the no-object class, then fill matches.
        target_ids = torch.full(
            logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=logits.device,
        )
        target_ids[matched] = matched_labels

        if self.loss_class_type == "ce_loss":
            loss_class = F.cross_entropy(
                logits.transpose(1, 2), target_ids, self.empty_weight
            )
        else:
            # One-hot over foreground classes only (drop the no-object column).
            batch, n_queries, n_cls = logits.shape
            onehot = logits.new_zeros(batch, n_queries, n_cls + 1)
            onehot.scatter_(2, target_ids.unsqueeze(-1), 1.0)
            onehot = onehot[..., :-1]
            loss_class = (
                sigmoid_focal_loss(
                    logits,
                    onehot,
                    num_boxes=num_boxes,
                    alpha=self.alpha,
                    gamma=self.gamma,
                )
                * n_queries
            )

        return {"loss_class": loss_class}

    def get_loss(
        self,
        loss: str,
        outputs: Dict,
        targets: Dict,
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: torch.Tensor,
        **kwargs,
    ):
        loss_map = {
            "class": self.loss_labels,
            "boxes": self.loss_boxes,
        }
        if loss not in loss_map:
            raise AssertionError(f"do you really want to compute {loss} loss?")
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs: Dict, targets: Dict):
        main_out = {
            key: val for key, val in outputs.items() if key != "aux_outputs"
        }
        matched = self.matcher(main_out, targets)

        # Normalize by total GT count (synced across ranks when distributed).
        n_boxes = sum(labels.numel() for labels in targets["gt_classes"])
        n_boxes = torch.as_tensor(
            [n_boxes],
            dtype=torch.float,
            device=next(iter(outputs.values())).device,
        )
        if dist_initialized():
            torch.distributed.all_reduce(n_boxes)
        n_boxes = torch.clamp(n_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss_name in self.losses:
            losses.update(
                self.get_loss(loss_name, outputs, targets, matched, n_boxes)
            )

        # Auxiliary decoder layers reuse the same weight keys with a suffix.
        weights = copy.deepcopy(self.weight_dict)
        if "aux_outputs" in outputs:
            for aux_i, aux_out in enumerate(outputs["aux_outputs"]):
                matched = self.matcher(aux_out, targets)
                for loss_name in self.losses:
                    aux_loss = self.get_loss(
                        loss_name, aux_out, targets, matched, n_boxes
                    )
                    losses.update(
                        {f"{k}_{aux_i}": v for k, v in aux_loss.items()}
                    )
                weights.update(
                    {f"{k}_{aux_i}": v for k, v in self.weight_dict.items()}
                )

        for key in list(losses.keys()):
            if key in weights:
                losses[key] = losses[key] * weights[key]
        return losses

    def __repr__(self):
        indent = " " * 4
        lines = [
            f"Criterion {self.__class__.__name__}",
            f"{indent}matcher: {self.matcher!r}",
            f"{indent}losses: {self.losses}",
            f"{indent}loss_class_type: {self.loss_class_type}",
            f"{indent}weight_dict: {self.weight_dict}",
            f"{indent}num_classes: {self.num_classes}",
            f"{indent}eos_coef: {self.eos_coef}",
            f"{indent}focal loss alpha: {self.alpha}",
            f"{indent}focal loss gamma: {self.gamma}",
        ]
        return "\n".join(lines)


@OBJECT_REGISTRY.register
class DeformableCriterion(SetCriterion):
    """Computes loss for Deformable-DETR and two-stage Deformable-DETR models.

    Args:
        num_classes: Number of object categories,
                 excluding 'no-object' category.
        matcher: Module for matching targets with proposals.
        weight_dict: Loss component names and their weights.
        losses: Types of losses to apply, defaulting to 'class' and 'boxes'.
        eos_coef: End-of-sequence class coefficient in classification loss.
        loss_class_type: Classification loss type ('ce_loss' or 'focal_loss').
        alpha: Alpha parameter for Focal Loss.
        gamma: Gamma parameter for Focal Loss.
        aux_loss: Include auxiliary losses.
    """

    def __init__(
        self,
        num_classes: int,
        matcher: nn.Module,
        weight_dict: Dict[str, float],
        losses: List[str] = None,
        eos_coef: float = 0.1,
        loss_class_type: str = "focal_loss",
        alpha: float = 0.25,
        gamma: float = 2.0,
        aux_loss: bool = True,
    ):
        super(DeformableCriterion, self).__init__(
            num_classes=num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            losses=losses,
            eos_coef=eos_coef,
            loss_class_type=loss_class_type,
            alpha=alpha,
            gamma=gamma,
        )

    def forward(self, outputs: Dict, targets: Dict):
        main_out = {
            key: val
            for key, val in outputs.items()
            if key not in ("aux_outputs", "enc_outputs")
        }
        matched = self.matcher(main_out, targets)

        n_boxes = sum(labels.numel() for labels in targets["gt_classes"])
        n_boxes = torch.as_tensor(
            [n_boxes],
            dtype=torch.float,
            device=next(iter(outputs.values())).device,
        )
        if dist_initialized():
            torch.distributed.all_reduce(n_boxes)
            n_boxes = torch.clamp(n_boxes / get_world_size(), min=1).item()
        else:
            n_boxes = torch.clamp(n_boxes, min=1).item()

        losses = {}
        extra = {}
        for loss_name in self.losses:
            losses.update(
                self.get_loss(
                    loss_name, outputs, targets, matched, n_boxes, **extra
                )
            )

        weights = copy.deepcopy(self.weight_dict)
        if "aux_outputs" in outputs:
            for aux_i, aux_out in enumerate(outputs["aux_outputs"]):
                matched = self.matcher(aux_out, targets)
                for loss_name in self.losses:
                    aux_loss = self.get_loss(
                        loss_name,
                        aux_out,
                        targets,
                        matched,
                        n_boxes,
                        **extra,
                    )
                    losses.update(
                        {f"{k}_{aux_i}": v for k, v in aux_loss.items()}
                    )
                weights.update(
                    {f"{k}_{aux_i}": v for k, v in self.weight_dict.items()}
                )

        # Two-stage encoder proposals: force class labels to zero (binary).
        if "enc_outputs" in outputs:
            enc_out = outputs["enc_outputs"]
            enc_targets = copy.deepcopy(targets)
            for bi in range(len(enc_targets["gt_classes"])):
                enc_targets["gt_classes"][bi] = torch.zeros_like(
                    enc_targets["gt_classes"][bi]
                )
            matched = self.matcher(enc_out, enc_targets)
            for loss_name in self.losses:
                enc_loss = self.get_loss(
                    loss_name,
                    enc_out,
                    enc_targets,
                    matched,
                    n_boxes,
                    **extra,
                )
                losses.update({f"{k}_enc": v for k, v in enc_loss.items()})
            weights.update(
                {f"{k}_enc": v for k, v in self.weight_dict.items()}
            )

        # Only scale keys present in the base weight_dict (legacy behavior).
        for key in list(losses.keys()):
            if key in self.weight_dict:
                losses[key] = losses[key] * weights[key]

        return losses
