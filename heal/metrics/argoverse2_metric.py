# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501


import math
from typing import Optional, Tuple, Union

import numpy as np
import torch
from hat_engine.metrics.metric import EvalMetric
from hat_engine.registry import OBJECT_REGISTRY

__all__ = ["BrierMetric", "MinADE", "MinFDE", "MissRate", "HitRate"]


def wrap_angle(
    angle: Union[torch.Tensor, np.ndarray],
    min_val: float = -math.pi,
    max_val: float = math.pi,
) -> torch.Tensor:
    """Convert angels into [min_val, max_val].

    Args:
        angle: The angle vector.
        min_val: The minimum value of angles.
        max_val: The maximum value of angles.

    Returns:
        angles: Converted angles.
    """
    return min_val + (angle + max_val) % (max_val - min_val)


def topk(
    max_guesses: int,
    pred: torch.Tensor,
    prob: Optional[torch.Tensor] = None,
    ptr: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pick the top-k modes by probability, or truncate when prob is absent."""
    del ptr  # retained for API compatibility; unused in this path
    num_modes = pred.size(1)
    k = min(max_guesses, num_modes)
    batch_size = pred.size(0)

    def _uniform_prob() -> torch.Tensor:
        return pred.new_ones((batch_size, k)) / float(k)

    # Requested k covers every mode: just (re)normalize probabilities.
    if k == num_modes:
        if prob is None:
            return pred, _uniform_prob()
        return pred, prob / prob.sum(dim=-1, keepdim=True)

    # Fewer modes than available.
    if prob is None:
        return pred[:, :k], _uniform_prob()

    _, mode_indices = torch.topk(prob, k=k, dim=-1, largest=True, sorted=True)
    batch_idx = (
        torch.arange(batch_size, device=pred.device)
        .unsqueeze(-1)
        .expand(-1, k)
    )
    pred_topk = pred[batch_idx, mode_indices]
    prob_topk = prob[batch_idx, mode_indices]
    prob_topk = prob_topk / prob_topk.sum(dim=-1, keepdim=True)
    return pred_topk, prob_topk


def valid_filter(
    pred: torch.Tensor,
    target: torch.Tensor,
    prob: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    keep_invalid_final_step: bool = True,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    """Keep only samples that have usable timesteps for metric evaluation."""
    if valid_mask is None:
        valid_mask = torch.ones(
            target.shape[:-1], dtype=torch.bool, device=target.device
        )

    # True: keep samples with any valid step.
    # False: require the final timestep to be valid.
    if keep_invalid_final_step:
        sample_keep = valid_mask.any(dim=-1)
    else:
        sample_keep = valid_mask[:, -1].to(dtype=torch.bool)

    pred = pred[sample_keep]
    target = target[sample_keep]
    if prob is not None:
        prob = prob[sample_keep]
    valid_mask = valid_mask[sample_keep]
    # Single contiguous graph pointer over the filtered batch: [0, N].
    ptr = target.new_zeros(2, dtype=torch.long)
    ptr[1] = target.size(0)
    return pred, target, prob, valid_mask, ptr


@OBJECT_REGISTRY.register
class BrierMetric(EvalMetric):
    """
    Brier metric for measuring the accuracy of probabilistic predictions.

    The Brier score measures the mean squared difference between predicted
    probabilities and the actual outcomes.
    Lower Brier scores indicate better predictive accuracy.

    Args:
        max_guesses: Maximum number of guesses to consider for
                top-k predictions.
        name: Name of the metric.
    """

    def __init__(
        self,
        max_guesses: int = 6,
        name: str = "Brier",
    ) -> None:
        super(BrierMetric, self).__init__(name)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.max_guesses = max_guesses

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        prob: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        keep_invalid_final_step: bool = True,
        min_criterion: str = "FDE",
    ) -> None:
        pred, target, prob, valid_mask, _ = valid_filter(
            pred, target, prob, valid_mask, keep_invalid_final_step
        )
        pred_topk, prob_topk = topk(self.max_guesses, pred, prob)
        if min_criterion == "FDE":
            inds_last = (
                valid_mask
                * torch.arange(1, valid_mask.size(-1) + 1, device=pred.device)
            ).argmax(dim=-1)
            inds_best = torch.norm(
                pred_topk[torch.arange(pred.size(0)), :, inds_last]
                - target[torch.arange(pred.size(0)), inds_last].unsqueeze(-2),
                p=2,
                dim=-1,
            ).argmin(dim=-1)
        elif min_criterion == "ADE":
            inds_best = (
                (
                    torch.norm(pred_topk - target.unsqueeze(1), p=2, dim=-1)
                    * valid_mask.unsqueeze(1)
                )
                .sum(dim=-1)
                .argmin(dim=-1)
            )
        else:
            raise ValueError(
                "{} is not a valid criterion".format(min_criterion)
            )
        self.sum += (
            (1.0 - prob_topk[torch.arange(pred.size(0)), inds_best])
            .pow(2)
            .sum()
        )
        self.count += pred.size(0)

    def compute(self) -> torch.Tensor:
        return self.sum / self.count


@OBJECT_REGISTRY.register
class MinADE(EvalMetric):
    """
    Minimum Average Displacement Error (minADE) for trajectory predictions.

    The minADE calculates the average displacement error of the best
    prediction out of a set of predictions.

    Args:
        max_guesses: Maximum number of guesses to consider for
                top-k predictions.
        name: Name of the metric.

    """

    def __init__(self, max_guesses: int = 6, name="minADE") -> None:
        super(MinADE, self).__init__(name)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.max_guesses = max_guesses

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        prob: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        keep_invalid_final_step: bool = True,
        min_criterion: str = "FDE",
    ) -> None:
        pred, target, prob, valid_mask, _ = valid_filter(
            pred, target, prob, valid_mask, keep_invalid_final_step
        )
        pred_topk, _ = topk(self.max_guesses, pred, prob)

        num_samples = pred.size(0)
        batch_idx = torch.arange(num_samples, device=pred.device)
        valid_counts = valid_mask.sum(dim=-1).to(pred.dtype)

        if min_criterion == "FDE":
            # Select the mode with the smallest final-step displacement, then
            # average its per-step L2 error over valid timesteps.
            step_weight = torch.arange(
                1, valid_mask.size(-1) + 1, device=pred.device
            )
            last_valid_idx = (valid_mask * step_weight).argmax(dim=-1)
            mode_endpoints = pred_topk[batch_idx, :, last_valid_idx]
            gt_endpoint = target[batch_idx, last_valid_idx].unsqueeze(-2)
            best_mode = torch.linalg.vector_norm(
                mode_endpoints - gt_endpoint, ord=2, dim=-1
            ).argmin(dim=-1)
            best_traj = pred_topk[batch_idx, best_mode]
            per_step_err = torch.linalg.vector_norm(
                best_traj - target, ord=2, dim=-1
            )
            ade = (per_step_err * valid_mask).sum(dim=-1) / valid_counts
            self.sum += ade.sum()
        elif min_criterion == "ADE":
            # Average displacement over valid steps for every mode, then take
            # the mode with the lowest ADE.
            per_step_err = torch.linalg.vector_norm(
                pred_topk - target.unsqueeze(1), ord=2, dim=-1
            )
            mode_ade = (per_step_err * valid_mask.unsqueeze(1)).sum(
                dim=-1
            ) / valid_counts.unsqueeze(1)
            self.sum += mode_ade.amin(dim=-1).sum()
        else:
            raise ValueError(
                "{} is not a valid criterion".format(min_criterion)
            )
        self.count += num_samples

    def compute(self) -> torch.Tensor:
        return self.sum / self.count


@OBJECT_REGISTRY.register
class MinFDE(EvalMetric):
    """
    Minimum Final Displacement Error (minFDE) for trajectory predictions.

    The minFDE calculates the final displacement error of the best
    prediction out of a set of predictions.

    Args:
        max_guesses: Maximum number of guesses to consider
            for top-k predictions.
        name: Name of the metric.
    """

    def __init__(self, max_guesses: int = 6, name: str = "minFDE") -> None:
        super(MinFDE, self).__init__(name)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.max_guesses = max_guesses

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        prob: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        keep_invalid_final_step: bool = True,
    ) -> None:
        pred, target, prob, valid_mask, _ = valid_filter(
            pred, target, prob, valid_mask, keep_invalid_final_step
        )
        pred_topk, _ = topk(self.max_guesses, pred, prob)

        num_samples = pred.size(0)
        batch_idx = torch.arange(num_samples, device=pred.device)
        # Locate the last valid time step per sample (largest valid index).
        step_weight = torch.arange(
            1, valid_mask.size(-1) + 1, device=pred.device
        )
        inds_last = (valid_mask * step_weight).argmax(dim=-1)

        # Final-step endpoint for each mode against the GT endpoint.
        final_pred = pred_topk[batch_idx, :, inds_last]
        final_gt = target[batch_idx, inds_last].unsqueeze(-2)
        endpoint_err = torch.norm(final_pred - final_gt, p=2, dim=-1)

        self.sum += endpoint_err.min(dim=-1)[0].sum()
        self.count += num_samples

    def compute(self) -> torch.Tensor:
        return self.sum / self.count


@OBJECT_REGISTRY.register
class MissRate(EvalMetric):
    """
    Miss Rate (MR) evaluation metric for trajectory predictions.

    The MR calculates the rate of missed predictions based on a
    specified criterion and threshold.

    Args:
        max_guesses: Maximum number of guesses to consider for top-k
            predictions.
        name: Name of the metric.
    """

    def __init__(self, max_guesses: int = 6, name: str = "MR") -> None:
        super(MissRate, self).__init__(name)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.max_guesses = max_guesses

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        prob: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        keep_invalid_final_step: bool = True,
        miss_criterion: str = "FDE",
        miss_threshold: float = 2.0,
    ) -> None:
        pred, target, prob, valid_mask, _ = valid_filter(
            pred, target, prob, valid_mask, keep_invalid_final_step
        )
        pred_topk, _ = topk(self.max_guesses, pred, prob)

        num_samples = pred.size(0)
        batch_idx = torch.arange(num_samples, device=pred.device)

        if miss_criterion == "FDE":
            step_weight = torch.arange(
                1, valid_mask.size(-1) + 1, device=pred.device
            )
            last_valid_idx = (valid_mask * step_weight).argmax(dim=-1)
            mode_endpoints = pred_topk[batch_idx, :, last_valid_idx]
            gt_endpoint = target[batch_idx, last_valid_idx].unsqueeze(-2)
            best_mode_error = torch.linalg.vector_norm(
                mode_endpoints - gt_endpoint, ord=2, dim=-1
            ).amin(dim=-1)
            self.sum += (best_mode_error > miss_threshold).sum()
        elif miss_criterion == "MAXDE":
            per_step_error = torch.linalg.vector_norm(
                pred_topk - target.unsqueeze(1), ord=2, dim=-1
            )
            validity = valid_mask.unsqueeze(1)
            worst_step_error = (per_step_error * validity).amax(dim=-1)
            best_mode_error = worst_step_error.amin(dim=-1)
            self.sum += (best_mode_error > miss_threshold).sum()
        else:
            raise ValueError(
                "{} is not a valid criterion".format(miss_criterion)
            )
        self.count += num_samples

    def compute(self) -> torch.Tensor:
        return self.sum / self.count


@OBJECT_REGISTRY.register
class HitRate(EvalMetric):
    """
    Hit rate evaluation metric for trajectory predictions.

    Args:
        max_guesses: Maximum number of guesses to consider for top-k
            predictions.
        name: Name of the metric.
    """

    def __init__(
        self,
        max_guesses: int = 6,
        name: str = "HitRate",
    ) -> None:
        super(HitRate, self).__init__(name)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.max_guesses = max_guesses

    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        prob: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        keep_invalid_final_step: bool = True,
        miss_criterion: str = "FDE",
        miss_threshold: float = 2.0,
    ) -> None:
        pred, target, prob, valid_mask, _ = valid_filter(
            pred, target, prob, valid_mask, keep_invalid_final_step
        )
        pred_topk, _ = topk(self.max_guesses, pred, prob)

        num_samples = pred.size(0)
        batch_idx = torch.arange(num_samples, device=pred.device)

        if miss_criterion == "FDE":
            step_weight = torch.arange(
                1, valid_mask.size(-1) + 1, device=pred.device
            )
            last_valid_idx = (valid_mask * step_weight).argmax(dim=-1)
            mode_endpoints = pred_topk[batch_idx, :, last_valid_idx]
            gt_endpoint = target[batch_idx, last_valid_idx].unsqueeze(-2)
            best_mode_error = torch.linalg.vector_norm(
                mode_endpoints - gt_endpoint, ord=2, dim=-1
            ).amin(dim=-1)
            self.sum += (best_mode_error <= miss_threshold).sum()
        elif miss_criterion == "MAXDE":
            per_step_error = torch.linalg.vector_norm(
                pred_topk - target.unsqueeze(1), ord=2, dim=-1
            )
            validity = valid_mask.unsqueeze(1)
            worst_step_error = (per_step_error * validity).amax(dim=-1)
            best_mode_error = worst_step_error.amin(dim=-1)
            self.sum += (best_mode_error <= miss_threshold).sum()
        else:
            raise ValueError(
                "{} is not a valid criterion".format(miss_criterion)
            )
        self.count += num_samples

    def compute(self) -> torch.Tensor:
        return self.sum / self.count
