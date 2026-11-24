# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import logging

import torch
from hat_engine.registry import OBJECT_REGISTRY
from torch import nn

from heal.models.task_modules.diffusion_planner.normalizer import (
    StateNormalizer,
)

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class DiffusionLoss(nn.Module):
    def __init__(
        self,
        model_type: str = "score",
        ego_loss_weight: float = 1.0,
        collision_loss: nn.Module = None,
        normalization_file_path=None,
        collision_loss_weight: float = 1.0,
        velocity_loss_weight: float = 1.0,
        current_frame_loss: bool = False,
        use_neighbors_future: bool = True,
    ):
        """model_type: "score" or "x_start", decides loss computation."""
        super(DiffusionLoss, self).__init__()
        self.model_type = model_type
        self.ego_loss_weight = ego_loss_weight
        self.collision_loss = collision_loss
        self.state_normalizer: StateNormalizer = StateNormalizer.from_json(
            normalization_file_path
        )
        self.collision_loss_weight = collision_loss_weight
        self.velocity_loss_weight = velocity_loss_weight
        self.current_frame_loss = current_frame_loss
        self.use_neighbors_future = use_neighbors_future

    def forward(self, decoder_output, inputs):
        """Compute the diffusion loss.

        :param decoder_output: model decoder output, should contain "score".
        :param inputs: input dict with "diffusion_std", "diffusion_noise",
            "sampled_trajectories", etc. When use_neighbors_future=True also
            needs "neighbors_future_valid".
        :return: (loss dict, decoder output).
        """
        score = decoder_output["score"][:, :, 1:, :]  # [B, P, T, 4]
        B, P, T, _ = score.shape
        std = inputs["diffusion_std"]
        z = inputs["diffusion_noise"]
        all_gt = inputs["ego_neighbor_gt"]

        if self.model_type == "score":
            dpm_loss = torch.sum((score * std + z) ** 2, dim=-1)
        elif self.model_type == "x_start":
            dpm_loss = torch.sum((score - all_gt[:, :, 1:, :4]) ** 2, dim=-1)

        loss = {}

        # Ego planning loss (always computed)
        loss["ego_planning_loss"] = dpm_loss[:, 0, :].mean()

        if self.use_neighbors_future:
            neighbors_future_valid = inputs["neighbors_future_valid"]
            masked_prediction_loss = dpm_loss[:, 1:, :][neighbors_future_valid]
            if masked_prediction_loss.numel() > 0:
                loss["neighbor_prediction_loss"] = (
                    masked_prediction_loss.mean()
                )
            else:
                loss["neighbor_prediction_loss"] = torch.tensor(
                    0.0, device=masked_prediction_loss.device
                )
            loss["loss"] = (
                loss["neighbor_prediction_loss"]
                + self.ego_loss_weight * loss["ego_planning_loss"]
            )
        else:
            # ego only, no neighbor loss
            loss["loss"] = self.ego_loss_weight * loss["ego_planning_loss"]

        if self.current_frame_loss:
            current_score = decoder_output["score"][:, :, 0, :]
            current_dpm_loss = torch.sum(
                (current_score - all_gt[:, :, 0, :4]) ** 2, dim=-1
            )
            loss["current_frame_loss"] = current_dpm_loss.mean()
            loss["loss"] += loss["current_frame_loss"]

        if self.collision_loss is not None:
            x_in = inputs["sampled_trajectories"].reshape(B, P, -1)
            x_fix = decoder_output["score"].reshape(B, P, -1) - x_in
            x_fix = x_fix.reshape(B, P, -1, 4)
            x_fix[:, :, 0] = 0.0
            x_in = x_in + x_fix.reshape(B, P, -1)
            x_in = self.state_normalizer.inverse(x_in.reshape(B, P, -1, 4))
            collision_loss = self.collision_loss(x_in, inputs)
            loss["collision_loss"] = collision_loss
            loss["loss"] += self.collision_loss_weight * loss["collision_loss"]

        assert not torch.isnan(dpm_loss).sum(), f"loss cannot be nan, z={z}"

        return loss, decoder_output
