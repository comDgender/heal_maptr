# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import torch
import torch.nn as nn
from hat_engine.registry import OBJECT_REGISTRY
from horizon_plugin_pytorch.quantization import QuantStub
from torch.quantization import DeQuantStub

from heal.models.task_modules.diffusion_planner.dpm_solver import ModelWrapper
from heal.models.task_modules.diffusion_planner.normalizer import (
    ObservationNormalizer,
    StateNormalizer,
)
from heal.models.task_modules.diffusion_planner.pred_sampling import DPMSampler


@OBJECT_REGISTRY.register
class DiffusionPlannerDecoder(nn.Module):
    """Diffusion planner decoder with train/infer unified forward.

    It denoises trajectory candidates with a DiT backbone under scene
    context, applying state/observation normalizers for deployment.
    """

    def __init__(
        self,
        dit: nn.Module,
        future_len: int,
        normalization_file_path: str,
        predicted_neighbor_num: int = 10,
        diffusion_pred_steps: int = 10,
        noise_schedule=None,
        guidance_fn=None,
        deploy_model: bool = False,
        use_neighbors_future: bool = True,
    ):
        """Initialize the diffusion planner decoder.

        Args:
            dit: DiT denoising model consuming sampled trajectories.
            future_len: Number of future steps to predict.
            normalization_file_path: Path to file with normalization stats.
            predicted_neighbor_num: Number of neighbor agents to predict.
            diffusion_pred_steps: Number of diffusion sampling steps.
            noise_schedule: Noise schedule used during sampling.
            guidance_fn: Optional classifier guidance callable.
            deploy_model: Use deploy-friendly input noise and output path.
            use_neighbors_future: Predict neighbor futures in addition to ego.
        """
        super().__init__()

        self._predicted_neighbor_num = predicted_neighbor_num
        self._future_len = future_len

        self.dit = dit

        self._state_normalizer: StateNormalizer = StateNormalizer.from_json(
            normalization_file_path, use_neighbors_future=use_neighbors_future
        )
        self._observation_normalizer: ObservationNormalizer = (
            ObservationNormalizer.from_json(normalization_file_path)
        )

        self._guidance_fn = guidance_fn
        self.diffusion_pred_steps = diffusion_pred_steps
        self.deploy_model = deploy_model
        self.use_neighbors_future = use_neighbors_future

        self.sample = DPMSampler(
            model_wrapper=ModelWrapper(
                model=self.dit,
                noise_schedule=noise_schedule,
                model_type=self.dit.model_type,
                guidance_scale=0.5,
                guidance_fn=self._guidance_fn,
                guidance_type=(
                    "classifier" if self._guidance_fn is not None else "uncond"
                ),
            ),
            diffusion_steps=self.diffusion_pred_steps,
            noise_schedule=noise_schedule,
        )

        self.ego_current_quant = QuantStub()
        self.encoding_quant = QuantStub()
        self.route_lanes_quant = QuantStub()
        self.noise_quant = QuantStub()
        self.dequant = DeQuantStub()

    @property
    def sde(self):
        """Return whether the underlying DiT uses SDE sampling."""
        return self.dit.sde

    def forward(
        self, encoder_outputs, inputs, quantized_neighbor_agents_past=None
    ):
        """Run diffusion decoding for training or inference.

        Args:
            encoder_outputs: Dict with scene encoding under ``encoding``.
            inputs: Batch dict of observations and route lane tensors.
            quantized_neighbor_agents_past: Optional quantized neighbor
                history.

        Returns:
            Dict with ``score`` during training or ``prediction`` at inference.
        """
        D = 4
        ego_current = inputs["ego_current_state"][:, None, :D]
        ego_current = self.ego_current_quant(ego_current)

        if (
            self.use_neighbors_future
            and quantized_neighbor_agents_past is not None
        ):
            neighbors_current = quantized_neighbor_agents_past[
                :, : self._predicted_neighbor_num, -1, :D
            ]
            neighbor_current_mask = (
                torch.sum(
                    torch.ne(neighbors_current[..., :4], 0).to(torch.int8),
                    dim=-1,
                )
                == 0
            )
            inputs["neighbor_current_mask"] = self.dequant(
                neighbor_current_mask
            )

            current_states = torch.cat(
                [ego_current, neighbors_current], dim=1
            )  # [B, 1 + Pn, 4]

            B, P, _ = current_states.shape
            assert P == (1 + self._predicted_neighbor_num)
        else:
            # ego only, no neighbor
            current_states = ego_current  # [B, 1, 4]
            B, P, _ = current_states.shape
            neighbor_current_mask = None

        # Extract context encoding
        ego_neighbor_encoding = self.encoding_quant(
            encoder_outputs["encoding"]
        )
        route_lanes = self.route_lanes_quant(inputs["route_lanes"])
        route_lanes_speed_limit = inputs["route_lanes_speed_limit"]
        route_lanes_has_speed_limit = inputs["route_lanes_has_speed_limit"]

        if self.training:
            sampled_trajectories = inputs["sampled_trajectories"].reshape(
                B, P, -1
            )
            diffusion_time = inputs["diffusion_time"]

            return {
                "score": self.dit(
                    sampled_trajectories,
                    diffusion_time,
                    ego_neighbor_encoding,
                    route_lanes,
                    route_lanes_speed_limit,
                    route_lanes_has_speed_limit,
                    neighbor_current_mask,
                    current_states[:, :, None, :],
                ).reshape(B, P, -1, 4)
            }
        else:
            if self.deploy_model:
                noise = inputs["noise"]
            else:
                noise = torch.randn(B, P, self._future_len, D).to(
                    current_states.device
                )
            noise = self.noise_quant(noise)
            xT = torch.cat(
                [current_states[:, :, None], noise * 0.5], dim=2
            ).reshape(B, P, -1)

            with torch.no_grad():
                model_condition = {
                    "cross_c": ego_neighbor_encoding,
                    "route_lanes": route_lanes,
                    "route_lanes_speed_limit": route_lanes_speed_limit,
                    "route_lanes_has_speed_limit": route_lanes_has_speed_limit,
                    "neighbor_current_mask": neighbor_current_mask,
                    "current_states": current_states[:, :, None, :],
                }

                x0 = self.sample(
                    x_T=xT,
                    model_condition=model_condition,
                    classifier_kwargs={
                        "model": self.dit,
                        "model_condition": model_condition,
                        "inputs": inputs,
                        "observation_normalizer": self._observation_normalizer,
                        "state_normalizer": self._state_normalizer,
                    },
                )
                x0 = self._state_normalizer.inverse(x0.reshape(B, P, -1, 4))[
                    :, :, 1:
                ]
                x0 = self.dequant(x0)

            return {"prediction": x0}
