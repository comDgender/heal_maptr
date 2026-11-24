# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import Sequence, Union

import numpy as np
import torch
from hat_engine.registry import OBJECT_REGISTRY

from heal.data.transforms.diffusion_planner.vehicle_parameters import (
    get_pacifica_parameters,
)
from heal.models.task_modules.diffusion_planner.normalizer import (
    ObservationNormalizer,
    StateNormalizer,
)

__all__ = [
    "StatePerturbation",
    "NoiseScheduleSampler",
]

NUM_REFINE = 20
REFINE_HORIZON = 2.0
TIME_INTERVAL = 0.1


def vector_transform(vector, transform_mat, bias=None):
    """vector: (B, ..., 2); transform_mat: (B, 2, 2); bias: (B, ..., 2)."""
    shape = vector.shape
    B = vector.shape[0]
    nexpand = vector.ndim - 2
    if bias is not None:
        vector = vector - bias.reshape(B, *([1] * nexpand), -1)
    vector = vector.reshape(B, -1, 2).permute(0, 2, 1)  # (B, 2, N1 * N2 ...)
    return (
        torch.bmm(transform_mat, vector).permute(0, 2, 1).reshape(*shape)
    )  # (B, ..., 2)


def heading_transform(heading, transform_mat):
    """heading: (B, ...); transform_mat: (B, 2, 2)."""
    B = heading.shape[0]
    shape = heading.shape
    heading = heading.reshape(B, -1)
    transform_mat = transform_mat.reshape(B, 1, 2, 2)
    return torch.atan2(
        torch.cos(heading) * transform_mat[..., 1, 0]
        + torch.sin(heading) * transform_mat[..., 1, 1],
        torch.cos(heading) * transform_mat[..., 0, 0]
        + torch.sin(heading) * transform_mat[..., 0, 1],
    ).reshape(*shape)


@OBJECT_REGISTRY.register
class StatePerturbation:
    """Data augmentation that perturbs the current ego position.

    Generates a feasible trajectory that satisfies polynomial constraints.
    """

    def __init__(
        self,
        low: Sequence[float] = (
            -0.0,
            -0.75,
            -0.35,
            -1,
            -0.5,
            -0.2,
            -0.1,
            0.0,
            -0.0,
        ),
        high: Sequence[float] = (0.0, 0.75, 0.35, 1, 0.5, 0.2, 0.1, 0.0, 0.0),
        augment_prob: float = 0.5,
        normalize=True,
        refine_horizon: float = REFINE_HORIZON,
        num_refine: int = NUM_REFINE,
        time_interval: float = TIME_INTERVAL,
    ) -> None:
        """Initialize the augmentor.

        :param low: lower bound of Uniform noise on
            [x, y, yaw, vx, vy, ax, ay, steering angle, yaw rate].
        :param high: upper bound of Uniform noise on the same vector.
        :param augment_prob: probability between 0 and 1 of applying the
            data augmentation.
        :param refine_horizon: time horizon (seconds) for polynomial
            interpolation.
        :param num_refine: number of future frames in the refine horizon.
        :param time_interval: time interval (seconds) between future frames.
        """
        self._augment_prob = augment_prob
        self._normalize = normalize
        self._low = torch.tensor(low)
        self._high = torch.tensor(high)
        self._wheel_base = get_pacifica_parameters().wheel_base

        self.refine_horizon = refine_horizon
        self.num_refine = num_refine
        self.time_interval = time_interval

        T = refine_horizon + time_interval
        self.coeff_matrix = torch.linalg.inv(
            torch.tensor(
                [
                    [1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0],
                    [0, 0, 2, 0, 0, 0],
                    [1, T, T**2, T**3, T**4, T**5],
                    [0, 1, 2 * T, 3 * T**2, 4 * T**3, 5 * T**4],
                    [0, 0, 2, 6 * T, 12 * T**2, 20 * T**3],
                ],
                dtype=torch.float32,
            )
        )
        self.t_matrix = torch.pow(
            torch.linspace(
                time_interval, refine_horizon, num_refine
            ).unsqueeze(1),
            torch.arange(6).unsqueeze(0),
        )

    def __call__(self, inputs):
        ego_future, neighbors_future = (
            inputs["ego_future_gt"],
            inputs["neighbors_future_gt"],
        )
        aug_flag, aug_ego_current_state = self.augment(inputs)
        interpolated_ego_future = self.interpolation_future_trajectory(
            aug_ego_current_state, ego_future
        )

        inputs["ego_current_state"][aug_flag] = aug_ego_current_state[aug_flag]
        ego_future[aug_flag] = interpolated_ego_future[aug_flag]

        return self.centric_transform(inputs, ego_future, neighbors_future)

    def augment(self, inputs):
        # Only aug current state
        ego_current_state = inputs["ego_current_state"].clone()
        _device = ego_current_state.device

        B = ego_current_state.shape[0]
        aug_flag = (torch.rand(B) >= self._augment_prob).bool().to(
            _device
        ) & ~(abs(ego_current_state[:, 4]) < 2.0)

        self._low = self._low.to(_device)
        self._high = self._high.to(_device)
        self.coeff_matrix = self.coeff_matrix.to(_device)
        self.t_matrix = self.t_matrix.to(_device)
        random_tensor = torch.rand(B, len(self._low)).to(_device)
        scaled_random_tensor = (
            self._low + (self._high - self._low) * random_tensor
        )

        new_state = torch.zeros((B, 9), dtype=torch.float32).to(_device)
        # x, y, h is 0 (ego-centric), update vx, vy, ax, ay, steering, yaw rate
        new_state[:, 3:] = ego_current_state[:, 4:10]
        new_state = new_state + scaled_random_tensor
        new_state[:, 3] = torch.max(
            new_state[:, 3], torch.tensor(0.0, device=new_state.device)
        )
        new_state[:, -1] = torch.clip(new_state[:, -1], -0.85, 0.85)

        ego_current_state[:, :2] = new_state[:, :2]
        ego_current_state[:, 2] = torch.cos(new_state[:, 2])
        ego_current_state[:, 3] = torch.sin(new_state[:, 2])
        ego_current_state[:, 4:8] = new_state[:, 3:7]
        ego_current_state[:, 8:10] = new_state[
            :, -2:
        ]  # steering angle, yaw rate

        # update steering angle and yaw rate
        cur_velocity = ego_current_state[:, 4]
        yaw_rate = ego_current_state[:, 9]

        steering_angle = torch.zeros_like(cur_velocity)
        new_yaw_rate = torch.zeros_like(yaw_rate)

        mask = torch.abs(cur_velocity) < 0.2
        not_mask = ~mask
        steering_angle[not_mask] = torch.atan(
            yaw_rate[not_mask]
            * self._wheel_base
            / torch.abs(cur_velocity[not_mask])
        )
        steering_angle[not_mask] = torch.clamp(
            steering_angle[not_mask], -2 / 3 * np.pi, 2 / 3 * np.pi
        )
        new_yaw_rate[not_mask] = yaw_rate[not_mask]

        ego_current_state[:, 8] = steering_angle
        ego_current_state[:, 9] = new_yaw_rate

        return aug_flag, ego_current_state

    def normalize_angle(
        self, angle: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def get_transform_matrix_batch(self, cur_state):
        processed_input = torch.column_stack(
            (
                cur_state[:, 2],  # cos
                cur_state[:, 3],  # sin
            )
        )

        reshaping_tensor = torch.tensor(
            [
                [1, 0, 0, 1],
                [0, 1, -1, 0],
            ],
            dtype=torch.float32,
        ).to(processed_input.device)

        # [b,2]*[2,4] -> [b,4] -> [b,2,2]
        return (processed_input @ reshaping_tensor).reshape(-1, 2, 2)

    def centric_transform(
        self,
        inputs: torch.Tensor,
        ego_future: torch.Tensor,
        neighbors_future: torch.Tensor,
    ):
        cur_state = inputs["ego_current_state"].clone()
        center_xy = cur_state[:, :2]
        transform_matrix = self.get_transform_matrix_batch(cur_state)

        # ego xy
        inputs["ego_current_state"][..., :2] = vector_transform(
            inputs["ego_current_state"][..., :2], transform_matrix, center_xy
        )
        # ego cos sin
        inputs["ego_current_state"][..., 2:4] = vector_transform(
            inputs["ego_current_state"][..., 2:4], transform_matrix
        )
        # ego vx, vy
        inputs["ego_current_state"][..., 4:6] = vector_transform(
            inputs["ego_current_state"][..., 4:6], transform_matrix
        )
        # ego ax, ay
        inputs["ego_current_state"][..., 6:8] = vector_transform(
            inputs["ego_current_state"][..., 6:8], transform_matrix
        )

        # ego future xy
        ego_future[..., :2] = vector_transform(
            ego_future[..., :2], transform_matrix, center_xy
        )
        ego_future[..., 2] = heading_transform(
            ego_future[..., 2], transform_matrix
        )
        inputs["ego_future_gt"] = ego_future

        # neighbor past xy
        mask = (
            torch.sum(
                torch.ne(inputs["neighbor_agents_past"][..., :6], 0), dim=-1
            )
            == 0
        )
        inputs["neighbor_agents_past"][..., :2] = vector_transform(
            inputs["neighbor_agents_past"][..., :2],
            transform_matrix,
            center_xy,
        )
        # neighbor past cos sin
        inputs["neighbor_agents_past"][..., 2:4] = vector_transform(
            inputs["neighbor_agents_past"][..., 2:4], transform_matrix
        )
        # neighbor past vx, vy
        inputs["neighbor_agents_past"][..., 4:6] = vector_transform(
            inputs["neighbor_agents_past"][..., 4:6], transform_matrix
        )
        inputs["neighbor_agents_past"][mask] = 0.0

        # neighbor future xy
        mask = torch.sum(torch.ne(neighbors_future[..., :2], 0), dim=-1) == 0
        neighbors_future[..., :2] = vector_transform(
            neighbors_future[..., :2], transform_matrix, center_xy
        )
        neighbors_future[..., 2] = heading_transform(
            neighbors_future[..., 2], transform_matrix
        )
        neighbors_future[mask] = 0.0
        inputs["neighbors_future_gt"] = neighbors_future

        # lanes
        mask = torch.sum(torch.ne(inputs["lanes"][..., :8], 0), dim=-1) == 0
        inputs["lanes"][..., :2] = vector_transform(
            inputs["lanes"][..., :2], transform_matrix, center_xy
        )
        inputs["lanes"][..., 2:4] = vector_transform(
            inputs["lanes"][..., 2:4], transform_matrix
        )
        inputs["lanes"][..., 4:6] = vector_transform(
            inputs["lanes"][..., 4:6], transform_matrix
        )
        inputs["lanes"][..., 6:8] = vector_transform(
            inputs["lanes"][..., 6:8], transform_matrix
        )
        inputs["lanes"][mask] = 0.0

        # route_lanes
        mask = (
            torch.sum(torch.ne(inputs["route_lanes"][..., :8], 0), dim=-1) == 0
        )
        inputs["route_lanes"][..., :2] = vector_transform(
            inputs["route_lanes"][..., :2], transform_matrix, center_xy
        )
        inputs["route_lanes"][..., 2:4] = vector_transform(
            inputs["route_lanes"][..., 2:4], transform_matrix
        )
        inputs["route_lanes"][..., 4:6] = vector_transform(
            inputs["route_lanes"][..., 4:6], transform_matrix
        )
        inputs["route_lanes"][..., 6:8] = vector_transform(
            inputs["route_lanes"][..., 6:8], transform_matrix
        )
        inputs["route_lanes"][mask] = 0.0

        # static objects xy
        mask = (
            torch.sum(torch.ne(inputs["static_objects"][..., :10], 0), dim=-1)
            == 0
        )
        inputs["static_objects"][..., :2] = vector_transform(
            inputs["static_objects"][..., :2], transform_matrix, center_xy
        )
        # static objects cos sin
        inputs["static_objects"][..., 2:4] = vector_transform(
            inputs["static_objects"][..., 2:4], transform_matrix
        )
        inputs["static_objects"][mask] = 0.0

        return inputs

    def interpolation_future_trajectory(self, aug_current_state, ego_future):
        """Refine future trajectory with quintic spline interpolation.

        Args:
            aug_current_state: (B, 16) current state after augmentation.
            ego_future: (B, 80, 3) future trajectory of the ego vehicle.
        Returns:
            ego_future: refined future trajectory.
        """
        P = self.num_refine
        dt = self.time_interval
        B = aug_current_state.shape[0]
        M_t = self.t_matrix.unsqueeze(0).expand(B, -1, -1)
        A = self.coeff_matrix.unsqueeze(0).expand(B, -1, -1)

        # state: [x, y, heading, velocity, acceleration, yaw_rate]
        x0, y0, theta0, v0, a0, omega0 = (
            aug_current_state[:, 0],
            aug_current_state[:, 1],
            torch.atan2(
                (ego_future[:, int(P / 2), 1] - aug_current_state[:, 1]),
                (ego_future[:, int(P / 2), 0] - aug_current_state[:, 0]),
            ),
            torch.norm(aug_current_state[:, 4:6], dim=-1),
            torch.norm(aug_current_state[:, 6:8], dim=-1),
            aug_current_state[:, 9],
        )

        xT, yT, thetaT, vT, aT, omegaT = (
            ego_future[:, P, 0],
            ego_future[:, P, 1],
            ego_future[:, P, 2],
            torch.norm(ego_future[:, P, :2] - ego_future[:, P - 1, :2], dim=-1)
            / dt,
            torch.norm(
                ego_future[:, P, :2]
                - 2 * ego_future[:, P - 1, :2]
                + ego_future[:, P - 2, :2],
                dim=-1,
            )
            / dt**2,
            self.normalize_angle(ego_future[:, P, 2] - ego_future[:, P - 1, 2])
            / dt,
        )

        # Boundary conditions
        sx = torch.stack(
            [
                x0,
                v0 * torch.cos(theta0),
                a0 * torch.cos(theta0) - v0 * torch.sin(theta0) * omega0,
                xT,
                vT * torch.cos(thetaT),
                aT * torch.cos(thetaT) - vT * torch.sin(thetaT) * omegaT,
            ],
            dim=-1,
        )

        sy = torch.stack(
            [
                y0,
                v0 * torch.sin(theta0),
                a0 * torch.sin(theta0) + v0 * torch.cos(theta0) * omega0,
                yT,
                vT * torch.sin(thetaT),
                aT * torch.sin(thetaT) + vT * torch.cos(thetaT) * omegaT,
            ],
            dim=-1,
        )

        ax = A @ sx[:, :, None]  # B, 6, 1
        ay = A @ sy[:, :, None]  # B, 6, 1

        traj_x = M_t @ ax
        traj_y = M_t @ ay
        traj_heading = torch.cat(
            [
                torch.atan2(
                    traj_y[:, :1, 0] - y0.unsqueeze(-1),
                    traj_x[:, :1, 0] - x0.unsqueeze(-1),
                ),
                torch.atan2(
                    traj_y[:, 1:, 0] - traj_y[:, :-1, 0],
                    traj_x[:, 1:, 0] - traj_x[:, :-1, 0],
                ),
            ],
            dim=1,
        )

        return torch.concatenate(
            [
                torch.cat([traj_x, traj_y, traj_heading[..., None]], axis=-1),
                ego_future[:, P:, :],
            ],
            axis=1,
        )


@OBJECT_REGISTRY.register
class AgentPerturbation:
    # Data augmentation that perturbs the scene including dynamic and static.

    def __init__(
        self,
        low: Sequence[float] = (
            -0.3,
            -0.3,
            -0.05,
            -0.05,
            -0.2,
            -0.2,
            -0.1,
            -0.1,
        ),
        high: Sequence[float] = (0.3, 0.3, 0.05, 0.05, 0.2, 0.2, 0.1, 0.1),
        augment_prob: float = 0.3,
    ) -> None:
        """Initialize the augmentor.

        :param low: lower bound of Uniform noise on
            [x, y, cos h, sin h, vx, vy, length, width].
        :param high: upper bound of Uniform noise on the same vector.
        :param augment_prob: probability between 0 and 1 of applying the
            data augmentation.
        """
        self._augment_prob = augment_prob
        self._low = torch.tensor(low)
        self._high = torch.tensor(high)

    def __call__(self, inputs):
        """inputs['neighbor_agents_past']: [B, 32, 21, 11]."""
        neighbor_agents_past = inputs["neighbor_agents_past"]
        device = neighbor_agents_past.device

        B, N, T, D = neighbor_agents_past.shape
        assert D == 11

        # batch-level augmentation flag
        aug_flag = torch.rand(B, device=device) < self._augment_prob

        self._low = self._low.to(device)
        self._high = self._high.to(device)

        # sample noise per agent (NOT per frame) [B, 32, 8]
        rand = torch.rand(B, N, len(self._low), device=device)
        scaled_random_tensor = self._low + (self._high - self._low) * rand

        # expand to time dimension [B, 32, 1, 8]
        scaled_random_tensor = scaled_random_tensor.unsqueeze(2)

        # apply augmentation only for aug_flag batches
        neighbor_agents_past[aug_flag, :, :, :8] += scaled_random_tensor[
            aug_flag
        ]

        # vx >= 0 (index 4)
        neighbor_agents_past[..., 4] = torch.clamp(
            neighbor_agents_past[..., 4], min=0.0
        )

        # normalize heading (cos, sin)
        cos_h = neighbor_agents_past[..., 2]
        sin_h = neighbor_agents_past[..., 3]
        norm = torch.sqrt(cos_h**2 + sin_h**2 + 1e-6)
        neighbor_agents_past[..., 2] = cos_h / norm
        neighbor_agents_past[..., 3] = sin_h / norm

        inputs["neighbor_agents_past"] = neighbor_agents_past
        return inputs


@OBJECT_REGISTRY.register
class LanePerturbation:
    """Geometric perturbation for lane polylines."""

    def __init__(
        self,
        xy_noise: float = 0.2,
        boundary_noise: float = 0.1,
        augment_prob: float = 0.3,
    ):
        self.xy_noise = xy_noise
        self.boundary_noise = boundary_noise
        self.augment_prob = augment_prob

    def __call__(self, inputs):
        """inputs['lanes']: [B, 70, 20, 12]."""
        lanes = inputs["lanes"]
        device = lanes.device

        B, L, T, D = lanes.shape
        assert D == 12

        aug_flag = torch.rand(B, device=device) < self.augment_prob

        # -------- 1. perturb centerline (x, y) --------
        xy_noise = (
            torch.rand(B, L, 1, 2, device=device) * 2 - 1
        ) * self.xy_noise
        lanes[aug_flag, :, :, 0:2] += xy_noise[aug_flag]

        # -------- 2. recompute dx, dy --------
        dxdy = lanes[:, :, 1:, 0:2] - lanes[:, :, :-1, 0:2]
        dxdy = torch.cat([dxdy, torch.zeros(B, L, 1, 2, device=device)], dim=2)
        lanes[:, :, :, 2:4] = dxdy

        # -------- 3. perturb left / right boundary offsets --------
        boundary_noise = (
            torch.rand(B, L, 1, 4, device=device) * 2 - 1
        ) * self.boundary_noise
        lanes[aug_flag, :, :, 4:8] += boundary_noise[aug_flag]

        # -------- 4. traffic light untouched --------
        inputs["lanes"] = lanes
        return inputs


@OBJECT_REGISTRY.register
class NoiseScheduleSampler:
    def __init__(
        self,
        sde,
        normalization_file_path,
        eps=1e-3,
        use_neighbors_future=True,
    ):
        """Initialize the noise schedule sampler.

        :param normalization_file_path: path to normalization mean/std json.
        :param sde: the SDE used by the diffusion model.
        :param eps: minimum value to prevent t=0.
        :param use_neighbors_future: whether to add noise to
            neighbors_future_gt; if False only ego_future_gt is noised.
        """
        self.marginal_prob = sde.marginal_prob
        self.eps = eps
        self.use_neighbors_future = use_neighbors_future
        self.observation_normalizer: ObservationNormalizer = (
            ObservationNormalizer.from_json(normalization_file_path)
        )
        self.state_normalizer = StateNormalizer.from_json(
            normalization_file_path, use_neighbors_future=use_neighbors_future
        )

    def __call__(self, inputs):
        ego_future = inputs["ego_future_gt"]

        # heading to cos sin
        ego_future = torch.cat(
            [
                ego_future[..., :2],
                torch.stack(
                    [ego_future[..., 2].cos(), ego_future[..., 2].sin()],
                    dim=-1,
                ),
            ],
            dim=-1,
        )

        inputs = self.observation_normalizer(inputs)

        B = ego_future.shape[0]

        if self.use_neighbors_future:
            neighbors_future = inputs["neighbors_future_gt"]
            mask = (
                torch.sum(torch.ne(neighbors_future[..., :3], 0), dim=-1) == 0
            )
            neighbors_future = torch.cat(
                [
                    neighbors_future[..., :2],
                    torch.stack(
                        [
                            neighbors_future[..., 2].cos(),
                            neighbors_future[..., 2].sin(),
                        ],
                        dim=-1,
                    ),
                ],
                dim=-1,
            )
            neighbors_future[mask] = 0.0

            neighbors_future_valid = ~mask  # [B, P, V]

            Pn = neighbors_future.shape[1]
            neighbors_current = inputs["neighbor_agents_past"][:, :Pn, -1, :4]
            neighbor_current_mask = (
                torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
            )
            neighbor_mask = torch.concat(
                (neighbor_current_mask.unsqueeze(-1), mask), dim=-1
            )

            gt_future = torch.cat(
                [ego_future[:, None, :, :], neighbors_future],
                dim=1,
            )  # [B, 1+Pn, T, 4]
            current_states = torch.cat(
                [inputs["ego_current_state"][:, None, :4], neighbors_current],
                dim=1,
            )  # [B, 1+Pn, 4]
        else:
            gt_future = ego_future[:, None, :, :]  # [B, 1, T, 4]
            current_states = inputs["ego_current_state"][
                :, None, :4
            ]  # [B, 1, 4]
            neighbors_future_valid = None
            neighbor_mask = None

        t = (
            torch.rand(B, device=gt_future.device) * (1 - self.eps) + self.eps
        )  # [B,]
        z = torch.randn_like(gt_future, device=gt_future.device)

        all_gt = torch.cat(
            [current_states[:, :, None, :], self.state_normalizer(gt_future)],
            dim=2,
        )
        if neighbor_mask is not None:
            all_gt[:, 1:][neighbor_mask] = 0.0

        mean, std = self.marginal_prob(all_gt[..., 1:, :], t)
        std = std.view(-1, *([1] * (len(all_gt[..., 1:, :].shape) - 1)))

        xT = mean + std * z
        xT = torch.cat([all_gt[:, :, :1, :], xT], dim=2)

        merged_inputs = {
            **inputs,
            "sampled_trajectories": xT,
            "diffusion_time": t,
            # for loss calculation
            "diffusion_mean": mean,
            "diffusion_std": std,
            "diffusion_noise": z,
            "ego_neighbor_gt": all_gt,
        }
        if neighbors_future_valid is not None:
            merged_inputs["neighbors_future_valid"] = neighbors_future_valid

        return merged_inputs
