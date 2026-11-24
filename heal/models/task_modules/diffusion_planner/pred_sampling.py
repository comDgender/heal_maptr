# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import torch
import torch.nn as nn

from heal.models.task_modules.diffusion_planner.dpm_solver import (
    DPMSolverScheduler,
)


class DPMSampler(nn.Module):
    """DPM-Solver based sampler for diffusion planner decoding.

    It wraps the planner model with DPMSolverScheduler to run multistep
    sampling.
    """

    def __init__(
        self,
        model_wrapper: nn.Module,
        diffusion_steps: int = 10,
        correcting_xt_fn=None,
        noise_schedule=None,
    ):
        """Initialize the DPM sampler.

        Args:
            model_wrapper: Wrapped planner model used as the noise
                prediction function.
            diffusion_steps: Number of sampling steps for the DPM-Solver.
            correcting_xt_fn: Optional callable applied to correct
                intermediate samples.
            noise_schedule: Optional noise schedule configuration passed
                to the scheduler.
        """
        super().__init__()

        self.diffusion_steps = diffusion_steps
        model_fn = model_wrapper

        self.dpm_solver = DPMSolverScheduler(
            model_fn,
            noise_schedule,
            algorithm_type="dpmsolver++",
            correcting_xt_fn=correcting_xt_fn,
        )

    @torch.no_grad()
    def forward(self, x_T, model_condition, classifier_kwargs=None):
        """Generate a sample from the initial noise.

        Args:
            x_T: Initial noise tensor at the terminal diffusion step.
            model_condition: Conditioning inputs forwarded to the planner
                model.
            classifier_kwargs: Optional keyword arguments for classifier
                guidance.

        Returns:
            Denoised sample tensor produced by the DPM-Solver.
        """
        if classifier_kwargs is None:
            classifier_kwargs = {}
        sample = self.dpm_solver(
            x=x_T,
            model_condition=model_condition,
            steps=self.diffusion_steps,
            order=2,
            skip_type="logSNR",
            method="multistep",
            denoise_to_zero=True,
            classifier_kwargs=classifier_kwargs,
        )
        return sample
