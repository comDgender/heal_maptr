# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import torch
import torch.nn as nn
from hat_engine.registry import OBJECT_REGISTRY

from heal.models.task_modules.diffusion_planner._dpm_utils import (
    expand_dims,
    interpolate_fn,
)
from heal.models.task_modules.diffusion_planner.dpm_scheduler import (  # noqa
    DPMSolverScheduler,
)

# Re-export the heavy solver classes for backward-compatible imports.
# DPM_Solver lives in dpm_solver_core; DPMSolverScheduler lives in
# dpm_scheduler (split to keep each file under 1000 lines).
from heal.models.task_modules.diffusion_planner.dpm_solver_core import (  # noqa
    DPM_Solver,
)


@OBJECT_REGISTRY.register
class NoiseScheduleVP:
    """Variance-preserving noise schedule for the forward diffusion SDE."""

    def __init__(
        self,
        schedule="discrete",
        betas=None,
        alphas_cumprod=None,
        continuous_beta_0=0.1,
        continuous_beta_1=20.0,
        dtype=torch.float32,
    ):
        """Create a wrapper class for the forward SDE (VP type).

        The forward SDE ensures that the condition distribution
        q_{t|0}(x_t | x_0) = N ( alpha_t * x_0, sigma_t^2 * I ).
        We define lambda_t = log(alpha_t) - log(sigma_t), the half-logSNR.
        """

        if schedule not in ["discrete", "linear"]:
            raise ValueError(
                "Unsupported noise schedule {}. The schedule needs to be "
                "'discrete' or 'linear'".format(schedule)
            )

        self.schedule = schedule
        if schedule == "discrete":
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.T = 1.0
            self.log_alpha_array = (
                self.numerical_clip_alpha(log_alphas)
                .reshape((1, -1))
                .to(dtype=dtype)
            )
            self.total_N = self.log_alpha_array.shape[1]
            self.t_array = (
                torch.linspace(0.0, 1.0, self.total_N + 1)[1:]
                .reshape((1, -1))
                .to(dtype=dtype)
            )
        else:
            self.T = 1.0
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1

    def numerical_clip_alpha(self, log_alphas, clipped_lambda=-5.1):
        """Clip log-SNR near t=T within -5.1 to ensure stability."""
        log_sigmas = 0.5 * torch.log(1.0 - torch.exp(2.0 * log_alphas))
        lambs = log_alphas - log_sigmas
        idx = torch.searchsorted(torch.flip(lambs, [0]), clipped_lambda)
        if idx > 0:
            log_alphas = log_alphas[:-idx]
        return log_alphas

    def marginal_log_mean_coeff(self, t):
        """Compute log(alpha_t) of a given continuous-time label t in [0, T].

        Args:
            t: Continuous-time label tensor in [0, T].

        Returns:
            log(alpha_t) tensor with the same shape as t.
        """
        if self.schedule == "discrete":
            return interpolate_fn(
                t.reshape((-1, 1)),
                self.t_array.to(t.device),
                self.log_alpha_array.to(t.device),
            ).reshape((-1))
        elif self.schedule == "linear":
            return (
                -0.25 * t**2 * (self.beta_1 - self.beta_0)
                - 0.5 * t * self.beta_0
            )

    def marginal_alpha(self, t):
        """Compute alpha_t of a given continuous-time label t in [0, T]."""
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """Compute sigma_t of a given continuous-time label t in [0, T]."""
        return torch.sqrt(
            1.0 - torch.exp(2.0 * self.marginal_log_mean_coeff(t))
        )

    def marginal_lambda(self, t):
        """Compute lambda_t = log(alpha_t) - log(sigma_t)."""
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1.0 - torch.exp(2.0 * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """Compute the continuous-time label t of a given half-logSNR lambda_t.

        Args:
            lamb: Half-logSNR tensor lambda_t.

        Returns:
            Continuous-time label tensor t corresponding to lamb.
        """
        if self.schedule == "linear":
            tmp = (
                2.0
                * (self.beta_1 - self.beta_0)
                * torch.logaddexp(-2.0 * lamb, torch.zeros((1,)).to(lamb))
            )
            delta = self.beta_0**2 + tmp
            return (
                tmp
                / (torch.sqrt(delta) + self.beta_0)
                / (self.beta_1 - self.beta_0)
            )
        elif self.schedule == "discrete":
            log_alpha = -0.5 * torch.logaddexp(
                torch.zeros((1,)).to(lamb.device), -2.0 * lamb
            )
            t = interpolate_fn(
                log_alpha.reshape((-1, 1)),
                torch.flip(self.log_alpha_array.to(lamb.device), [1]),
                torch.flip(self.t_array.to(lamb.device), [1]),
            )
            return t.reshape((-1,))


class ModelWrapper(nn.Module):
    """Wrap a diffusion model to expose a unified noise-prediction API."""

    def __init__(
        self,
        model,
        noise_schedule,
        model_type="noise",
        guidance_fn=None,
        guidance_type="uncond",
        guidance_scale=1.0,
    ):
        """Convert different types of models to the noise prediction model.

        model_type: "noise" | "x_start" | "v" | "score"
        guidance_type: "uncond" | "classifier" | "classifier-free"
        """
        super().__init__()

        self.wrapper_model = model
        self.noise_schedule = noise_schedule
        self.model_type = model_type
        self.guidance_fn = guidance_fn
        self.guidance_type = guidance_type
        self.guidance_scale = guidance_scale

        assert model_type in ["noise", "x_start", "v", "score"]
        assert guidance_type in ["uncond", "classifier", "classifier-free"]

    def get_model_input_time(self, t_cont):
        """Convert continuous time to the model's expected time input.

        Args:
            t_cont: Continuous-time label tensor in [0, T].

        Returns:
            Time tensor consumed by the wrapped model.
        """
        if self.noise_schedule.schedule == "discrete":
            return (t_cont - 1.0 / self.noise_schedule.total_N) * 1000.0
        else:
            return t_cont

    # --------------------- noise_pred_fn ---------------------
    def noise_pred_fn(
        self, x, t_continuous, cond, model_kwargs, step: int = 0
    ):
        """Compute the unified noise prediction from the wrapped model output.

        Args:
            x: Noisy sample tensor x_t.
            t_continuous: Continuous-time label tensor in [0, T].
            cond: Conditioning tensor passed to the model, or None.
            model_kwargs: Extra keyword arguments forwarded to the model.
            step: Solver step index selecting the model from a ModuleList.

        Returns:
            Noise prediction tensor with the same shape as x.
        """
        t_input = self.get_model_input_time(t_continuous)
        model = (
            self.wrapper_model[step]
            if isinstance(self.wrapper_model, nn.ModuleList)
            else self.wrapper_model
        )

        # call model (conditional or not)
        if cond is None:
            out = model(x, t_input, **model_kwargs)
        else:
            out = model(x, t_input, cond, **model_kwargs)

        # convert to noise prediction
        if self.model_type == "noise":
            return out

        alpha_t = self.noise_schedule.marginal_alpha(t_continuous)
        sigma_t = self.noise_schedule.marginal_std(t_continuous)

        if self.model_type == "x_start":
            return (x - expand_dims(alpha_t, x.dim()) * out) / expand_dims(
                sigma_t, x.dim()
            )

        elif self.model_type == "v":
            return (
                expand_dims(alpha_t, x.dim()) * out
                + expand_dims(sigma_t, x.dim()) * x
            )

        elif self.model_type == "score":
            return -expand_dims(sigma_t, x.dim()) * out

    # --------------------- classifier gradient ---------------------
    def cond_grad_fn(self, x, t_input, condition, classifier_kwargs):
        """Compute the classifier gradient grad_x log p(y|x_t).

        Args:
            x: Noisy sample tensor x_t.
            t_input: Model time input tensor.
            condition: Condition tensor y for the guidance function.
            classifier_kwargs: Extra keyword arguments for guidance_fn.

        Returns:
            Gradient tensor with the same shape as x.
        """
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            logprob = self.guidance_fn(
                x_req, t_input, condition, **classifier_kwargs
            )
            grad = torch.autograd.grad(logprob.sum(), x_req)[0]
        return grad

    # --------------------- main forward: return model_fn(x,t_cont) -----
    def forward(
        self,
        x,
        t_continuous,
        model_kwargs=None,
        classifier_kwargs=None,
        condition=None,
        step: int = 0,
    ):
        """Return model_fn(x, t_continuous), used for each DPM-Solver step."""
        if model_kwargs is None:
            model_kwargs = {}
        if classifier_kwargs is None:
            classifier_kwargs = {}
        t_continuous = t_continuous.expand((x.shape[0]))
        if self.guidance_type == "uncond":
            return self.noise_pred_fn(
                x,
                t_continuous,
                cond=None,
                model_kwargs=model_kwargs,
                step=step,
            )

        elif self.guidance_type == "classifier":
            assert self.guidance_fn is not None
            t_input = self.get_model_input_time(t_continuous)

            # grad_x log p(y|x_t)
            cond_grad = self.cond_grad_fn(
                x, t_input, condition, classifier_kwargs
            )
            sigma_t = self.noise_schedule.marginal_std(t_continuous)

            noise = self.noise_pred_fn(
                x,
                t_continuous,
                cond=None,
                model_kwargs=model_kwargs,
                step=step,
            )

            return (
                noise
                - self.guidance_scale
                * expand_dims(sigma_t, x.dim())
                * cond_grad
            )
