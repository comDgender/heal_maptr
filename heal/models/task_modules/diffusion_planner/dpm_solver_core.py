# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import torch
import torch.nn as nn

from heal.models.task_modules.diffusion_planner._dpm_utils import expand_dims


class DPM_Solver:
    """DPM-Solver sampler for diffusion models.

    It wraps the model and noise schedule and provides singlestep,
    multistep and adaptive update steps plus unified sample/inverse
    routines.
    """

    def __init__(
        self,
        model_fn,
        noise_schedule,
        algorithm_type="dpmsolver++",
        correcting_x0_fn=None,
        correcting_xt_fn=None,
        thresholding_max_val=1.0,
        dynamic_thresholding_ratio=0.995,
    ):
        """Construct a DPM-Solver.

        We support both DPM-Solver (``algorithm_type="dpmsolver"``) and
        DPM-Solver++ (``algorithm_type="dpmsolver++"``), plus the "dynamic
        thresholding" method in Imagen.
        """
        self.dit_model = model_fn
        self.noise_schedule = noise_schedule
        assert algorithm_type in ["dpmsolver", "dpmsolver++"]
        self.algorithm_type = algorithm_type
        if correcting_x0_fn == "dynamic_thresholding":
            self.correcting_x0_fn = self.dynamic_thresholding_fn
        else:
            self.correcting_x0_fn = correcting_x0_fn
        self.correcting_xt_fn = correcting_xt_fn
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.thresholding_max_val = thresholding_max_val

    def dynamic_thresholding_fn(self, x0, t):
        """Apply dynamic thresholding to the predicted x0.

        Args:
            x0: Predicted clean data tensor.
            t: Timestep tensor (unused, kept for corrector interface).

        Returns:
            Thresholded and rescaled x0 tensor.
        """
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(
            torch.maximum(
                s, self.thresholding_max_val * torch.ones_like(s).to(s.device)
            ),
            dims,
        )
        x0 = torch.clamp(x0, -s, s) / s
        return x0

    def noise_prediction_fn(
        self,
        x,
        t,
        model_kwargs,
        classifier_kwargs=None,
        condition=None,
        step: int = 0,
    ):
        """Return the noise prediction model."""
        model = (
            self.dit_model[step]
            if isinstance(self.dit_model, nn.ModuleList)
            else self.dit_model
        )
        return model(x, t, model_kwargs, classifier_kwargs, condition, step)

    def data_prediction_fn(
        self,
        x,
        t,
        model_kwargs,
        classifier_kwargs=None,
        condition=None,
        step: int = 0,
    ):
        """Return the data prediction model (with corrector)."""
        noise = self.noise_prediction_fn(
            x, t, model_kwargs, classifier_kwargs, condition, step
        )
        alpha_t, sigma_t = (
            self.noise_schedule.marginal_alpha(t),
            self.noise_schedule.marginal_std(t),
        )
        x0 = (x[..., :324] - sigma_t * noise) / alpha_t
        if self.correcting_x0_fn is not None:
            x0 = self.correcting_x0_fn(x0, t)
        return x0

    def model_fn(
        self,
        x,
        t,
        model_kwargs,
        classifier_kwargs=None,
        condition=None,
        step: int = 0,
    ):
        """Convert the model to the noise / data prediction model."""
        if self.algorithm_type == "dpmsolver++":
            return self.data_prediction_fn(
                x, t, model_kwargs, classifier_kwargs, condition, step
            )
        else:
            return self.noise_prediction_fn(
                x, t, model_kwargs, classifier_kwargs, condition, step
            )

    def get_time_steps(self, skip_type, t_T, t_0, n, device):
        """Compute the intermediate time steps for sampling."""
        if skip_type == "logSNR":
            lambda_T = self.noise_schedule.marginal_lambda(
                torch.tensor(t_T).to(device)
            )
            lambda_0 = self.noise_schedule.marginal_lambda(
                torch.tensor(t_0).to(device)
            )
            log_snr_steps = torch.linspace(
                lambda_T.cpu().item(), lambda_0.cpu().item(), n + 1
            ).to(device)
            return self.noise_schedule.inverse_lambda(log_snr_steps)

        elif skip_type == "time_uniform":
            return torch.linspace(t_T, t_0, n + 1).to(device)
        elif skip_type == "time_quadratic":
            t_order = 2
            t = (
                torch.linspace(
                    t_T ** (1.0 / t_order), t_0 ** (1.0 / t_order), n + 1
                )
                .pow(t_order)
                .to(device)
            )
            return t
        else:
            raise ValueError(
                "Unsupported skip_type {}, need to be 'logSNR' or "
                "'time_uniform' or 'time_quadratic'".format(skip_type)
            )

    def get_orders_and_timesteps_for_singlestep_solver(
        self, steps, order, skip_type, t_T, t_0, device
    ):
        """Get the order of each step for the singlestep DPM-Solver."""
        if order == 3:
            k = steps // 3 + 1
            if steps % 3 == 0:
                orders = [3] * (k - 2) + [2, 1]
            elif steps % 3 == 1:
                orders = [3] * (k - 1) + [1]
            else:
                orders = [3] * (k - 1) + [2]
        elif order == 2:
            if steps % 2 == 0:
                k = steps // 2
                orders = [2] * k
            else:
                k = steps // 2 + 1
                orders = [2] * (k - 1) + [1]
        elif order == 1:
            k = steps
            orders = [1] * steps
        else:
            raise ValueError("'order' must be '1' or '2' or '3'.")
        if skip_type == "logSNR":
            timesteps_outer = self.get_time_steps(
                skip_type, t_T, t_0, k, device
            )
        else:
            timesteps_outer = self.get_time_steps(
                skip_type, t_T, t_0, steps, device
            )[torch.cumsum(torch.tensor([0] + orders), 0).to(device)]
        return timesteps_outer, orders

    def denoise_to_zero_fn(
        self,
        x,
        s,
        model_kwargs,
        classifier_kwargs=None,
        condition=None,
        step: int = 10,
    ):
        """Denoise at the final step (first-order discretization to infty)."""
        return self.data_prediction_fn(
            x, s, model_kwargs, classifier_kwargs, condition, step
        )

    def dpm_solver_first_update(
        self, x, s, t, model_s=None, return_intermediate=False
    ):
        """DPM-Solver-1 (equivalent to DDIM) from time `s` to time `t`."""
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = (
            ns.marginal_log_mean_coeff(s),
            ns.marginal_log_mean_coeff(t),
        )
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = sigma_t / sigma_s * x - alpha_t * phi_1 * model_s
            if return_intermediate:
                return x_t, {"model_s": model_s}
            else:
                return x_t
        else:
            phi_1 = torch.expm1(h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = (
                torch.exp(log_alpha_t - log_alpha_s) * x
                - (sigma_t * phi_1) * model_s
            )
            if return_intermediate:
                return x_t, {"model_s": model_s}
            else:
                return x_t

    def singlestep_dpm_solver_second_update(
        self,
        x,
        s,
        t,
        r1=0.5,
        model_s=None,
        return_intermediate=False,
        solver_type="dpmsolver",
    ):
        """Singlestep solver DPM-Solver-2 from time `s` to time `t`."""
        if solver_type not in ["dpmsolver", "taylor"]:
            raise ValueError(
                "'solver_type' must be either 'dpmsolver' or 'taylor', "
                "got {}".format(solver_type)
            )
        if r1 is None:
            r1 = 0.5
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        s1 = ns.inverse_lambda(lambda_s1)
        log_alpha_s, log_alpha_s1, log_alpha_t = (
            ns.marginal_log_mean_coeff(s),
            ns.marginal_log_mean_coeff(s1),
            ns.marginal_log_mean_coeff(t),
        )
        sigma_s, sigma_s1, sigma_t = (
            ns.marginal_std(s),
            ns.marginal_std(s1),
            ns.marginal_std(t),
        )
        alpha_s1, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_11 = torch.expm1(-r1 * h)
            phi_1 = torch.expm1(-h)

            if model_s is None:
                model_s = self.model_fn(x, s)
            x_s1 = (sigma_s1 / sigma_s) * x - (alpha_s1 * phi_11) * model_s
            model_s1 = self.model_fn(x_s1, s1)
            if solver_type == "dpmsolver":
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    - (0.5 / r1) * (alpha_t * phi_1) * (model_s1 - model_s)
                )
            elif solver_type == "taylor":
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    + (1.0 / r1)
                    * (alpha_t * (phi_1 / h + 1.0))
                    * (model_s1 - model_s)
                )
        else:
            phi_11 = torch.expm1(r1 * h)
            phi_1 = torch.expm1(h)

            if model_s is None:
                model_s = self.model_fn(x, s)
            x_s1 = (
                torch.exp(log_alpha_s1 - log_alpha_s) * x
                - (sigma_s1 * phi_11) * model_s
            )
            model_s1 = self.model_fn(x_s1, s1)
            if solver_type == "dpmsolver":
                x_t = (
                    torch.exp(log_alpha_t - log_alpha_s) * x
                    - (sigma_t * phi_1) * model_s
                    - (0.5 / r1) * (sigma_t * phi_1) * (model_s1 - model_s)
                )
            elif solver_type == "taylor":
                x_t = (
                    torch.exp(log_alpha_t - log_alpha_s) * x
                    - (sigma_t * phi_1) * model_s
                    - (1.0 / r1)
                    * (sigma_t * (phi_1 / h - 1.0))
                    * (model_s1 - model_s)
                )
        if return_intermediate:
            return x_t, {"model_s": model_s, "model_s1": model_s1}
        else:
            return x_t

    def singlestep_dpm_solver_third_update(
        self,
        x,
        s,
        t,
        r1=1.0 / 3.0,
        r2=2.0 / 3.0,
        model_s=None,
        model_s1=None,
        return_intermediate=False,
        solver_type="dpmsolver",
    ):
        """Singlestep solver DPM-Solver-3 from time `s` to time `t`."""
        if solver_type not in ["dpmsolver", "taylor"]:
            raise ValueError(
                "'solver_type' must be either 'dpmsolver' or 'taylor', "
                "got {}".format(solver_type)
            )
        if r1 is None:
            r1 = 1.0 / 3.0
        if r2 is None:
            r2 = 2.0 / 3.0
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        lambda_s2 = lambda_s + r2 * h
        s1 = ns.inverse_lambda(lambda_s1)
        s2 = ns.inverse_lambda(lambda_s2)
        log_alpha_s, log_alpha_s1, log_alpha_s2, log_alpha_t = (
            ns.marginal_log_mean_coeff(s),
            ns.marginal_log_mean_coeff(s1),
            ns.marginal_log_mean_coeff(s2),
            ns.marginal_log_mean_coeff(t),
        )
        sigma_s, sigma_s1, sigma_s2, sigma_t = (
            ns.marginal_std(s),
            ns.marginal_std(s1),
            ns.marginal_std(s2),
            ns.marginal_std(t),
        )
        alpha_s1, alpha_s2, alpha_t = (
            torch.exp(log_alpha_s1),
            torch.exp(log_alpha_s2),
            torch.exp(log_alpha_t),
        )

        if self.algorithm_type == "dpmsolver++":
            phi_11 = torch.expm1(-r1 * h)
            phi_12 = torch.expm1(-r2 * h)
            phi_1 = torch.expm1(-h)
            phi_22 = torch.expm1(-r2 * h) / (r2 * h) + 1.0
            phi_2 = phi_1 / h + 1.0
            phi_3 = phi_2 / h - 0.5

            if model_s is None:
                model_s = self.model_fn(x, s)
            if model_s1 is None:
                x_s1 = (sigma_s1 / sigma_s) * x - (alpha_s1 * phi_11) * model_s
                model_s1 = self.model_fn(x_s1, s1)
            x_s2 = (
                (sigma_s2 / sigma_s) * x
                - (alpha_s2 * phi_12) * model_s
                + r2 / r1 * (alpha_s2 * phi_22) * (model_s1 - model_s)
            )
            model_s2 = self.model_fn(x_s2, s2)
            if solver_type == "dpmsolver":
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    + (1.0 / r2) * (alpha_t * phi_2) * (model_s2 - model_s)
                )
            elif solver_type == "taylor":
                d1_0 = (1.0 / r1) * (model_s1 - model_s)
                d1_1 = (1.0 / r2) * (model_s2 - model_s)
                d1 = (r2 * d1_0 - r1 * d1_1) / (r2 - r1)
                d2 = 2.0 * (d1_1 - d1_0) / (r2 - r1)
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    + (alpha_t * phi_2) * d1
                    - (alpha_t * phi_3) * d2
                )
        else:
            phi_11 = torch.expm1(r1 * h)
            phi_12 = torch.expm1(r2 * h)
            phi_1 = torch.expm1(h)
            phi_22 = torch.expm1(r2 * h) / (r2 * h) - 1.0
            phi_2 = phi_1 / h - 1.0
            phi_3 = phi_2 / h - 0.5

            if model_s is None:
                model_s = self.model_fn(x, s)
            if model_s1 is None:
                x_s1 = (
                    torch.exp(log_alpha_s1 - log_alpha_s) * x
                    - (sigma_s1 * phi_11) * model_s
                )
                model_s1 = self.model_fn(x_s1, s1)
            x_s2 = (
                torch.exp(log_alpha_s2 - log_alpha_s) * x
                - (sigma_s2 * phi_12) * model_s
                - r2 / r1 * (sigma_s2 * phi_22) * (model_s1 - model_s)
            )
            model_s2 = self.model_fn(x_s2, s2)
            if solver_type == "dpmsolver":
                x_t = (
                    torch.exp(log_alpha_t - log_alpha_s) * x
                    - (sigma_t * phi_1) * model_s
                    - (1.0 / r2) * (sigma_t * phi_2) * (model_s2 - model_s)
                )
            elif solver_type == "taylor":
                d1_0 = (1.0 / r1) * (model_s1 - model_s)
                d1_1 = (1.0 / r2) * (model_s2 - model_s)
                d1 = (r2 * d1_0 - r1 * d1_1) / (r2 - r1)
                d2 = 2.0 * (d1_1 - d1_0) / (r2 - r1)
                x_t = (
                    torch.exp(log_alpha_t - log_alpha_s) * x
                    - (sigma_t * phi_1) * model_s
                    - (sigma_t * phi_2) * d1
                    - (sigma_t * phi_3) * d2
                )

        if return_intermediate:
            return x_t, {
                "model_s": model_s,
                "model_s1": model_s1,
                "model_s2": model_s2,
            }
        else:
            return x_t

    def multistep_dpm_solver_second_update(
        self, x, model_prev_list, t_prev_list, t, solver_type="dpmsolver"
    ):
        """Multistep solver DPM-Solver-2."""
        if solver_type not in ["dpmsolver", "taylor"]:
            raise ValueError(
                "'solver_type' must be either 'dpmsolver' or 'taylor', "
                "got {}".format(solver_type)
            )
        ns = self.noise_schedule
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]
        lambda_prev_1, lambda_prev_0, lambda_t = (
            ns.marginal_lambda(t_prev_1),
            ns.marginal_lambda(t_prev_0),
            ns.marginal_lambda(t),
        )
        log_alpha_prev_0, log_alpha_t = (
            ns.marginal_log_mean_coeff(t_prev_0),
            ns.marginal_log_mean_coeff(t),
        )
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        d1_0 = (1.0 / r0) * (model_prev_0 - model_prev_1)
        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            if solver_type == "dpmsolver":
                x_t = (
                    (sigma_t / sigma_prev_0) * x
                    - (alpha_t * phi_1) * model_prev_0
                    - 0.5 * (alpha_t * phi_1) * d1_0
                )
            elif solver_type == "taylor":
                x_t = (
                    (sigma_t / sigma_prev_0) * x
                    - (alpha_t * phi_1) * model_prev_0
                    + (alpha_t * (phi_1 / h + 1.0)) * d1_0
                )
        else:
            phi_1 = torch.expm1(h)
            if solver_type == "dpmsolver":
                x_t = (
                    torch.exp(log_alpha_t - log_alpha_prev_0) * x
                    - (sigma_t * phi_1) * model_prev_0
                    - 0.5 * (sigma_t * phi_1) * d1_0
                )
            elif solver_type == "taylor":
                x_t = (
                    torch.exp(log_alpha_t - log_alpha_prev_0) * x
                    - (sigma_t * phi_1) * model_prev_0
                    - (sigma_t * (phi_1 / h - 1.0)) * d1_0
                )
        return x_t

    def multistep_dpm_solver_third_update(
        self, x, model_prev_list, t_prev_list, t, solver_type="dpmsolver"
    ):
        """Multistep solver DPM-Solver-3."""
        ns = self.noise_schedule
        (
            model_prev_2,
            model_prev_1,
            model_prev_0,
        ) = model_prev_list
        t_prev_2, t_prev_1, t_prev_0 = t_prev_list
        lambda_prev_2, lambda_prev_1, lambda_prev_0, lambda_t = (
            ns.marginal_lambda(t_prev_2),
            ns.marginal_lambda(t_prev_1),
            ns.marginal_lambda(t_prev_0),
            ns.marginal_lambda(t),
        )
        log_alpha_prev_0, log_alpha_t = (
            ns.marginal_log_mean_coeff(t_prev_0),
            ns.marginal_log_mean_coeff(t),
        )
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)
        h_1 = lambda_prev_1 - lambda_prev_2
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0, r1 = h_0 / h, h_1 / h
        d1_0 = (1.0 / r0) * (model_prev_0 - model_prev_1)
        d1_1 = (1.0 / r1) * (model_prev_1 - model_prev_2)
        d1 = d1_0 + (r0 / (r0 + r1)) * (d1_0 - d1_1)
        d2 = (1.0 / (r0 + r1)) * (d1_0 - d1_1)
        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            phi_2 = phi_1 / h + 1.0
            phi_3 = phi_2 / h - 0.5
            x_t = (
                (sigma_t / sigma_prev_0) * x
                - (alpha_t * phi_1) * model_prev_0
                + (alpha_t * phi_2) * d1
                - (alpha_t * phi_3) * d2
            )
        else:
            phi_1 = torch.expm1(h)
            phi_2 = phi_1 / h - 1.0
            phi_3 = phi_2 / h - 0.5
            x_t = (
                torch.exp(log_alpha_t - log_alpha_prev_0) * x
                - (sigma_t * phi_1) * model_prev_0
                - (sigma_t * phi_2) * d1
                - (sigma_t * phi_3) * d2
            )
        return x_t

    def singlestep_dpm_solver_update(
        self,
        x,
        s,
        t,
        order,
        return_intermediate=False,
        solver_type="dpmsolver",
        r1=None,
        r2=None,
    ):
        """Singlestep DPM-Solver with the order `order` from `s` to `t`."""
        if order == 1:
            return self.dpm_solver_first_update(
                x, s, t, return_intermediate=return_intermediate
            )
        elif order == 2:
            return self.singlestep_dpm_solver_second_update(
                x,
                s,
                t,
                return_intermediate=return_intermediate,
                solver_type=solver_type,
                r1=r1,
            )
        elif order == 3:
            return self.singlestep_dpm_solver_third_update(
                x,
                s,
                t,
                return_intermediate=return_intermediate,
                solver_type=solver_type,
                r1=r1,
                r2=r2,
            )
        else:
            raise ValueError(
                "Solver order must be 1 or 2 or 3, got {}".format(order)
            )

    def multistep_dpm_solver_update(
        self,
        x,
        model_prev_list,
        t_prev_list,
        t,
        order,
        solver_type="dpmsolver",
    ):
        """Multistep DPM-Solver with the order `order`."""
        if order == 1:
            return self.dpm_solver_first_update(
                x, t_prev_list[-1], t, model_s=model_prev_list[-1]
            )
        elif order == 2:
            return self.multistep_dpm_solver_second_update(
                x, model_prev_list, t_prev_list, t, solver_type=solver_type
            )
        elif order == 3:
            return self.multistep_dpm_solver_third_update(
                x, model_prev_list, t_prev_list, t, solver_type=solver_type
            )
        else:
            raise ValueError(
                "Solver order must be 1 or 2 or 3, got {}".format(order)
            )

    def dpm_solver_adaptive(
        self,
        x,
        order,
        t_T,
        t_0,
        h_init=0.05,
        atol=0.0078,
        rtol=0.05,
        theta=0.9,
        t_err=1e-5,
        solver_type="dpmsolver",
    ):
        """Adaptive step size solver based on singlestep DPM-Solver."""
        ns = self.noise_schedule
        s = t_T * torch.ones((1,)).to(x)
        lambda_s = ns.marginal_lambda(s)
        lambda_0 = ns.marginal_lambda(t_0 * torch.ones_like(s).to(x))
        h = h_init * torch.ones_like(s).to(x)
        x_prev = x
        nfe = 0
        if order == 2:
            r1 = 0.5
            lower_update = lambda x, s, t: self.dpm_solver_first_update(
                x, s, t, return_intermediate=True
            )
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_second_update(  # noqa
                x, s, t, r1=r1, solver_type=solver_type, **kwargs
            )
        elif order == 3:
            r1, r2 = 1.0 / 3.0, 2.0 / 3.0
            lower_update = lambda x, s, t: self.singlestep_dpm_solver_second_update(  # noqa
                x,
                s,
                t,
                r1=r1,
                return_intermediate=True,
                solver_type=solver_type,
            )
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_third_update(  # noqa
                x, s, t, r1=r1, r2=r2, solver_type=solver_type, **kwargs
            )
        else:
            raise ValueError(
                "For adaptive step size solver, order must be 2 or 3, "
                "got {}".format(order)
            )
        while torch.abs((s - t_0)).mean() > t_err:
            t = ns.inverse_lambda(lambda_s + h)
            x_lower, lower_noise_kwargs = lower_update(x, s, t)
            x_higher = higher_update(x, s, t, **lower_noise_kwargs)
            delta = torch.max(
                torch.ones_like(x).to(x) * atol,
                rtol * torch.max(torch.abs(x_lower), torch.abs(x_prev)),
            )
            norm_fn = lambda v: torch.sqrt(  # noqa: E731
                torch.square(v.reshape((v.shape[0], -1))).mean(
                    dim=-1, keepdim=True
                )
            )
            e = norm_fn((x_higher - x_lower) / delta).max()
            if torch.all(e <= 1.0):
                x = x_higher
                s = t
                x_prev = x_lower
                lambda_s = ns.marginal_lambda(s)
            h = torch.min(
                theta * h * torch.float_power(e, -1.0 / order).float(),
                lambda_0 - lambda_s,
            )
            nfe += order
        print("adaptive solver nfe", nfe)
        return x

    def add_noise(self, x, t, noise=None):
        """Compute the noised input xt = alpha_t * x + sigma_t * noise."""
        alpha_t, sigma_t = (
            self.noise_schedule.marginal_alpha(t),
            self.noise_schedule.marginal_std(t),
        )
        if noise is None:
            noise = torch.randn((t.shape[0], *x.shape), device=x.device)
        x = x.reshape((-1, *x.shape))
        xt = (
            expand_dims(alpha_t, x.dim()) * x
            + expand_dims(sigma_t, x.dim()) * noise
        )
        if t.shape[0] == 1:
            return xt.squeeze(0)
        else:
            return xt

    def inverse(
        self,
        x,
        steps=20,
        t_start=None,
        t_end=None,
        order=2,
        skip_type="time_uniform",
        method="multistep",
        lower_order_final=True,
        denoise_to_zero=False,
        solver_type="dpmsolver",
        atol=0.0078,
        rtol=0.05,
        return_intermediate=False,
    ):
        """Inverse the sample `x` from time `t_start` to `t_end`."""
        t_0 = 1.0 / self.noise_schedule.total_N if t_start is None else t_start
        t_T = self.noise_schedule.T if t_end is None else t_end
        assert t_0 > 0 and t_T > 0
        return self.sample(
            x,
            steps=steps,
            t_start=t_0,
            t_end=t_T,
            order=order,
            skip_type=skip_type,
            method=method,
            lower_order_final=lower_order_final,
            denoise_to_zero=denoise_to_zero,
            solver_type=solver_type,
            atol=atol,
            rtol=rtol,
            return_intermediate=return_intermediate,
        )

    def sample(
        self,
        x,
        steps=20,
        t_start=None,
        t_end=None,
        order=2,
        skip_type="time_uniform",
        method="multistep",
        lower_order_final=True,
        denoise_to_zero=False,
        solver_type="dpmsolver",
        atol=0.0078,
        rtol=0.05,
        return_intermediate=False,
    ):
        """Compute the sample at time `t_end` by DPM-Solver."""
        t_0 = 1.0 / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert t_0 > 0 and t_T > 0
        if return_intermediate:
            assert method in [
                "multistep",
                "singlestep",
                "singlestep_fixed",
            ], "Cannot use adaptive solver when saving intermediate values"
        if self.correcting_xt_fn is not None:
            assert method in [
                "multistep",
                "singlestep",
                "singlestep_fixed",
            ], "Cannot use adaptive solver when correcting_xt_fn is not None"
        device = x.device
        intermediates = []
        with torch.no_grad():
            if method == "adaptive":
                x = self.dpm_solver_adaptive(
                    x,
                    order=order,
                    t_T=t_T,
                    t_0=t_0,
                    atol=atol,
                    rtol=rtol,
                    solver_type=solver_type,
                )
            elif method == "multistep":
                assert steps >= order
                timesteps = self.get_time_steps(
                    skip_type=skip_type,
                    t_T=t_T,
                    t_0=t_0,
                    n=steps,
                    device=device,
                )
                assert timesteps.shape[0] - 1 == steps
                step = 0
                t = timesteps[step]
                t_prev_list = [t]
                model_prev_list = [self.model_fn(x, t)]
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step)
                if return_intermediate:
                    intermediates.append(x)
                for step in range(1, order):
                    t = timesteps[step]
                    x = self.multistep_dpm_solver_update(
                        x[..., :324],
                        model_prev_list,
                        t_prev_list,
                        t,
                        step,
                        solver_type=solver_type,
                    )
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                    t_prev_list.append(t)
                    model_prev_list.append(self.model_fn(x, t))
                for step in range(order, steps + 1):
                    t = timesteps[step]
                    if lower_order_final and steps < 10:
                        step_order = min(order, steps + 1 - step)
                    else:
                        step_order = order
                    x = self.multistep_dpm_solver_update(
                        x[..., :324],
                        model_prev_list,
                        t_prev_list,
                        t,
                        step_order,
                        solver_type=solver_type,
                    )
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
                    for i in range(order - 1):
                        t_prev_list[i] = t_prev_list[i + 1]
                        model_prev_list[i] = model_prev_list[i + 1]
                    t_prev_list[-1] = t
                    if step < steps:
                        model_prev_list[-1] = self.model_fn(x, t)
            elif method in ["singlestep", "singlestep_fixed"]:
                if method == "singlestep":
                    (
                        timesteps_outer,
                        orders,
                    ) = self.get_orders_and_timesteps_for_singlestep_solver(
                        steps=steps,
                        order=order,
                        skip_type=skip_type,
                        t_T=t_T,
                        t_0=t_0,
                        device=device,
                    )
                elif method == "singlestep_fixed":
                    k = steps // order
                    orders = [order] * k
                    timesteps_outer = self.get_time_steps(
                        skip_type=skip_type,
                        t_T=t_T,
                        t_0=t_0,
                        n=k,
                        device=device,
                    )
                for step, order in enumerate(orders):
                    s, t = timesteps_outer[step], timesteps_outer[step + 1]
                    timesteps_inner = self.get_time_steps(
                        skip_type=skip_type,
                        t_T=s.item(),
                        t_0=t.item(),
                        n=order,
                        device=device,
                    )
                    lambda_inner = self.noise_schedule.marginal_lambda(
                        timesteps_inner
                    )
                    h = lambda_inner[-1] - lambda_inner[0]
                    r1 = (
                        None
                        if order <= 1
                        else (lambda_inner[1] - lambda_inner[0]) / h
                    )
                    r2 = (
                        None
                        if order <= 2
                        else (lambda_inner[2] - lambda_inner[0]) / h
                    )
                    x = self.singlestep_dpm_solver_update(
                        x, s, t, order, solver_type=solver_type, r1=r1, r2=r2
                    )
                    if self.correcting_xt_fn is not None:
                        x = self.correcting_xt_fn(x, t, step)
                    if return_intermediate:
                        intermediates.append(x)
            else:
                raise ValueError("Got wrong method {}".format(method))
            if denoise_to_zero:
                t = torch.ones((1,)).to(device) * t_0
                x = self.denoise_to_zero_fn(x, t)
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step + 1)
                if return_intermediate:
                    intermediates.append(x)
        if return_intermediate:
            return x, intermediates
        else:
            return x[..., :324]
