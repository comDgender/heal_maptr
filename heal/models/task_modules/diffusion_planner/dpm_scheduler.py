# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import torch
import torch.nn as nn

from heal.models.task_modules.diffusion_planner.dpm_solver_core import (
    DPM_Solver,
)


class DPMSolverScheduler(nn.Module, DPM_Solver):
    """Quantifiable nn.Module wrapper around the DPM_Solver sampling loop."""

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
        """Initialize the scheduler by combining nn.Module and DPM_Solver init.

        Args:
            model_fn: Denoising model callable invoked as
                model_fn(x, t, **kwargs) returning noise/x0 prediction.
            noise_schedule: Noise schedule providing total_N, T and
                marginal_lambda.
            algorithm_type: DPM-Solver algorithm type, e.g. "dpmsolver++".
            correcting_x0_fn: Optional x0 correcting function or None.
            correcting_xt_fn: Optional xt correcting function or None.
            thresholding_max_val: Max value for thresholding options.
            dynamic_thresholding_ratio: Ratio for dynamic thresholding.
        """
        nn.Module.__init__(self)
        DPM_Solver.__init__(
            self,
            model_fn=model_fn,
            noise_schedule=noise_schedule,
            algorithm_type=algorithm_type,
            correcting_x0_fn=correcting_x0_fn,
            correcting_xt_fn=correcting_xt_fn,
            thresholding_max_val=thresholding_max_val,
            dynamic_thresholding_ratio=dynamic_thresholding_ratio,
        )

    def forward(
        self,
        x,
        model_condition,
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
        condition=None,
        classifier_kwargs=None,
    ):
        """Compute sample at time `t_end` by running the DPM-Solver loop.

        Args:
            x: Initial noisy sample tensor at time `t_start`.
            model_condition: Conditioning dict passed to model_fn via
                model_kwargs.
            steps: Number of discrete solver steps.
            t_start: Start time of sampling or None to use noise_schedule.T.
            t_end: End time of sampling or None to use
                1.0 / noise_schedule.total_N.
            order: DPM-Solver order for step updates.
            skip_type: Time discretization type for get_time_steps.
            method: Solver loop among "adaptive", "multistep",
                "singlestep", "singlestep_fixed".
            lower_order_final: Whether to reduce order in the last steps
                when steps < 10.
            denoise_to_zero: Whether to append an extra denoise step at t_0.
            solver_type: Internal solver variant, e.g. "dpmsolver".
            atol: Absolute tolerance for the adaptive solver.
            rtol: Relative tolerance for the adaptive solver.
            return_intermediate: Whether to also return intermediate samples.
            condition: Additional conditioning passed to model_fn.
            classifier_kwargs: Extra classifier guidance kwargs for
                model_fn.

        Returns:
            Sample tensor at time t_end, truncated to the first 324
            channels; when return_intermediate is True, a tuple of
            (final sample, list of intermediates).
        """
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
                model_prev_list = [
                    self.model_fn(
                        x,
                        t,
                        model_kwargs=model_condition,
                        classifier_kwargs=classifier_kwargs,
                        condition=condition,
                        step=step,
                    )
                ]
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
                    model_prev_list.append(
                        self.model_fn(
                            x,
                            t,
                            model_kwargs=model_condition,
                            classifier_kwargs=classifier_kwargs,
                            condition=condition,
                            step=step,
                        )
                    )
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
                        model_prev_list[-1] = self.model_fn(
                            x,
                            t,
                            model_kwargs=model_condition,
                            classifier_kwargs=classifier_kwargs,
                            condition=condition,
                            step=step,
                        )
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
                x = self.denoise_to_zero_fn(
                    x,
                    t,
                    model_kwargs=model_condition,
                    classifier_kwargs=classifier_kwargs,
                    condition=condition,
                    step=steps,
                )
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step + 1)
                if return_intermediate:
                    intermediates.append(x)
        if return_intermediate:
            return x, intermediates
        else:
            return x[..., :324]
