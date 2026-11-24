# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import abc

import torch
from hat_engine.registry import OBJECT_REGISTRY

STD_MIN = 1e-6


@OBJECT_REGISTRY.register
class SDE(abc.ABC):
    # SDE abstract class. Functions are designed for a mini-batch of inputs.

    def __init__(self):
        """Construct an SDE."""
        super().__init__()

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, t):
        """Return the drift and diffusion coefficients of the SDE.

        return (drift $f(x,t)$, diffusion $g(t)$)
        """
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        """Parameters to determine the marginal distribution of the SDE.

        return mean, std
        """
        pass

    @abc.abstractmethod
    def diffusion_coeff(self, t):
        """Return the diffusion coefficient of the SDE.

        return $g(t)$
        """
        pass

    @abc.abstractmethod
    def marginal_prob_std(self, t):
        """Parameters to determine the marginal distribution of the SDE.

        return std
        """
        pass


@OBJECT_REGISTRY.register
class VPSDE_linear(SDE):
    def __init__(self, beta_max=20.0, beta_min=0.1):
        r"""VP SDE.

        SDE:
        $ \\mathrm{d}x = -\\frac{\\beta(t)}{2} x \\mathrm{d}t
        + \\sqrt{\\beta(t)} \\mathrm{d}W_t $
        """
        super().__init__()

        self._beta_max = beta_max
        self._beta_min = beta_min

    @property
    def T(self):
        return 1.0

    def sde(self, x, t):
        r"""SDE of diffusion process.

        drift = $-\\frac{\\beta(t)}{2} x$
        diffusion = $\\sqrt{\\beta(t)}$
        """
        shape = x.shape
        reshape = [-1] + [1] * (len(shape) - 1)
        t = t.reshape(reshape)

        beta_t = (self._beta_max - self._beta_min) * t + self._beta_min
        drift = -0.5 * beta_t * x
        diffusion = torch.sqrt(beta_t)

        return drift, diffusion

    def marginal_prob(self, x, t):
        """Parameters to determine the marginal distribution of the SDE."""
        shape = x.shape
        reshape = [-1] + [1] * (len(shape) - 1)
        t = t.reshape(reshape)
        # 即log(alpha_t)
        mean_log_coeff = (
            -0.25 * t**2 * (self._beta_max - self._beta_min)
            - 0.5 * self._beta_min * t
        )
        # 即alpha_t*x
        mean = torch.exp(mean_log_coeff) * x
        # alpha_t^2+sigma_t^2=1
        std = torch.sqrt(1 - torch.exp(2.0 * mean_log_coeff))
        return mean, std

    def diffusion_coeff(self, t):
        beta_t = (self._beta_max - self._beta_min) * t + self._beta_min
        diffusion = torch.sqrt(beta_t)
        return diffusion

    def marginal_prob_std(self, t):
        discount = torch.exp(
            -0.5 * t**2 * (self._beta_max - self._beta_min)
            - self._beta_min * t
        )
        std = torch.sqrt(1 - discount)
        return std


@OBJECT_REGISTRY.register
class subVPSDE_exp(SDE):
    def __init__(self, sigma=25.0):
        """subVPSDE.

        $beta(t) = sigma^t$
        """
        raise NotImplementedError
