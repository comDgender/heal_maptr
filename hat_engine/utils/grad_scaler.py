# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from collections.abc import Iterable
from typing import Any, Optional, Union

import torch
from torch.cuda.amp import GradScaler

from hat_engine.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class LossSafeGradScaler(GradScaler):
    def __init__(
        self,
        init_scale: float = 2.0 ** 16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
        enabled: bool = True,
    ) -> None:
        super(LossSafeGradScaler, self).__init__(
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            enabled=enabled,
        )
        self.skip_step = False

    def scale(
        self,
        output: Union[torch.Tensor],
    ) -> Union[torch.Tensor, Iterable[torch.Tensor]]:
        if torch.isinf(output).any() or torch.isnan(output).any():
            self.skip_step = True
        elif (
            torch.isinf(output * self._scale).any()
            or torch.isnan(output * self._scale).any()
        ):
            tmp_output = output * self._scale
            while (
                torch.isinf(tmp_output).any() or torch.isnan(tmp_output).any()
            ):
                self._scale *= self._backoff_factor
        else:
            return super(LossSafeGradScaler, self).scale(output)

    def step(
        self, optimizer: torch.optim.Optimizer, *args: Any, **kwargs: Any
    ) -> Optional[float]:
        if self.skip_step:
            return None
        else:
            return super(LossSafeGradScaler, self).step(
                optimizer, *args, **kwargs
            )

    def update(self) -> None:
        if not self.skip_step:
            super(LossSafeGradScaler, self).update()
        else:
            self.skip_step = False
