# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501


from typing import Optional

import torch
import torch.nn as nn
from horizon_plugin_pytorch.nn.quantized import FloatFunctional

__all__ = ["MlpModule2d", "MLP", "FFN"]


class MlpModule2d(nn.Sequential):
    """A mlp block that bundles two fc layers.

    Args:
        in_channels: Number of input channels.
        hidden_channels: Number of hidden channels.
        out_channels: Number of output channels.
        act_layer: Activation layer. Default: None.
        drop_ratio: Dropout ratio of output. Default: 0.0.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        act_layer: Optional[nn.Module] = None,
        drop_ratio: float = 0.0,
    ):
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        fc1 = nn.Linear(in_channels, hidden_channels)
        fc2 = nn.Linear(hidden_channels, out_channels)
        if drop_ratio > 0.0:
            drop1 = nn.Dropout(drop_ratio)
            drop2 = nn.Dropout(drop_ratio)
        else:
            drop1, drop2 = None, None
        layer_list = [fc1, act_layer, drop1, fc2, drop2]
        self.layer_list = [layer for layer in layer_list if layer is not None]
        super(MlpModule2d, self).__init__(*self.layer_list)


class MLP(nn.Module):
    """Stacked Linear(+ReLU) blocks without residual or dropout.

    Hidden layers use ``Linear -> ReLU``; the last layer is a plain
    ``Linear``. Depth is controlled by ``num_layers``.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ):
        super().__init__()
        self.num_layers = num_layers
        dims_in = [input_dim] + [hidden_dim] * (num_layers - 1)
        dims_out = [hidden_dim] * (num_layers - 1) + [output_dim]
        blocks = []
        for li, (din, dout) in enumerate(zip(dims_in, dims_out)):
            if li < num_layers - 1:
                blocks.append(nn.Sequential(nn.Linear(din, dout), nn.ReLU()))
            else:
                blocks.append(nn.Linear(din, dout))
        self.layers = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor):
        for block in self.layers:
            x = block(x)
        return x


class FFN(nn.Module):
    """Transformer FFN with optional residual add.

    Builds ``num_fcs`` linear stages: the first ``num_fcs - 1`` are
    ``Linear -> activation -> Dropout``, then a final
    ``Linear -> Dropout``. When ``add_identity`` is set, the residual is
    added via ``FloatFunctional``.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        feedforward_dim: int = 1024,
        output_dim: Optional[int] = None,
        num_fcs: int = 2,
        activation: Optional[nn.Module] = None,
        ffn_drop: float = 0.0,
        fc_bias: bool = True,
        add_identity: bool = True,
    ):
        super(FFN, self).__init__()
        assert num_fcs >= 2, (
            "num_fcs should be no less " f"than 2. got {num_fcs}."
        )
        self.embed_dim = embed_dim
        self.feedforward_dim = feedforward_dim
        self.num_fcs = num_fcs
        self.activation = (
            nn.ReLU(inplace=True) if activation is None else activation
        )
        out_dim = embed_dim if output_dim is None else output_dim

        stages = []
        ch = embed_dim
        for _ in range(num_fcs - 1):
            stages.append(
                nn.Sequential(
                    nn.Linear(ch, feedforward_dim, bias=fc_bias),
                    self.activation,
                    nn.Dropout(ffn_drop),
                )
            )
            ch = feedforward_dim
        stages.append(nn.Linear(feedforward_dim, out_dim, bias=fc_bias))
        stages.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*stages)
        self.add_identity = add_identity
        self.add_identity_op = FloatFunctional()

    def forward(
        self, x: torch.Tensor, identity: Optional[torch.Tensor] = None
    ):
        """Apply FFN; optionally residual-add ``identity`` (fail if None)."""
        y = self.layers(x)
        if not self.add_identity:
            return y
        return self.add_identity_op.add(identity, y)
