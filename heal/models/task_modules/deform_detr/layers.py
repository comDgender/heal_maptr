# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
import copy
import logging
from typing import List, Optional, Union

import torch
import torch.nn as nn
from hat_engine.utils.model_helpers import fx_wrap

__all__ = [
    "BaseTransformerLayer",
    "TransformerLayerSequence",
]

logger = logging.getLogger(__name__)


class BaseTransformerLayer(nn.Module):
    """Generic transformer block driven by an ``operation_order`` schedule.

    Attention / FFN / Norm modules are provided directly so callers can
    compose layers via LazyConfig. Putting ``norm`` first enables pre-norm
    (see https://arxiv.org/abs/2002.04745).

    Args:
        embed_dim: Channel size of token embeddings.
        attn: One attention module, or a list with one entry per
            ``self_attn`` / ``cross_attn`` slot in ``operation_order``.
        ffn: Feed-forward block cloned once per ``ffn`` slot.
        norm: Normalization block cloned once per ``norm`` slot.
        operation_order: Ordered ops, e.g.
            ``('self_attn', 'norm', 'ffn', 'norm')``.
    """

    def __init__(
        self,
        embed_dim: int,
        attn: Union[List[nn.Module], nn.Module],
        ffn: nn.Module,
        norm: nn.Module,
        operation_order: tuple,
    ):
        super(BaseTransformerLayer, self).__init__()
        allowed = {"self_attn", "norm", "cross_attn", "ffn"}
        assert set(operation_order).issubset(allowed)

        attn_slots = operation_order.count(
            "self_attn"
        ) + operation_order.count("cross_attn")
        if isinstance(attn, nn.Module):
            attn_mods = [copy.deepcopy(attn) for _ in range(attn_slots)]
        else:
            assert len(attn) == attn_slots, (
                f"attn list length {len(attn)} != "
                f"attention count {attn_slots} in {operation_order}"
            )
            attn_mods = list(attn)

        self.num_attn = attn_slots
        self.operation_order = operation_order
        self.pre_norm = operation_order[0] == "norm"
        # Build attention ModuleList in operation_order sequence.
        self.attentions = nn.ModuleList()
        attn_i = 0
        for name in operation_order:
            if name in ("self_attn", "cross_attn"):
                self.attentions.append(attn_mods[attn_i])
                attn_i += 1

        self.embed_dim = embed_dim
        self.ffns = nn.ModuleList(
            [copy.deepcopy(ffn) for _ in range(operation_order.count("ffn"))]
        )
        self.norms = nn.ModuleList(
            [copy.deepcopy(norm) for _ in range(operation_order.count("norm"))]
        )

        self.attn_types = []
        attn_idx = 0
        for layer in self.operation_order:
            if layer[-4:] == "attn":
                name = self.attentions[attn_idx].__class__.__name__
                print("attn name", name)
                self.attn_types.append(name)
                attn_idx += 1

    @fx_wrap()
    def check_attn_len(self, attn_masks):
        assert len(attn_masks) == self.num_attn, (
            f"The length of "
            f"attn_masks {len(attn_masks)} must be equal "
            f"to the number of attention in "
            f"operation_order {self.num_attn}"
        )

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_pos: Optional[torch.Tensor] = None,
        attn_masks: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        query_key_padding_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
    ):
        """Forward function for `BaseTransformerLayer`.

        Args:
            query: Query embeddings with shape
                `(num_query, bs, embed_dim)` or `(bs, num_query, embed_dim)`
                which should be specified follows the attention module used in
                `BaseTransformerLayer`.
            key: Key embeddings used in `Attention`.
            value: Value embeddings with the same shape as `key`.
            query_pos: The position embedding for `query`.
            key_pos: The position embedding for `key`.
            attn_masks: A list of 2D ByteTensor used
                in calculation the corresponding attention. The length of
                `attn_masks` should be equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask: ByteTensor for `query`, with
                shape `(bs, num_query)`. Only used in `self_attn` layer.
            key_padding_mask: ByteTensor for `key`, with shape `(bs, num_key)`.
            reference_points: For multiscale deform attention
            spatial_shapes: For multiscale deform attention
            level_start_index: For multiscale deform attention
        """
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            logger.warning(
                f"Use same attn_mask in all attentions in "
                f"{self.__class__.__name__} "
            )
        else:
            self.check_attn_len(attn_masks)

        for layer in self.operation_order:
            if layer[-4:] == "attn":
                new_kwargs = {}  # copy.deepcopy(kwargs)
                if self.attn_types[attn_index] == "MultiheadAttention":
                    new_kwargs["attn_mask"] = attn_masks[attn_index]
                    new_kwargs["key_pos"] = (
                        key_pos if layer == "cross_attn" else query_pos
                    )
                    if layer == "self_attn":
                        new_kwargs["key_padding_mask"] = query_key_padding_mask
                    else:
                        new_kwargs["key_padding_mask"] = key_padding_mask
                elif (
                    self.attn_types[attn_index]
                    == "MultiScaleDeformableAttention"
                ):
                    new_kwargs["spatial_shapes"] = spatial_shapes
                    new_kwargs["reference_points"] = reference_points

            if layer == "self_attn":
                # Self-attn: Q/K/V all from the current query stream.
                query = self.attentions[attn_index](
                    query,
                    query,
                    query,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    **new_kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    **new_kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "ffn":
                residual = identity if self.pre_norm else query
                query = self.ffns[ffn_index](query, residual)
                ffn_index += 1

        return query


class TransformerLayerSequence(nn.Module):
    """Stack of ``BaseTransformerLayer`` used by encoder / decoder.

    Accepts either a single layer (deep-copied ``num_layers`` times) or an
    explicit list of layers of length ``num_layers``. Subclasses should
    implement ``forward``.

    Args:
        transformer_layers: Template layer or list of layers.
        num_layers: Depth of the stack.
    """

    def __init__(
        self,
        transformer_layers: Union[
            List[BaseTransformerLayer], BaseTransformerLayer
        ],  # noqa
        num_layers: int,
    ):
        super(TransformerLayerSequence, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        if isinstance(transformer_layers, nn.Module):
            self.layers.extend(
                copy.deepcopy(transformer_layers) for _ in range(num_layers)
            )
        else:
            assert (
                isinstance(transformer_layers, list)
                and len(transformer_layers) == num_layers
            )

    def forward(self):
        raise NotImplementedError()
