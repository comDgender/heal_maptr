"""Tests for heal.models.task_modules.bevformer.encoder."""

import pytest


def test_single_frame_encoder_layer_forward():
    pytest.importorskip("torch")
    pytest.importorskip("horizon_plugin_pytorch")
    import torch
    import torch.nn as nn

    from heal.models.task_modules.bevformer.encoder import (
        SingleFrameBEVFormerEncoderLayer,
    )

    class MockSelfAttention(nn.Module):
        def forward(self, query, key=None, value=None, **kwargs):
            return query

    class MockCrossAttention(nn.Module):
        def forward(self, query, key=None, value=None, **kwargs):
            return query + 0.1

    layer = SingleFrameBEVFormerEncoderLayer(
        MockSelfAttention(),
        MockCrossAttention(),
        embed_dims=16,
        dropout=0.0,
    ).eval()

    num_query, bs, dim = 8, 2, 16
    query = torch.randn(num_query, bs, dim)
    key = torch.randn(num_query, bs, dim)
    value = key
    query_pos = torch.randn(num_query, bs, dim)
    ref_2d = torch.rand(bs, num_query, 1, 2)
    sa_spatial_shapes = torch.tensor([[4, 4]], dtype=torch.float32)
    cross_spatial_shapes = torch.tensor([[8, 8]], dtype=torch.float32)

    out = layer(
        query=query,
        key=key,
        value=value,
        query_pos=query_pos,
        ref_2d=ref_2d,
        sa_spatial_shapes=sa_spatial_shapes,
        cross_spatial_shapes=cross_spatial_shapes,
        queries_rebatch_grid=torch.zeros(bs * 2, 4, 4, 2),
        restore_bev_grid=torch.zeros(bs, 8, 4, 2),
        reference_points_rebatch=torch.zeros(bs * 2, 16, 2, 1),
        bev_pillar_counts=torch.ones(bs, 4, 4, 1),
    )
    assert out.shape == query.shape


def test_bevformer_encoder_flatten_mlvl_feats():
    pytest.importorskip("torch")
    pytest.importorskip("horizon_plugin_pytorch")
    import torch
    import torch.nn as nn

    from heal.models.task_modules.bevformer.encoder import BEVFormerEncoder

    class DummyLayer(nn.Module):
        def forward(self, **kwargs):
            return kwargs["query"]

    encoder = BEVFormerEncoder(
        bev_h=4,
        bev_w=4,
        embed_dims=16,
        encoder_layer=DummyLayer(),
        num_feature_levels=1,
        num_cams=2,
        num_layers=1,
    ).eval()

    mlvl_feats = [torch.randn(2, 2, 16, 8, 8)]
    feat_flatten, cross_shapes, sa_shapes = encoder._flatten_mlvl_feats(
        mlvl_feats
    )
    assert feat_flatten.ndim == 4
    assert cross_shapes.shape[0] == 1
    assert sa_shapes.tolist() == [[4.0, 4.0]]
