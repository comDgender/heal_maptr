"""Tests for heal.models.task_modules.sparse_maptr.blocks."""

import pytest


def test_sparse_point3d_encoder_forward():
    pytest.importorskip("torch")
    import torch

    from heal.models.task_modules.sparse_maptr.blocks import (
        SparsePoint3DEncoder,
    )

    encoder = SparsePoint3DEncoder(embed_dims=16, input_dim=8)
    points = torch.randn(2, 4, 8)
    out = encoder(points)

    assert out.shape == (2, 4, 16)


def test_sparse_point3d_keypoints_generator_forward():
    pytest.importorskip("torch")
    import torch

    from heal.models.task_modules.sparse_maptr.blocks import (
        SparsePoint3DKeyPointsGenerator,
    )

    generator = SparsePoint3DKeyPointsGenerator(
        embed_dims=16,
        num_sample=2,
        num_learnable_pts=2,
        fix_height=(0, 0.5),
        ground_height=-1.0,
    )
    anchor = torch.randn(1, 3, 4)
    instance_feature = torch.randn(1, 3, 16)

    key_points = generator(anchor, instance_feature)

    assert key_points.ndim == 4
    assert key_points.shape[0] == 1
    assert key_points.shape[1] == 3


def test_sparse_point3d_refinement_module_forward():
    pytest.importorskip("torch")
    import torch

    from heal.models.task_modules.sparse_maptr.blocks import (
        SparsePoint3DRefinementModule,
    )

    refine = SparsePoint3DRefinementModule(
        embed_dims=16,
        num_sample=2,
        coords_dim=2,
        num_cls=3,
    )
    instance_feature = torch.randn(1, 4, 16)
    anchor = torch.randn(1, 4, 4)
    anchor_embed = torch.randn(1, 4, 16)

    new_anchor, cls = refine(instance_feature, anchor, anchor_embed)

    assert new_anchor.shape == anchor.shape
    assert cls.shape == (1, 4, 3)
