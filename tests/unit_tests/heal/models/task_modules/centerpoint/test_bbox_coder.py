"""Tests for heal.models.task_modules.centerpoint.bbox_coders."""

import pytest


def test_centerpoint_bbox_coder_topk():
    pytest.importorskip("torch")
    import torch

    from heal.models.task_modules.centerpoint.bbox_coders import (
        CenterPointBBoxCoder,
    )

    coder = CenterPointBBoxCoder(
        pc_range=[-51.2, -51.2],
        out_size_factor=8,
        voxel_size=[0.1, 0.1],
    )
    scores = torch.randn(2, 3, 16, 16)
    topk_score, topk_inds, topk_clses, topk_ys, topk_xs = coder._topk(
        scores, K=5
    )
    assert topk_score.shape == (2, 5)
    assert topk_inds.shape == (2, 5)
    assert topk_clses.shape == (2, 5)
    assert topk_ys.shape == (2, 5)
    assert topk_xs.shape == (2, 5)


def test_centerpoint_bbox_coder_gather_feat():
    pytest.importorskip("torch")
    import torch

    from heal.models.task_modules.centerpoint.bbox_coders import (
        CenterPointBBoxCoder,
    )

    coder = CenterPointBBoxCoder(
        pc_range=[-51.2, -51.2],
        out_size_factor=8,
        voxel_size=[0.1, 0.1],
    )
    feats = torch.randn(2, 100, 8)
    inds = torch.randint(0, 100, (2, 10))
    gathered = coder._gather_feat(feats, inds)
    assert gathered.shape == (2, 10, 8)
