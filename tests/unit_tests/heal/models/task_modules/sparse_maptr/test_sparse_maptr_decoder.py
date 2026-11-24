"""Tests for heal.models.task_modules.sparse_maptr.decoder."""

import pytest


def test_sparse_maptr_decoder_importable():
    from heal.models.task_modules.sparse_maptr.decoder import (
        SparseMapPerceptionDecoder,
    )

    assert SparseMapPerceptionDecoder is not None


def test_sparse_maptr_decoder_get_outputs():
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn

    from heal.models.task_modules.sparse_maptr.decoder import (
        SparseMapPerceptionDecoder,
    )

    decoder = SparseMapPerceptionDecoder(
        decoder=nn.Identity(),
        embed_dims=16,
        num_vec_one2one=2,
        num_vec_one2many=0,
        num_pts_per_vec=4,
        transform_method="minmax",
    ).eval()

    outputs_classes = [torch.randn(1, 2, 3)]
    reference_out = [torch.randn(1, 2, 8)]

    preds = decoder.get_outputs(outputs_classes, reference_out)

    assert preds["all_cls_scores"].shape == (1, 1, 2, 3)
    assert preds["all_bbox_preds"].shape == (1, 1, 2, 4)
    assert preds["all_pts_preds"].shape == (1, 1, 2, 4, 2)


def test_sparse_maptr_decoder_quant_analysis_skips_post_process(monkeypatch):
    """quant_analysis needs om.all_* tensors, not decoded instances."""
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn

    from heal.models.task_modules.sparse_maptr.decoder import (
        SparseMapPerceptionDecoder,
    )

    class MockHead(nn.Module):
        def forward(self, mlvl_feats, data, **kwargs):
            return {
                "classification": [torch.randn(1, 2, 3)],
                "prediction": [torch.randn(1, 2, 8)],
            }

    class BoomPostProcess(nn.Module):
        def forward(self, outputs, data):
            raise AssertionError("post_process must not run in quant_analysis")

    decoder = SparseMapPerceptionDecoder(
        decoder=MockHead(),
        embed_dims=16,
        num_vec_one2one=2,
        num_pts_per_vec=4,
        post_process=BoomPostProcess(),
    ).eval()
    monkeypatch.setenv("HAT_TRAINING_STEP", "quant_analysis")

    outputs = decoder(None, [torch.randn(6, 16, 8, 8)], data={})

    assert "all_cls_scores" in outputs
    assert "all_bbox_preds" in outputs
    assert "all_pts_preds" in outputs


def test_sparse_maptr_decoder_deploy_forward():
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn

    from heal.models.task_modules.sparse_maptr.decoder import (
        SparseMapPerceptionDecoder,
    )

    class MockHead(nn.Module):
        def forward(self, mlvl_feats, data, **kwargs):
            return {
                "classification": torch.randn(1, 2, 3),
                "prediction": torch.randn(1, 2, 8),
            }

    decoder = SparseMapPerceptionDecoder(
        decoder=MockHead(),
        embed_dims=16,
        num_vec_one2one=2,
        is_deploy=True,
    ).eval()

    feats = [torch.randn(6, 16, 8, 8)]
    outputs = decoder(None, feats, data={})

    assert "classification" in outputs
    assert "prediction" in outputs
