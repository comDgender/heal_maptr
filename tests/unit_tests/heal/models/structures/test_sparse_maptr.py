"""Tests for heal.models.structures.sparse_maptr."""

import pytest


def test_sparse_maptr_importable():
    from heal.models.structures.sparse_maptr import (
        SparseMapIrInfer,
        SparseMapTR,
    )

    assert SparseMapTR is not None
    assert SparseMapIrInfer is not None


def _make_sparse_maptr_modules():
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn

    class Backbone(nn.Module):
        def forward(self, x):
            return [
                torch.randn(x.shape[0], 8, x.shape[-2] // 4, x.shape[-1] // 4),
                torch.randn(
                    x.shape[0], 16, x.shape[-2] // 8, x.shape[-1] // 8
                ),
            ]

    class Neck(nn.Module):
        def forward(self, feats):
            return feats

    class Decoder(nn.Module):
        def forward(self, bev_embed, mlvl_feats, data):
            batch = mlvl_feats[0].shape[0] // 6
            return {
                "all_cls_scores": torch.randn(batch, 1, 2, 3),
                "all_pts_preds": torch.randn(batch, 1, 2, 4, 2),
            }

    return Backbone, Neck, Decoder


def test_sparse_maptr_forward_eval():
    pytest.importorskip("torch")
    import torch

    from heal.models.structures.sparse_maptr import SparseMapTR

    BB, Neck, Dec = _make_sparse_maptr_modules()
    model = SparseMapTR(
        BB(),
        out_indices=(0, 1),
        neck=Neck(),
        bev_decoders=[Dec()],
    ).eval()

    outputs = model({"img": torch.randn(6, 3, 32, 48)})

    assert isinstance(outputs, list)
    assert "all_cls_scores" in outputs[0]


def test_sparse_maptr_extract_feat_and_multi_decoder():
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn

    from heal.models.structures.sparse_maptr import SparseMapTR

    BB, Neck, Dec = _make_sparse_maptr_modules()

    class TagDecoder(nn.Module):
        def __init__(self, tag):
            super().__init__()
            self.tag = tag

        def forward(self, bev_embed, mlvl_feats, data):
            batch = mlvl_feats[0].shape[0] // 6
            return {self.tag: torch.randn(batch, 1)}

    model = SparseMapTR(
        BB(),
        out_indices=(0, 1),
        neck=Neck(),
        bev_decoders=[TagDecoder("a"), TagDecoder("b")],
    ).eval()
    feats = model.extract_feat(torch.randn(6, 3, 32, 48))
    outputs = model({"img": torch.randn(6, 3, 32, 48)})

    assert len(feats) == 2
    assert len(outputs) == 2
    assert "a" in outputs[0] and "b" in outputs[1]


def test_sparse_map_head_gen_projection_mat():
    pytest.importorskip("torch")
    import torch

    from heal.models.task_modules.sparse_maptr.head import SparseMapHead

    metas = {
        "img": [torch.randn(3, 32, 48) for _ in range(6)],
        "lidar2img": torch.eye(4).unsqueeze(0).repeat(6, 1, 1),
    }
    projection_mat = SparseMapHead.gen_projection_mat("lidar2img", metas)

    assert projection_mat.shape == (6, 4, 4)


def test_sparse_maptr_ir_infer_forward():
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn

    from heal.models.structures.sparse_maptr import SparseMapIrInfer

    class IrModel(nn.Module):
        def forward(self, inputs):
            batch = inputs["projection_mat"].shape[0]
            return {
                "classification": torch.randn(batch, 2, 3),
                "prediction": torch.randn(batch, 2, 8),
            }

    class BevDecoder(nn.Module):
        def get_outputs(self, outputs_classes, reference_out):
            return {"preds": outputs_classes[0]}

        def post_process(self, outputs, data):
            return outputs

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.bev_decoders = nn.ModuleList([BevDecoder()])

    infer = SparseMapIrInfer(
        ir_model=IrModel(),
        test_model=TestModel(),
        model_convert_pipeline=lambda model: model,
        projection_mat_key="lidar2img",
    ).eval()

    metas = {
        "img": torch.randn(6, 3, 32, 48),
        "lidar2img": torch.eye(4).unsqueeze(0).repeat(6, 1, 1),
    }
    outputs = infer(metas)

    assert "preds" in outputs
