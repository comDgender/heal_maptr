"""Tests for heal.models.structures.bevformer."""

import pytest


def test_bevformer_importable():
    from heal.models.structures.bevformer import BevFormer, BevFormerIrInfer

    assert BevFormer is not None
    assert BevFormerIrInfer is not None


def _make_bevformer_modules():
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn

    class Backbone(nn.Module):
        def forward(self, x):
            return [
                torch.randn(x.shape[0], 8, x.shape[-2] // 4, x.shape[-1] // 4)
            ]

    class Neck(nn.Module):
        def forward(self, feats):
            return feats

    class ViewTransformer(nn.Module):
        def forward(self, feats, data):
            return torch.randn(feats[0].shape[0], 16, 8, 8)

        def get_reference_points_cam(self, data):
            return (
                torch.zeros(1, 4, 4, 2),
                torch.zeros(1, 8, 8, 2),
                torch.zeros(1, 8, 4),
                torch.ones(1, 8, 8, 1),
            )

    class BevDecoder(nn.Module):
        def __init__(self, tag):
            super().__init__()
            self.tag = tag

        def forward(self, bev_feat, data):
            return {self.tag: bev_feat}

    return Backbone, Neck, ViewTransformer, BevDecoder


def test_bevformer_forward_eval():
    pytest.importorskip("torch")
    import torch

    from heal.models.structures.bevformer import BevFormer

    BB, Neck, VT, Dec = _make_bevformer_modules()
    model = BevFormer(
        BB(),
        Neck(),
        VT(),
        out_indices=(0,),
        bev_decoders=[Dec("det")],
    ).eval()
    results = model({"img": torch.randn(1, 3, 32, 32)})
    assert isinstance(results, list)
    assert results[0]["det"] is not None


def test_bevformer_extract_feat_and_export_reference_points():
    pytest.importorskip("torch")
    import torch

    from heal.models.structures.bevformer import BevFormer

    BB, Neck, VT, Dec = _make_bevformer_modules()
    model = BevFormer(
        BB(), Neck(), VT(), out_indices=(0,), bev_decoders=[Dec("det")]
    ).eval()
    img = torch.randn(1, 3, 32, 32)
    feats = model.extract_feat(img)
    assert isinstance(feats, list)
    ref = model.export_reference_points({"img": img})
    assert len(ref) == 4


def test_bevformer_multiple_decoders():
    pytest.importorskip("torch")
    import torch

    from heal.models.structures.bevformer import BevFormer

    BB, Neck, VT, Dec = _make_bevformer_modules()
    model = BevFormer(
        BB(),
        Neck(),
        VT(),
        out_indices=(0,),
        bev_decoders=[Dec("det"), Dec("aux")],
    ).eval()
    results = model({"img": torch.randn(1, 3, 32, 32)})
    assert len(results) == 2
    assert "det" in results[0] and "aux" in results[1]


def test_bevformer_ir_infer_forward():
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn

    from heal.models.structures.bevformer import BevFormerIrInfer

    class ViewTransformer(nn.Module):
        global_pose_key = "ego2global"

        def get_deploy_prev(self, data, prev_frame_info):
            return torch.zeros(1, 64, 16), torch.zeros(1, 8, 8, 8, 2)

    class DeployModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.view_transformer = ViewTransformer()
            self.bev_decoders = nn.ModuleList([DecoderHead()])

        def export_reference_points(self, data):
            return (
                torch.zeros(1, 4, 4, 2),
                torch.zeros(1, 8, 8, 2),
                torch.zeros(1, 8, 4),
                torch.ones(1, 8, 8, 1),
            )

    class DecoderHead(nn.Module):
        def get_outputs(self, cls_scores, refs, coords, bev_embed):
            return {"boxes": coords[0]}

        def post_process(self, outputs):
            return [outputs]

    class IrModel(nn.Module):
        def forward(self, inputs):
            return (
                torch.randn(1, 64, 16),
                torch.randn(1, 4, 10),
                torch.randn(1, 4, 3),
                torch.randn(1, 4, 10),
            )

    infer = BevFormerIrInfer(
        IrModel(),
        DeployModel(),
        lambda m: m,
    ).eval()
    data = {
        "img": torch.randn(1, 3, 32, 32),
        "seq_meta": [
            {"meta": [{"scene": "s0"}], "ego2global": [torch.eye(4)]}
        ],
    }
    results = infer(data)
    assert isinstance(results, list)
    assert "boxes" in results[0]
