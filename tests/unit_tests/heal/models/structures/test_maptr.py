"""Tests for heal.models.structures.maptr."""

import pytest


def test_maptr_importable():
    from heal.models.structures.maptr import MapTR, MapTRIrInfer

    assert MapTR is not None
    assert MapTRIrInfer is not None


def _make_maptr_modules():
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

    class ViewTransformer(nn.Module):
        def forward(self, feats, data):
            batch = feats[0].shape[0] // 6
            return torch.randn(batch, 16, 8, 8)

        def get_reference_points_cam(self, data):
            return (
                torch.zeros(6, 4, 4, 2),
                torch.zeros(1, 8, 8, 2),
                torch.zeros(6, 16, 4, 2),
                torch.ones(1, 8, 8, 1),
            )

    class OsmEncoder(nn.Module):
        def forward(self, osm):
            return torch.randn(osm.shape[0], 16, 8, 8)

    class BevFusion(nn.Module):
        def forward(self, bev_feat, osm_feat):
            return bev_feat + osm_feat.mean(dim=(2, 3), keepdim=True)

    class BevDecoder(nn.Module):
        def forward(self, bev_feat, feats, data):
            batch = bev_feat.shape[0]
            return {
                "all_cls_scores": torch.randn(batch, 1, 2, 3),
                "all_pts_preds": torch.randn(batch, 1, 2, 4, 2),
            }

    return Backbone, Neck, ViewTransformer, OsmEncoder, BevFusion, BevDecoder


def test_maptr_forward_eval_with_sd_map_fusion():
    pytest.importorskip("torch")
    import torch

    from heal.models.structures.maptr import MapTR

    BB, Neck, VT, OsmEnc, Fusion, Dec = _make_maptr_modules()
    model = MapTR(
        BB(),
        out_indices=(0, 1),
        neck=Neck(),
        view_transformer=VT(),
        sd_map_fusion=True,
        osm_encoder=OsmEnc(),
        bev_fusion=Fusion(),
        bev_decoders=[Dec()],
    ).eval()

    outputs = model(
        {
            "img": torch.randn(6, 3, 32, 48),
            "osm_mask": torch.randn(1, 1, 8, 8),
            "seq_meta": [
                {"ego2img": [torch.eye(4).unsqueeze(0).repeat(6, 1, 1)]}
            ],
        }
    )

    assert isinstance(outputs, list)
    assert "all_cls_scores" in outputs[0]


def test_maptr_extract_feat_and_export_reference_points():
    pytest.importorskip("torch")
    import torch

    from heal.models.structures.maptr import MapTR

    BB, Neck, VT, OsmEnc, Fusion, Dec = _make_maptr_modules()
    model = MapTR(
        BB(),
        out_indices=(0, 1),
        neck=Neck(),
        view_transformer=VT(),
        sd_map_fusion=True,
        osm_encoder=OsmEnc(),
        bev_fusion=Fusion(),
        bev_decoders=[Dec()],
    ).eval()

    img = torch.randn(6, 3, 32, 48)
    data = {
        "img": img,
        "osm_mask": torch.randn(1, 1, 8, 8),
        "seq_meta": [{"ego2img": [torch.eye(4).unsqueeze(0).repeat(6, 1, 1)]}],
    }
    feats = model.extract_feat(img)
    ref_points = model.export_reference_points(data)

    assert len(feats) == 2
    assert len(ref_points) == 4


def test_maptr_ir_infer_forward():
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn

    from heal.models.structures.maptr import MapTRIrInfer

    class IrModel(nn.Module):
        def forward(self, inputs):
            batch = inputs["osm_mask"].shape[0]
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

        def export_reference_points(self, data):
            return (
                torch.zeros(6, 4, 4, 2),
                torch.zeros(1, 8, 8, 2),
                torch.zeros(6, 16, 4, 2),
                torch.ones(1, 8, 8, 1),
            )

    infer = MapTRIrInfer(
        ir_model=IrModel(),
        test_model=TestModel(),
        model_convert_pipeline=lambda model: model,
        sd_map_fusion=True,
    ).eval()

    outputs = infer(
        {
            "img": torch.randn(6, 3, 32, 48),
            "osm_mask": torch.randn(1, 1, 8, 8),
            "seq_meta": [
                {"ego2img": [torch.eye(4).unsqueeze(0).repeat(6, 1, 1)]}
            ],
        }
    )

    assert isinstance(outputs, list)
    assert "preds" in outputs[0]
