"""Tests for heal.models.structures.sparsebev."""

import pytest


def test_sparsebev_importable():
    from heal.models.structures.sparsebev import SparseBEV, SparseBEVIrInfer

    assert SparseBEV is not None
    assert SparseBEVIrInfer is not None


def _make_sparsebev_modules():
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn

    class Backbone(nn.Module):
        def forward(self, x):
            return [
                torch.randn(x.shape[0], 16, x.shape[-2] // 4, x.shape[-1] // 4)
            ]

    class Neck(nn.Module):
        def forward(self, feats):
            return feats

    class Head(nn.Module):
        def forward(self, feature_maps, data, compiler_model=False):
            batch = feature_maps[0].shape[0]
            return {
                "classification": torch.randn(batch, 4, 10),
                "prediction": torch.randn(batch, 4, 11),
            }

    class DepthBranch(nn.Module):
        def forward(self, feature_maps, data):
            return [torch.randn(1)]

        def loss(self, depths, data):
            return depths[0].sum() * 0.0

    return Backbone, Neck, Head, DepthBranch


def test_sparsebev_forward_eval():
    pytest.importorskip("torch")
    import torch

    from heal.models.structures.sparsebev import SparseBEV

    BB, Neck, Head, _ = _make_sparsebev_modules()
    model = SparseBEV(BB(), Head(), Neck()).eval()
    outputs = model({"img": torch.randn(1, 3, 32, 32)})

    assert "classification" in outputs
    assert "prediction" in outputs


def test_sparsebev_forward_with_depth_branch_train():
    pytest.importorskip("torch")
    import torch

    from heal.models.structures.sparsebev import SparseBEV

    BB, Neck, Head, Depth = _make_sparsebev_modules()
    model = SparseBEV(BB(), Head(), Neck(), depth_branch=Depth()).train()
    outputs = model({"img": torch.randn(1, 3, 32, 32)})

    assert "loss_dense_depth" in outputs


def test_sparsebev_extract_feat():
    pytest.importorskip("torch")
    import torch

    from heal.models.structures.sparsebev import SparseBEV

    BB, Neck, Head, _ = _make_sparsebev_modules()
    model = SparseBEV(BB(), Head(), Neck()).eval()
    feats = model.extract_feat(torch.randn(1, 3, 32, 32))

    assert isinstance(feats, list)
    assert feats[0].shape[1] == 16


def test_sparsebev_set_qconfig():
    pytest.importorskip("torch")
    import torch.nn as nn

    from heal.models.structures.sparsebev import SparseBEV

    BB, Neck, Head, _ = _make_sparsebev_modules()

    class QConfigChild(nn.Module):
        def __init__(self):
            super().__init__()
            self.qconfig_set = False

        def set_qconfig(self):
            self.qconfig_set = True

    bb = QConfigChild()
    model = SparseBEV(bb, Head(), Neck())
    model.set_qconfig()
    assert bb.qconfig_set


def test_sparsebev_ir_infer_first_frame():
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn

    from heal.models.structures.sparsebev import SparseBEVIrInfer
    from heal.models.task_modules.sparsebev.decoder import SparseBEVDecoder

    num_anchor = 8
    embed_dims = 16

    class IrModel(nn.Module):
        def forward(self, inputs):
            batch = inputs["cached_anchor"].shape[0]
            return {
                "classification": torch.randn(batch, num_anchor, 3),
                "prediction": torch.randn(batch, num_anchor, 11),
                "quality": torch.randn(batch, num_anchor, 2),
                "feature": torch.randn(batch, num_anchor, embed_dims),
            }

    infer = SparseBEVIrInfer(
        ir_model=IrModel(),
        projection_mat_key="lidar2img",
        global_mat_key="lidar2global",
        first_frame_input={
            "cached_anchor": torch.zeros(1, num_anchor, 11),
            "cached_feature": torch.zeros(1, num_anchor, embed_dims),
            "cached_confidence": torch.zeros(1, num_anchor),
            "mask": torch.zeros(1, dtype=torch.bool),
        },
        decoder=SparseBEVDecoder(num_output=4),
        use_memory_bank=False,
        num_temp_instances=4,
    ).eval()

    metas = {
        "img": torch.randn(6, 3, 32, 48),
        "timestamp": torch.tensor([0.0]),
        "lidar2global": torch.eye(4).unsqueeze(0),
        "lidar2img": torch.eye(4).unsqueeze(0).repeat(6, 1, 1),
    }
    results = infer(metas)

    assert isinstance(results, list)
    assert "bboxes" in results[0]


def test_instance_bank_topk_and_cache():
    pytest.importorskip("torch")
    import torch

    from heal.models.task_modules.sparsebev.instance_bank import (
        InstanceBank,
        topk,
    )

    confidence = torch.tensor(
        [[[0.1, 0.2, 0.3], [0.9, 0.8, 0.7], [0.5, 0.4, 0.6], [0.2, 0.1, 0.3]]]
    )
    feature = torch.arange(12, dtype=torch.float32).view(1, 4, 3)
    anchor = torch.arange(44, dtype=torch.float32).view(1, 4, 11)

    top_conf, gathered = topk(confidence.max(dim=-1)[0], 2, feature, anchor)

    assert top_conf.shape == (1, 2)
    assert gathered[0].shape == (1, 2, 3)
    assert gathered[1].shape == (1, 2, 11)

    cached_conf, cached_feat, cached_anchor = InstanceBank.cache_instance(
        num_temp_instances=2,
        instance_feature=feature,
        anchor=anchor,
        confidence=confidence,
        cached_confidence=torch.zeros(1, 4),
        confidence_decay=0.6,
    )
    assert cached_conf.shape == (1, 2)
    assert cached_feat.shape == (1, 2, 3)
    assert cached_anchor.shape == (1, 2, 11)
