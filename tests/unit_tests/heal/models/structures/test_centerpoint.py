"""Tests for heal.models.structures.centerpoint."""


def test_centerpoint_detector_importable():
    from heal.models.structures.centerpoint import (
        CenterPointDetector,
        CenterPointDetectorIrInfer,
    )

    assert CenterPointDetector is not None
    assert CenterPointDetectorIrInfer is not None


def _make_centerpoint_modules():
    import pytest

    pytest.importorskip("torch")
    import torch
    import torch.nn as nn

    class PreProcess(nn.Module):
        def forward(self, points, is_deploy):
            batch_size = len(points)
            features = torch.randn(batch_size, 8, 4)
            coords = torch.zeros(batch_size, 4, dtype=torch.int32)
            return features, coords

    class Reader(nn.Module):
        def forward(self, features, horizon_preprocess=True):
            return features

    class ScatterBackbone(nn.Module):
        def forward(self, features, coors, batch_size, input_shape):
            return torch.randn(batch_size, 16, 8, 8)

    class Neck(nn.Module):
        def forward(self, x):
            return x

    class Head(nn.Module):
        def forward(self, x):
            return [
                {
                    "reg": torch.randn(x.shape[0], 2, 8, 8),
                    "height": torch.randn(x.shape[0], 1, 8, 8),
                    "dim": torch.randn(x.shape[0], 3, 8, 8),
                    "rot": torch.randn(x.shape[0], 2, 8, 8),
                    "vel": torch.randn(x.shape[0], 2, 8, 8),
                    "heatmap": torch.randn(x.shape[0], 1, 8, 8),
                }
            ]

    class PostProcess(nn.Module):
        def forward(self, outs):
            return [{"boxes": torch.zeros(1, 9), "scores": torch.ones(1)}]

    return PreProcess, Reader, ScatterBackbone, Neck, Head, PostProcess


def test_centerpoint_detector_deploy_forward():
    import pytest

    pytest.importorskip("torch")
    import torch

    from heal.models.structures.centerpoint import CenterPointDetector

    (
        PreProcess,
        Reader,
        ScatterBackbone,
        Neck,
        Head,
        PostProcess,
    ) = _make_centerpoint_modules()
    model = CenterPointDetector(
        feature_map_shape=[512, 512, 1],
        reader=Reader(),
        scatter=None,
        backbone=ScatterBackbone(),
        neck=Neck(),
        head=Head(),
        is_deploy=True,
    ).eval()

    outputs = model(
        {
            "features": torch.randn(1, 8, 4),
            "coors": torch.zeros(1, 4, dtype=torch.int32),
        }
    )
    assert isinstance(outputs, list)
    assert len(outputs) == 6


def test_centerpoint_detector_ir_infer_forward():
    import pytest

    pytest.importorskip("torch")
    import torch
    import torch.nn as nn

    from heal.models.structures.centerpoint import CenterPointDetectorIrInfer

    PreProcess, _, _, _, _, PostProcess = _make_centerpoint_modules()

    class IrModel(nn.Module):
        def forward(self, data):
            batch_size = data["batch_size"]
            return [
                torch.randn(batch_size, 2, 8, 8),
                torch.randn(batch_size, 1, 8, 8),
                torch.randn(batch_size, 3, 8, 8),
                torch.randn(batch_size, 2, 8, 8),
                torch.randn(batch_size, 2, 8, 8),
                torch.randn(batch_size, 1, 8, 8),
            ]

    tasks = [{"num_class": 1, "class_names": ["car"]}]
    infer = CenterPointDetectorIrInfer(
        ir_model=IrModel(),
        pre_process=PreProcess(),
        feature_map_shape=[512, 512, 1],
        postprocess=PostProcess(),
        tasks=tasks,
        transpose_output=False,
    ).eval()

    results = infer({"points": [torch.randn(32, 5)]})
    assert isinstance(results, list)
    assert "boxes" in results[0]


def test_centerpoint_detector_ir_infer_transpose_output():
    import pytest

    pytest.importorskip("torch")
    import torch
    import torch.nn as nn

    from heal.models.structures.centerpoint import CenterPointDetectorIrInfer

    PreProcess, _, _, _, _, PostProcess = _make_centerpoint_modules()

    class IrModel(nn.Module):
        def forward(self, data):
            batch_size = data["batch_size"]
            return [
                torch.randn(batch_size, 8, 8, 2),
                torch.randn(batch_size, 8, 8, 1),
                torch.randn(batch_size, 8, 8, 3),
                torch.randn(batch_size, 8, 8, 2),
                torch.randn(batch_size, 8, 8, 2),
                torch.randn(batch_size, 8, 8, 1),
            ]

    tasks = [{"num_class": 1, "class_names": ["car"]}]
    infer = CenterPointDetectorIrInfer(
        ir_model=IrModel(),
        pre_process=PreProcess(),
        feature_map_shape=[512, 512, 1],
        postprocess=PostProcess(),
        tasks=tasks,
        transpose_output=True,
    ).eval()

    results = infer({"points": [torch.randn(32, 5)]})
    assert isinstance(results, list)
    assert "boxes" in results[0]
