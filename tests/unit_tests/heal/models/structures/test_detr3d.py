"""Tests for heal.models.structures.detr3d."""


def test_detr3d_importable():
    from heal.models.structures.detr3d import Detr3d, Detr3dIrInfer

    assert Detr3d is not None
    assert Detr3dIrInfer is not None
