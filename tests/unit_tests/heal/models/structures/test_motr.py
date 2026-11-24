"""Tests for heal.models.structures.motr."""


def test_motr_importable():
    from heal.models.structures.motr import Motr, MotrIrInfer

    assert Motr is not None
    assert MotrIrInfer is not None
