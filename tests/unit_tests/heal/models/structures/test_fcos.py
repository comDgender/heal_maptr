"""Tests for heal.models.structures.fcos."""


def test_fcos_importable():
    from heal.models.structures.fcos import FCOS, FCOSIrInfer

    assert FCOS is not None
    assert FCOSIrInfer is not None
