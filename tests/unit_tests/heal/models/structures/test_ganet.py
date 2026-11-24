"""Tests for heal.models.structures.ganet."""


def test_ganet_importable():
    from heal.models.structures.ganet import GaNet, GaNetIrInfer

    assert GaNet is not None
    assert GaNetIrInfer is not None
