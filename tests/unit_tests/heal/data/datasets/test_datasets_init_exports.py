"""Tests for heal.data.datasets __init__ exports."""


def test_datasets_init_exports_cityscapes():
    from heal.data.datasets import CITYSCAPES_LABLE_MAPPINGS, Cityscapes

    assert Cityscapes is not None
    assert isinstance(CITYSCAPES_LABLE_MAPPINGS, tuple)
