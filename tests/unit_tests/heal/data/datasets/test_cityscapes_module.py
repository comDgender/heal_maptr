"""Tests for heal.data.datasets.cityscapes."""


def test_cityscapes_label_mapping_defined():
    from heal.data.datasets.cityscapes import CITYSCAPES_LABLE_MAPPINGS

    assert isinstance(CITYSCAPES_LABLE_MAPPINGS, tuple)
    assert len(CITYSCAPES_LABLE_MAPPINGS) > 0
