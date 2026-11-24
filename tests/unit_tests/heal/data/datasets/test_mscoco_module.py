"""Tests for heal.data.datasets.mscoco."""


def test_coco_label_remap_table_has_expected_len():
    from heal.data.datasets.mscoco import COCO_CLASS_NAMES, COCO_LABLE_TO_LABLE

    assert isinstance(COCO_LABLE_TO_LABLE, dict)
    assert len(COCO_LABLE_TO_LABLE) >= 80
    assert len(COCO_CLASS_NAMES) == 80
