"""Tests for heal.data.metas."""

from heal.data.metas import img_metas


def test_img_metas_contains_common_keys():
    assert "img" in img_metas
    assert "gt_bboxes" in img_metas
    assert isinstance(img_metas, list)
