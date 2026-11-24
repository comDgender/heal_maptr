"""Tests for heal.models.task_modules.seg __init__."""


def test_seg_task_modules_exports():
    from heal.models.task_modules.seg import SegHead

    assert SegHead is not None
