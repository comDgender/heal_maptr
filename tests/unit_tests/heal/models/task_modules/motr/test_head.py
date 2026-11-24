"""Tests for heal.models.task_modules.motr.head."""


def test_motr_head_importable():
    from heal.models.task_modules.motr.head import MotrHead

    assert MotrHead is not None
