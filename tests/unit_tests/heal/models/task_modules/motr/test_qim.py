"""Tests for heal.models.task_modules.motr.qim."""


def test_qim_importable():
    from heal.models.task_modules.motr.qim import QueryInteractionModule

    assert QueryInteractionModule is not None
