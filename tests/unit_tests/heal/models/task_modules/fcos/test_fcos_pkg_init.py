"""Tests for heal.models.task_modules.fcos __init__."""


def test_fcos_task_modules_exports():
    from heal.models.task_modules.fcos import (
        DynamicFcosTarget,
        FCOSDecoder,
        FCOSHead,
        distance2bbox,
        get_points,
        multiclass_nms,
    )

    assert FCOSHead is not None
    assert FCOSDecoder is not None
    assert DynamicFcosTarget is not None
    assert callable(get_points)
    assert callable(distance2bbox)
    assert callable(multiclass_nms)
