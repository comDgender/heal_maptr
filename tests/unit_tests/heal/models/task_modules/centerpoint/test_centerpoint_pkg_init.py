"""Tests for heal.models.task_modules.centerpoint __init__."""


def test_centerpoint_task_modules_exports():
    from heal.models.task_modules.centerpoint import (
        CenterPointBBoxCoder,
        CenterPointHead,
        CenterPointLidarTarget,
        CenterPointLoss,
        CenterPointPostProcess,
        CenterPointPreProcess,
        CenterPointTargetForLidarBBOX,
    )

    assert CenterPointPreProcess is not None
    assert CenterPointLidarTarget is not None

    assert CenterPointBBoxCoder is not None
    assert CenterPointHead is not None
    assert CenterPointLoss is not None
    assert CenterPointPostProcess is not None
    assert CenterPointTargetForLidarBBOX is not None
