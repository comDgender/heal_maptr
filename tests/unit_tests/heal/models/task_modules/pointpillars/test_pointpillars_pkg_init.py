"""Tests for heal.models.task_modules.pointpillars modules."""


def test_pointpillars_task_modules_importable():
    from heal.models.task_modules.pointpillars.head import PointPillarsHead
    from heal.models.task_modules.pointpillars.loss import PointPillarsLoss
    from heal.models.task_modules.pointpillars.postprocess import (
        PointPillarsPostProcess,
    )
    from heal.models.task_modules.pointpillars.preprocess import (
        PointPillarsPreProcess,
    )

    assert PointPillarsHead is not None
    assert PointPillarsLoss is not None
    assert PointPillarsPostProcess is not None
    assert PointPillarsPreProcess is not None
