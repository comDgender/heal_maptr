"""Tests for heal.models.structures.pointpillars."""


def test_pointpillars_detector_importable():
    from heal.models.structures.pointpillars import (
        PointPillarsDetector,
        PointPillarsDetectorIrInfer,
    )

    assert PointPillarsDetector is not None
    assert PointPillarsDetectorIrInfer is not None
