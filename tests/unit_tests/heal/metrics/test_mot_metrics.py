"""Tests for heal.metrics.mot_metrics."""


def test_mot_metric_importable():
    from heal.metrics.mot_metrics import MotMetric

    assert MotMetric is not None
