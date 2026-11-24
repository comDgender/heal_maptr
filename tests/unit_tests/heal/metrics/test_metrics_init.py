"""Tests for heal.metrics __init__."""


def test_metrics_exports_coco_detection_metric():
    from heal.metrics import COCODetectionMetric

    assert COCODetectionMetric is not None
