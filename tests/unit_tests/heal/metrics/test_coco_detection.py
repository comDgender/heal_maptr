"""Tests for heal.metrics.coco_detection."""


def test_coco_detection_metric_class_registered():
    from heal.metrics.coco_detection import COCODetectionMetric

    assert COCODetectionMetric.__name__ == "COCODetectionMetric"
