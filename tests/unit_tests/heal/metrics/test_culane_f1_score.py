"""Tests for heal.metrics.culane_f1_score."""

import pytest


def test_culane_f1_score_importable():
    from heal.metrics.culane_f1_score import CulaneF1Score

    assert CulaneF1Score is not None


def test_culane_f1_score_perfect_match():
    pytest.importorskip("cv2")
    import numpy as np

    from heal.metrics.culane_f1_score import CulaneF1Score

    metric = CulaneF1Score(img_shape=(32, 32, 1), width=3, samples=10)
    lane = np.array([[4, 4], [4, 20]], dtype=np.float32)
    metric.update(annos=[[lane]], preds=[[lane]])
    f1 = metric.compute()
    assert f1 == 1.0


def test_culane_f1_score_no_match():
    pytest.importorskip("cv2")
    import numpy as np

    from heal.metrics.culane_f1_score import CulaneF1Score

    metric = CulaneF1Score(img_shape=(32, 32, 1), width=1, samples=10)
    pred_lane = np.array([[2, 4], [2, 20]], dtype=np.float32)
    anno_lane = np.array([[20, 4], [20, 20]], dtype=np.float32)
    metric.update(annos=[[anno_lane]], preds=[[pred_lane]])
    f1 = metric.compute()
    assert f1 == 0.0
