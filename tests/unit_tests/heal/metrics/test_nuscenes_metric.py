"""Tests for heal.metrics.nuscenes_metric."""

import pytest


def test_nuscenes_metric_fast_get_returns_cached_score():
    pytest.importorskip("nuscenes")

    from heal.metrics.nuscenes_metric import NuscenesMetric

    metric = NuscenesMetric(
        data_root="/tmp",
        version="v1.0-mini",
        verbose=False,
    )
    metric.ret = ["NDS", 0.42]
    metric.nusc_annos = {"sample_token": []}

    name, value = metric.fast_get()
    assert name == "NDS"
    assert value == 0.42


def test_nuscenes_mono_metric_fast_get_returns_cached_score():
    pytest.importorskip("nuscenes")

    from heal.metrics.nuscenes_metric import NuscenesMonoMetric

    metric = NuscenesMonoMetric(
        data_root="/tmp",
        version="v1.0-mini",
        verbose=False,
    )
    metric.ret = ["NDS", 0.31]

    name, value = metric.fast_get()
    assert name == "NDS"
    assert value == 0.31
