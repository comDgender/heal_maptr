"""Tests for heal.visualize.lidar_det."""


def test_lidar_det_visualize_importable():
    from heal.visualize.lidar_det import lidar_det_visualize

    assert callable(lidar_det_visualize)
