"""Tests for heal.visualize.lane_lines."""


def test_laneline_viz_importable():
    from heal.visualize.lane_lines import LanelineViz

    assert LanelineViz is not None
