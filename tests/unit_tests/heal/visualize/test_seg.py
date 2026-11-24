"""Tests for heal.visualize.seg."""


def test_segviz_importable():
    from heal.visualize.seg import SegViz

    assert SegViz is not None
