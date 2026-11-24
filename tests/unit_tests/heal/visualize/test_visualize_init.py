"""Tests for heal.visualize __init__."""


def test_visualize_public_exports():
    from heal.visualize import ClsViz, DetViz, SegViz

    assert ClsViz is not None
    assert DetViz is not None
    assert SegViz is not None
