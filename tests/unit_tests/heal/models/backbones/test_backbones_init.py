"""Tests for heal.models.backbones __init__."""


def test_backbones_subpackage_importable():
    import heal.models.backbones as bb

    assert bb is not None
