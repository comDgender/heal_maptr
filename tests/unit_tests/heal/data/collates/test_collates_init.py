"""Tests for heal.data.collates __init__."""


def test_collates_package_exports():
    from heal.data.collates import collate_2d, default_collate_v2

    assert callable(collate_2d)
    assert callable(default_collate_v2)
