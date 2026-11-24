"""Tests for heal.models __init__."""


def test_models_package_importable():
    import heal.models as models_pkg

    assert models_pkg is not None
