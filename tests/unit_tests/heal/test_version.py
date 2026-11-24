"""Tests for heal.version."""

import heal.version as version


def test_version_string_is_defined():
    assert hasattr(version, "__version__")
    assert isinstance(version.__version__, str)
    assert len(version.__version__) > 0
