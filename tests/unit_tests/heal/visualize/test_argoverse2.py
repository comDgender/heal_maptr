"""Tests for heal.visualize.argoverse2."""


def test_argoverse2_viz_importable():
    from heal.visualize.argoverse2 import Argoverse2Viz

    assert Argoverse2Viz is not None


def test_argoverse2_viz_registered():
    from hat_engine.registry import OBJECT_REGISTRY

    import heal  # noqa: F401

    assert "Argoverse2Viz" in OBJECT_REGISTRY
