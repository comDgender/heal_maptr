"""Tests for heal.data.datasets.argoverse2_dataset."""


def test_argoverse2_dataset_importable():
    from heal.data.datasets.argoverse2_dataset import Argoverse2PackedDataset

    assert Argoverse2PackedDataset is not None


def test_argoverse2_dataset_registered():
    from hat_engine.registry import OBJECT_REGISTRY

    import heal  # noqa: F401

    assert "Argoverse2PackedDataset" in OBJECT_REGISTRY
