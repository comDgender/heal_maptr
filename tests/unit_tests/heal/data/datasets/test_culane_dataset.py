"""Tests for heal.data.datasets.culane_dataset."""


def test_culane_dataset_importable():
    from heal.data.datasets.culane_dataset import (
        CuLaneDataset,
        CuLaneFromImage,
        CuLanePacker,
    )

    assert CuLaneDataset is not None
    assert CuLaneFromImage is not None
    assert CuLanePacker is not None
