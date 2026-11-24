"""Tests for heal.data.datasets.mot17_dataset."""


def test_mot17_dataset_importable():
    from heal.data.datasets.mot17_dataset import (
        Mot17Dataset,
        Mot17FromImage,
        Mot17Packer,
    )

    assert Mot17Dataset is not None
    assert Mot17FromImage is not None
    assert Mot17Packer is not None
