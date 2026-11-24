"""Tests for heal.data.datasets.imagenet package."""


def test_imagenet_subpackage_importable():
    from heal.data.datasets.imagenet import ImageNetPacker

    assert ImageNetPacker is not None
