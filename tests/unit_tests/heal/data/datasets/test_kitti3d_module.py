"""Tests for heal.data.datasets.kitti3d."""


def test_kitti3d_classes_importable():
    from heal.data.datasets.kitti3d import (
        Kitti3D,
        Kitti3DDetection,
        Kitti3DReader,
    )

    assert Kitti3D is not None
    assert Kitti3DDetection is not None
    assert Kitti3DReader is not None
