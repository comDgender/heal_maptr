"""Tests for heal.data.transforms __init__ exports."""


def test_transforms_public_api_importable():
    from heal.data.transforms import (
        LabelRemap,
        Scale,
        SegOneHot,
        SegRandomAffine,
        SegResize,
    )

    assert LabelRemap is not None
    assert Scale is not None
    assert SegOneHot is not None
    assert SegRandomAffine is not None
    assert SegResize is not None
