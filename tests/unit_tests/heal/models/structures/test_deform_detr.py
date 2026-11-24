"""Tests for heal.models.structures.deform_detr."""


def test_deform_detr_importable():
    from heal.models.structures.deform_detr import (
        DeformableDETR,
        DeformDetrIrInfer,
    )

    assert DeformableDETR is not None
    assert DeformDetrIrInfer is not None
