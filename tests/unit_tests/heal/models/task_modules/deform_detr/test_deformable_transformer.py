"""Tests for heal.models.task_modules.deform_detr.deformable_transformer."""


def test_deformable_transformer_importable():
    from heal.models.task_modules.deform_detr.deformable_transformer import (
        DeformableDetrTransformer,
        DeformableDetrTransformerDecoder,
        DeformableDetrTransformerEncoder,
    )

    assert DeformableDetrTransformer is not None
    assert DeformableDetrTransformerEncoder is not None
    assert DeformableDetrTransformerDecoder is not None
