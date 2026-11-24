"""Tests for heal.models.task_modules.deform_detr __init__."""


def test_deform_detr_task_modules_exports():
    from heal.models.task_modules.deform_detr import (
        ChannelMapperNeck,
        DeformableCriterion,
        DeformableDetrTransformer,
        DeformableDetrTransformerDecoder,
        DeformableDetrTransformerEncoder,
        DeformDetrPostProcess,
        SetCriterion,
    )

    assert DeformableCriterion is not None
    assert SetCriterion is not None
    assert DeformableDetrTransformer is not None
    assert DeformableDetrTransformerEncoder is not None
    assert DeformableDetrTransformerDecoder is not None
    assert ChannelMapperNeck is not None
    assert DeformDetrPostProcess is not None
