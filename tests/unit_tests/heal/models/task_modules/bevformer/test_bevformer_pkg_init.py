"""Tests for heal.models.task_modules.bevformer __init__."""


def test_bevformer_task_modules_exports():
    from heal.models.task_modules.bevformer import (
        BevFormerCriterion,
        BEVFormerDetDecoder,
        BEVFormerEncoder,
        BEVFormerEncoderLayer,
        BevFormerHungarianAssigner3D,
        BevFormerProcess,
        BevFormerViewTransformer,
        DetectionTransformerDecoder,
        DetrTransformerDecoderLayer,
        HorizonMultiScaleDeformableAttention,
        HorizonMultiScaleDeformableAttention3D,
        HorizonSpatialCrossAttention,
        HorizonTemporalSelfAttention,
        SingleFrameBEVFormerEncoder,
        SingleFrameBEVFormerEncoderLayer,
        SingleFrameBevFormerViewTransformer,
    )

    exports = (
        BEVFormerEncoder,
        BEVFormerEncoderLayer,
        BevFormerViewTransformer,
        BevFormerCriterion,
        BevFormerProcess,
        BEVFormerDetDecoder,
        BevFormerHungarianAssigner3D,
        HorizonTemporalSelfAttention,
        HorizonSpatialCrossAttention,
        HorizonMultiScaleDeformableAttention,
        HorizonMultiScaleDeformableAttention3D,
        DetectionTransformerDecoder,
        DetrTransformerDecoderLayer,
        SingleFrameBEVFormerEncoder,
        SingleFrameBEVFormerEncoderLayer,
        SingleFrameBevFormerViewTransformer,
    )
    assert all(item is not None for item in exports)
