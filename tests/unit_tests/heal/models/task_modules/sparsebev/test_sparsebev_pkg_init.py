"""Tests for heal.models.task_modules.sparsebev __init__."""


def test_sparsebev_task_modules_exports():
    from heal.models.task_modules.sparsebev import (
        AsymmetricFFN,
        DeformableFeatureAggregation,
        DeformableFeatureAggregationv2,
        DenseDepthNet,
        InstanceBank,
        MemoryBank,
        SparseBEVDecoder,
        SparseBEVEncoder,
        SparseBEVHead,
        SparseBEVKeyPointsGenerator,
        SparseBEVRefinementModule,
        SparseBEVTarget,
    )

    assert SparseBEVHead is not None
    assert SparseBEVDecoder is not None
    assert DeformableFeatureAggregation is not None
    assert DeformableFeatureAggregationv2 is not None
    assert InstanceBank is not None
    assert MemoryBank is not None
    assert SparseBEVEncoder is not None
    assert SparseBEVKeyPointsGenerator is not None
    assert SparseBEVRefinementModule is not None
    assert AsymmetricFFN is not None
    assert DenseDepthNet is not None
    assert SparseBEVTarget is not None
