"""Tests for heal.models.structures __init__."""


def test_structures_exports():
    from heal.models.structures import (
        Bev,
        BevIrInfer,
        MapTR,
        MapTRIrInfer,
        Segmentor,
        SegmentorIrInfer,
        SparseBEV,
        SparseBEVIrInfer,
        SparseMapIrInfer,
        SparseMapTR,
        SparseMultitask,
        SparseMultitaskIrInfer,
    )

    assert Segmentor is not None
    assert SegmentorIrInfer is not None
    assert Bev is not None
    assert BevIrInfer is not None
    assert MapTR is not None
    assert MapTRIrInfer is not None
    assert SparseBEV is not None
    assert SparseBEVIrInfer is not None
    assert SparseMapTR is not None
    assert SparseMapIrInfer is not None
    assert SparseMultitask is not None
    assert SparseMultitaskIrInfer is not None
