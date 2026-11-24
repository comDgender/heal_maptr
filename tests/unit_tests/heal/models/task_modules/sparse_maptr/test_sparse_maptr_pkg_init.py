"""Tests for heal.models.task_modules.sparse_maptr __init__."""


def test_sparse_maptr_task_modules_exports():
    from heal.models.task_modules.sparse_maptr import (
        SparseMapHead,
        SparseMapPerceptionDecoder,
        SparsePoint3DEncoder,
        SparsePoint3DKeyPointsGenerator,
        SparsePoint3DRefinementModule,
    )

    assert SparseMapHead is not None
    assert SparseMapPerceptionDecoder is not None
    assert SparsePoint3DEncoder is not None
    assert SparsePoint3DKeyPointsGenerator is not None
    assert SparsePoint3DRefinementModule is not None
