"""Tests for heal.models.task_modules.petr modules."""


def test_petr_task_modules_importable():
    from heal.models.task_modules.petr.head import (
        PETRDecoder,
        PETRDecoderLayer,
        PETRHead,
        PETRTransformer,
    )

    assert PETRHead is not None
    assert PETRTransformer is not None
    assert PETRDecoder is not None
    assert PETRDecoderLayer is not None
