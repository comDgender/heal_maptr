"""Tests for heal.models.task_modules.motr modules."""


def test_motr_task_modules_importable():
    from heal.models.task_modules.motr.criterion import MotrCriterion
    from heal.models.task_modules.motr.head import MotrHead
    from heal.models.task_modules.motr.motr_deformable_transformer import (
        DeformableTransformerDecoder,
        DeformableTransformerDecoderLayer,
        DeformableTransformerEncoder,
        DeformableTransformerEncoderLayer,
        MotrDeformableTransformer,
        MotrDeformAttn,
        SinPos,
    )
    from heal.models.task_modules.motr.post_process import (
        MotrPostProcess,
        RuntimeTrackerBase,
    )
    from heal.models.task_modules.motr.qim import QueryInteractionModule

    assert MotrHead is not None
    assert MotrCriterion is not None
    assert QueryInteractionModule is not None
    assert MotrPostProcess is not None
    assert RuntimeTrackerBase is not None
    assert MotrDeformableTransformer is not None
    assert DeformableTransformerEncoder is not None
    assert DeformableTransformerDecoder is not None
    assert DeformableTransformerEncoderLayer is not None
    assert DeformableTransformerDecoderLayer is not None
    assert SinPos is not None
    assert MotrDeformAttn is not None
