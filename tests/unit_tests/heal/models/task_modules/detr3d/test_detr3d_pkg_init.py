"""Tests for heal.models.task_modules.detr3d modules."""


def test_detr3d_task_modules_importable():
    from heal.models.task_modules.detr3d.head import (
        FFN,
        Detr3dCrossAtten,
        Detr3dDecoder,
        Detr3dDecoderLayer,
        Detr3dHead,
        Detr3dTransformer,
        FeatureSampler,
    )
    from heal.models.task_modules.detr3d.post_process import Detr3dPostProcess
    from heal.models.task_modules.detr3d.target import (
        Detr3dTarget,
        HungarianAssigner3D,
    )

    assert Detr3dHead is not None
    assert Detr3dTransformer is not None
    assert Detr3dDecoder is not None
    assert Detr3dDecoderLayer is not None
    assert Detr3dCrossAtten is not None
    assert FeatureSampler is not None
    assert FFN is not None
    assert Detr3dTarget is not None
    assert HungarianAssigner3D is not None
    assert Detr3dPostProcess is not None
