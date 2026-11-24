"""Tests for heal.models.task_modules.maptr __init__."""


def test_maptr_task_modules_exports():
    from heal.models.task_modules.maptr import (
        MapInstanceDecoder,
        MapInstanceDetectorHead,
        MapTRAssigner,
        MapTRCriterion,
        MapTRDecoder,
        MapTRPerceptionDecoder,
        MapTRPostProcess,
        OrderedPtsL1Cost,
        PtsDirCosLoss,
        PtsL1Cost,
        PtsL1Loss,
    )

    assert MapTRPerceptionDecoder is not None
    assert MapTRDecoder is not None
    assert MapTRAssigner is not None
    assert MapTRCriterion is not None
    assert MapTRPostProcess is not None
    assert MapInstanceDecoder is not None
    assert MapInstanceDetectorHead is not None
    assert PtsDirCosLoss is not None
    assert PtsL1Loss is not None
    assert PtsL1Cost is not None
    assert OrderedPtsL1Cost is not None
