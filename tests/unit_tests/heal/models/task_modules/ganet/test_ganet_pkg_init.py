"""Tests for heal.models.task_modules.ganet __init__."""


def test_ganet_task_modules_exports():
    from heal.models.task_modules.ganet import (
        GaNetDecoder,
        GaNetHead,
        GaNetLoss,
        GaNetNeck,
        GaNetTarget,
    )

    assert GaNetHead is not None
    assert GaNetNeck is not None
    assert GaNetDecoder is not None
    assert GaNetTarget is not None
    assert GaNetLoss is not None
