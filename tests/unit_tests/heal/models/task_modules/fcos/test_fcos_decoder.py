"""Tests for heal.models.task_modules.fcos.decoder."""


def test_fcos_decoder_importable():
    from heal.models.task_modules.fcos.decoder import (
        FCOSDecoder,
        multiclass_nms,
    )

    assert FCOSDecoder is not None
    assert callable(multiclass_nms)
