"""Tests for heal.models.task_modules.flashocc __init__."""


def test_flashocc_task_modules_exports():
    from heal.models.task_modules.flashocc import (
        BEVOCCHead2D,
        FlashOccDecoderInfer,
        FlashOccDetDecoder,
    )

    assert FlashOccDetDecoder is not None
    assert FlashOccDecoderInfer is not None
    assert BEVOCCHead2D is not None
