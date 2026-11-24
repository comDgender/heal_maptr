"""Tests for heal.models.task_modules.bev __init__."""


def test_bev_task_modules_exports():
    from heal.models.task_modules.bev import (
        BevDetDecoderInfer,
        BevDetDecoderV2,
        BevEncoder,
        LSSTransformer,
    )

    assert BevDetDecoderInfer is not None
    assert BevDetDecoderV2 is not None
    assert BevEncoder is not None
    assert LSSTransformer is not None
