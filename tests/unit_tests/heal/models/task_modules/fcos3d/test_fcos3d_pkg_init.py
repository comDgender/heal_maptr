"""Tests for heal.models.task_modules.fcos3d modules."""


def test_fcos3d_task_modules_importable():
    from heal.models.task_modules.fcos3d.bbox_coder import FCOS3DBBoxCoder
    from heal.models.task_modules.fcos3d.head import FCOS3DHead
    from heal.models.task_modules.fcos3d.loss import FCOS3DLoss
    from heal.models.task_modules.fcos3d.post_process import FCOS3DPostProcess
    from heal.models.task_modules.fcos3d.target import FCOS3DTarget

    assert FCOS3DHead is not None
    assert FCOS3DLoss is not None
    assert FCOS3DPostProcess is not None
    assert FCOS3DBBoxCoder is not None
    assert FCOS3DTarget is not None
