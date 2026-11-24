"""Tests for heal.models.task_modules.lidar modules."""


def test_lidar_task_modules_importable():
    from heal.models.task_modules.lidar.anchor_generator import (
        Anchor3DGeneratorStride,
    )
    from heal.models.task_modules.lidar.box_coders import GroundBox3dCoder
    from heal.models.task_modules.lidar.pillar_encoder import (
        PillarFeatureNet,
        PointPillarScatter,
    )
    from heal.models.task_modules.lidar.target_assigner import (
        LidarTargetAssigner,
    )

    assert Anchor3DGeneratorStride is not None
    assert GroundBox3dCoder is not None
    assert PillarFeatureNet is not None
    assert PointPillarScatter is not None
    assert LidarTargetAssigner is not None
