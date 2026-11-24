"""Tests for heal.models.task_modules.lidar.anchor_generator."""

import pytest


def test_anchor3d_generator_stride_forward():
    pytest.importorskip("torch")

    from heal.models.task_modules.lidar.anchor_generator import (
        Anchor3DGeneratorStride,
    )

    generator = Anchor3DGeneratorStride(
        class_names=["Car"],
        anchor_sizes=[[1.6, 3.9, 1.56]],
        anchor_strides=[[0.16, 0.16, 4.0]],
        anchor_offsets=[[0.08, 0.08, 0.0]],
        rotations=[[0, 1.57]],
        match_thresholds=[0.6],
        unmatch_thresholds=[0.45],
    )
    anchor_dict = generator([1, 4, 4], device="cpu")
    assert "anchors" in anchor_dict
    assert len(anchor_dict["anchors"]) == 1
    assert anchor_dict["anchors"][0].shape[-1] == 7
