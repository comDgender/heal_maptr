"""Tests for heal.data.transforms.lidar_utils split modules."""


def test_lidar_utils_submodules_importable():
    from heal.data.transforms.lidar_utils import (
        augmentations,
        filters,
        samplers,
    )

    assert hasattr(samplers, "BatchSampler")
    assert hasattr(filters, "DBFilterByDifficulty")
    assert hasattr(augmentations, "random_flip")


def test_batch_sampler_sample():
    import pytest

    pytest.importorskip("numpy")
    from heal.data.transforms.lidar_utils.samplers import BatchSampler

    sampler = BatchSampler([10, 20, 30], name="Car", shuffle=False)
    batch = sampler.sample(2)
    assert batch == [10, 20]


def test_filter_gt_box_outside_range_keeps_in_range_boxes():
    import numpy as np

    from heal.data.transforms.lidar_utils.filters import (
        filter_gt_box_outside_range,
    )

    bv_range = [0, -39.68, 69.12, 39.68]
    gt_boxes = np.array(
        [[12.6, -5.5, -0.8, 1.7, 4.1, 1.5, 2.0]],
        dtype=np.float32,
    )
    mask = filter_gt_box_outside_range(gt_boxes, bv_range)
    assert mask.tolist() == [True]
