"""Tests for heal.core.box_utils."""

import pytest


def test_fp16_clamp_cpu_float32():
    pytest.importorskip("torch")
    import torch

    from heal.core.box_utils import fp16_clamp

    x = torch.tensor([-1.0, 0.5, 2.0])
    y = fp16_clamp(x, min=0.0, max=1.0)
    assert y.dtype == torch.float32
    assert y.tolist() == [0.0, 0.5, 1.0]


def test_bbox_overlaps_iou_torch():
    pytest.importorskip("torch")
    import torch

    from heal.core.box_utils import bbox_overlaps

    b1 = torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0]])
    b2 = torch.tensor([[1.0, 1.0, 3.0, 3.0]])
    iou = bbox_overlaps(b1, b2, mode="iou", is_aligned=False)
    assert iou.shape == (2, 1)
    assert (iou >= 0).all() and (iou <= 1).all()
