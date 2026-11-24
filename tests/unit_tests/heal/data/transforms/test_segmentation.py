"""Tests for heal.data.transforms.segmentation."""

import pytest


def test_label_remap_torch():
    pytest.importorskip("torch")
    import torch

    from heal.data.transforms.segmentation import LabelRemap

    mapping = [0, 1, 2, 3]
    t = LabelRemap(mapping)
    data = {"gt_seg": torch.tensor([[0, 1], [2, 3]], dtype=torch.long)}
    out = t(data)
    assert torch.equal(out["gt_seg"], data["gt_seg"])
