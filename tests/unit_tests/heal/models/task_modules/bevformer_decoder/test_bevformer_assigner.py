"""Tests for heal.models.task_modules.bevformer_decoder.assigner."""

import pytest


def _make_assigner():
    pytest.importorskip("torch")
    pytest.importorskip("scipy")
    import torch
    import torch.nn as nn

    from heal.models.task_modules.bevformer_decoder.assigner import (
        BevFormerHungarianAssigner3D,
    )

    class ClsCost(nn.Module):
        def forward(self, cls_pred, gt_labels):
            num_bboxes = cls_pred.shape[0]
            num_gts = gt_labels.shape[0]
            return torch.zeros(num_bboxes, num_gts)

    class RegCost(nn.Module):
        def forward(self, bbox_pred, gt_bboxes):
            num_bboxes = bbox_pred.shape[0]
            num_gts = gt_bboxes.shape[0]
            return torch.zeros(num_bboxes, num_gts)

    return BevFormerHungarianAssigner3D(ClsCost(), RegCost())


def test_assigner_no_ground_truth():
    pytest.importorskip("torch")
    import torch

    assigner = _make_assigner()
    bbox_pred = torch.randn(4, 10)
    cls_pred = torch.randn(4, 3)
    assigned = assigner.assign(
        bbox_pred, cls_pred, torch.empty(0, 9), torch.empty(0)
    )
    assert assigned.tolist() == [0, 0, 0, 0]


def test_assigner_matches_predictions_to_gt():
    pytest.importorskip("torch")
    import torch

    assigner = _make_assigner()
    bbox_pred = torch.randn(4, 10)
    cls_pred = torch.randn(4, 3)
    gt_bboxes = torch.randn(2, 9)
    gt_labels = torch.tensor([0, 1], dtype=torch.long)
    assigned = assigner.assign(bbox_pred, cls_pred, gt_bboxes, gt_labels)
    assert assigned.shape == (4,)
    assert (assigned >= 0).all()
    assert (assigned <= 2).all()
    assert (assigned > 0).sum() == 2
