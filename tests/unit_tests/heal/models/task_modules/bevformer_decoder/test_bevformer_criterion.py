"""Tests for heal.models.task_modules.bevformer_decoder.criterion."""

import pytest


def test_bevformer_criterion_empty_gt():
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn

    from heal.models.task_modules.bevformer_decoder.criterion import (
        BevFormerCriterion,
    )

    class MockAssigner(nn.Module):
        def assign(self, bbox_pred, cls_pred, gt_bboxes, gt_labels):
            return bbox_pred.new_zeros(bbox_pred.shape[0], dtype=torch.long)

    class MockLossCls(nn.Module):
        def forward(self, cls_scores, labels, label_weights, avg_factor):
            return {"cls": cls_scores.sum() * 0.0}

    class MockLossBBox(nn.Module):
        def forward(self, pred, target, weight, avg_factor):
            return pred.sum() * 0.0

    criterion = BevFormerCriterion(
        assigner=MockAssigner(),
        num_classes=3,
        sync_cls_avg_factor=False,
        loss_cls=MockLossCls(),
        loss_bbox=MockLossBBox(),
        num_group_detr=1,
    ).train()

    preds = {
        "all_cls_scores": torch.randn(2, 1, 4, 3),
        "all_bbox_preds": torch.randn(2, 1, 4, 10),
    }
    data = {"seq_meta": [{"ego_bboxes_labels": [[]]}]}
    loss_dict = criterion(preds, data)
    assert "loss_cls" in loss_dict
    assert "loss_bbox" in loss_dict
    assert "d0.loss_cls" in loss_dict
