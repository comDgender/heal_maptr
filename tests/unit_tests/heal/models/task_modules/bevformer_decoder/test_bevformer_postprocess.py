"""Tests for heal.models.task_modules.bevformer_decoder.postprocess."""

import pytest


def test_bevformer_process_forward():
    pytest.importorskip("torch")
    import torch

    from heal.models.task_modules.bevformer_decoder.postprocess import (
        BevFormerProcess,
    )

    process = BevFormerProcess(
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        post_center_range=[-40, -40, -5, 40, 40, 3],
        max_num=10,
        score_threshold=0.01,
        num_classes=3,
    ).eval()

    cls_scores = torch.randn(1, 8, 3)
    bbox_preds = torch.randn(1, 8, 10)
    preds = {
        "all_cls_scores": cls_scores.unsqueeze(0),
        "all_bbox_preds": bbox_preds.unsqueeze(0),
    }
    outputs = process(preds)
    assert len(outputs) == 1
    assert outputs[0].ndim == 2
    assert outputs[0].shape[-1] == 11
