"""Tests for heal.models.task_modules.sparsebev.decoder."""

import pytest


def test_sparsebev_decoder_importable():
    from heal.models.task_modules.sparsebev.decoder import SparseBEVDecoder

    assert SparseBEVDecoder is not None


def test_sparsebev_decoder_basic():
    pytest.importorskip("torch")
    import torch

    from heal.models.task_modules.sparsebev.decoder import SparseBEVDecoder

    decoder = SparseBEVDecoder(num_output=5)
    cls_scores = torch.randn(2, 10, 4)
    box_preds = torch.randn(2, 10, 11)

    outputs = decoder(cls_scores, box_preds)

    assert len(outputs) == 2
    assert set(outputs[0].keys()) == {"bboxes", "scores", "labels"}
    assert outputs[0]["bboxes"].shape[0] <= 5
    assert outputs[0]["scores"].shape[0] <= 5
    assert outputs[0]["labels"].shape[0] <= 5


def test_sparsebev_decoder_list_inputs_and_quality():
    pytest.importorskip("torch")
    import torch

    from heal.models.task_modules.sparsebev.decoder import SparseBEVDecoder

    decoder = SparseBEVDecoder(num_output=3, score_threshold=0.1)
    cls_scores = [
        torch.randn(1, 8, 3),
        torch.randn(1, 8, 3),
    ]
    box_preds = [
        torch.randn(1, 8, 11),
        torch.randn(1, 8, 11),
    ]
    quality = [
        torch.randn(1, 8, 2),
        torch.randn(1, 8, 2),
    ]

    outputs = decoder(cls_scores, box_preds, quality)

    assert len(outputs) == 1
    assert outputs[0]["bboxes"].shape[-1] == 10
    assert outputs[0]["scores"].numel() <= 3
