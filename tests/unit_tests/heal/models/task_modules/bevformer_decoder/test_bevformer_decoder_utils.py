"""Tests for heal.models.task_modules.bevformer_decoder.utils."""

import pytest


def test_normalize_denormalize_bbox_roundtrip():
    pytest.importorskip("torch")
    import torch

    from heal.models.task_modules.bevformer_decoder.utils import (
        denormalize_bbox,
        normalize_bbox,
    )

    boxes = torch.tensor(
        [
            [1.0, 2.0, 0.5, 4.0, 5.0, 2.0, 0.3, 1.0, -0.2],
            [-2.0, 3.0, 1.0, 2.0, 3.0, 1.5, -0.5, 0.1, 0.4],
        ],
        dtype=torch.float32,
    )
    normalized = normalize_bbox(boxes)
    recovered = denormalize_bbox(normalized)
    assert recovered.shape == boxes.shape
    assert torch.allclose(recovered, boxes, atol=1e-5)
