"""Tests for heal.metrics.argoverse2_metric."""

import pytest


def test_argoverse2_metrics_importable():
    from heal.metrics.argoverse2_metric import (
        BrierMetric,
        HitRate,
        MinADE,
        MinFDE,
        MissRate,
    )

    assert BrierMetric is not None
    assert MinADE is not None
    assert MinFDE is not None
    assert MissRate is not None
    assert HitRate is not None


def test_argoverse2_metrics_registered():
    from hat_engine.registry import OBJECT_REGISTRY

    import heal  # noqa: F401

    for name in ["BrierMetric", "MinADE", "MinFDE", "MissRate", "HitRate"]:
        assert name in OBJECT_REGISTRY


def test_min_ade_zero_when_prediction_matches_target():
    pytest.importorskip("torch")
    import torch

    from heal.metrics.argoverse2_metric import MinADE

    num_modes = 6
    num_steps = 3
    target = torch.rand(1, num_steps, 2)
    # Every mode predicts the ground-truth trajectory exactly.
    pred = target.unsqueeze(1).repeat(1, num_modes, 1, 1)
    prob = torch.ones(1, num_modes) / num_modes
    valid_mask = torch.ones(1, num_steps, dtype=torch.bool)

    metric = MinADE(max_guesses=num_modes)
    metric.update(pred, target, prob, valid_mask)
    assert torch.allclose(metric.compute(), torch.tensor(0.0), atol=1e-5)
