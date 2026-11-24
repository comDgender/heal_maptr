"""Tests for heal.models.weight_init."""

import pytest


def test_normal_init_runs_on_linear():
    pytest.importorskip("torch")
    import torch.nn as nn

    from heal.models.weight_init import normal_init

    m = nn.Linear(4, 2)
    normal_init(m, mean=0.0, std=0.01, bias=0.1)
    assert m.weight.shape == (2, 4)


def test_bias_init_with_prob_finite():
    from heal.models.weight_init import bias_init_with_prob

    b = bias_init_with_prob(0.01)
    assert isinstance(b, float)
    assert b == b  # not nan
