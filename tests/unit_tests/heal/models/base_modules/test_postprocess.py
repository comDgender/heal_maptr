"""Tests for heal.models.base_modules.postprocess."""

import pytest


def test_post_processor_base_requires_subclass_forward():
    pytest.importorskip("torch")
    import torch

    from heal.models.base_modules.postprocess import PostProcessorBase

    class DummyPP(PostProcessorBase):
        def forward(self, pred, meta_data=None):
            return pred

    m = DummyPP()
    t = torch.zeros(1)
    assert torch.equal(m(t), t)
