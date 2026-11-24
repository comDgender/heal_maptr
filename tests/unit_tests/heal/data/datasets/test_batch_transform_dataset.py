"""Tests for heal.data.datasets.batch_transform_dataset."""

import pytest


def test_batch_transform_dataset_len_delegates():
    pytest.importorskip("torch")
    from torch.utils.data import TensorDataset

    from heal.data.datasets.batch_transform_dataset import (
        BatchTransformDataset,
    )

    base = TensorDataset(__import__("torch").zeros(5, 1))
    ds = BatchTransformDataset(base, transforms_cfgs=[[]], epoch_steps=[0])
    assert len(ds) == 5
