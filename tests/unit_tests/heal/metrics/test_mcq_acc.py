# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
"""Unit tests for ``heal.metrics.generative_model.mcq_acc.MCQAccuracy``."""

import json
import math
import os

from hat_engine.metrics.metric import EvalMetric
from hat_engine.registry import OBJECT_REGISTRY

from heal.metrics.generative_model.mcq_acc import MCQAccuracy


def test_metric_shape():
    metric = MCQAccuracy()
    assert issubclass(MCQAccuracy, EvalMetric)
    for attr in ("_gather", "_get", "update", "reset", "get"):
        assert hasattr(metric, attr), attr


def test_empty_records():
    metric = MCQAccuracy()
    name, value = metric.get()
    assert name == "MCQAccuracy"
    assert value == 0.0


def test_no_subject():
    metric = MCQAccuracy()
    metric.update({"pred": "A", "target": "A"})
    metric.update({"pred": "B", "target": "C"})
    metric.update({"pred": "C", "target": "C"})
    name, value = metric.get()
    assert name == "MCQAccuracy"
    assert math.isclose(value, 2 / 3, rel_tol=1e-4)


def test_with_subject():
    metric = MCQAccuracy()
    fixtures = [
        ("s1", "A", "A"),
        ("s1", "B", "A"),
        ("s2", "C", "C"),
        ("s2", "D", "D"),
        ("s3", "A", "B"),
        ("s3", "C", "C"),
    ]
    for subj, pred, target in fixtures:
        metric.update({"pred": pred, "target": target, "subject": subj})
    name, overall = metric.get()
    assert name == "MCQAccuracy"
    assert math.isclose(overall, 4 / 6, rel_tol=1e-4)
    assert set(metric._names[:-1]) == {"s1", "s2", "s3"}
    assert metric._names[-1] == "MCQAccuracy"
    assert metric._values[-1] == overall


def test_noisy_predictions():
    metric = MCQAccuracy()
    metric.update({"pred": "The answer is A.", "target": "A"})
    metric.update({"pred": "A) foo", "target": "A"})
    metric.update({"pred": "C", "target": "C"})
    _, value = metric.get()
    assert math.isclose(value, 1.0, rel_tol=1e-4)


def test_reset():
    metric = MCQAccuracy()
    metric.update({"pred": "A", "target": "B"})
    metric.update({"pred": "A", "target": "C"})
    metric.reset()
    metric.update({"pred": "D", "target": "D"})
    _, value = metric.get()
    assert math.isclose(value, 1.0, rel_tol=1e-4)


def test_save_prefix_mixed_updates(tmp_path):
    metric = MCQAccuracy(
        save_prefix=str(tmp_path), save_name="mmlu_infer.jsonl"
    )
    metric.update({"pred": "A", "target": "A"})
    metric.update({"pred": "B", "target": "B", "subject": "foo", "index": 1})
    metric.get()
    path = tmp_path / "mmlu_infer.jsonl"
    assert path.is_file()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    row0 = json.loads(lines[0])
    row1 = json.loads(lines[1])
    assert row0["subject"] is None
    assert row0["index"] is None
    assert row1["subject"] == "foo"
    assert row1["index"] == 1


def test_save_prefix_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    metric = MCQAccuracy()
    metric.update({"pred": "A", "target": "A"})
    metric.get()
    assert os.listdir(tmp_path) == []


def test_registry():
    from heal.data.datasets.generative_model.mmlu import MMLU

    assert OBJECT_REGISTRY.get("MMLU") is MMLU
    assert OBJECT_REGISTRY.get("MCQAccuracy") is MCQAccuracy


def test_ddp_dedup_by_index():
    """Same ``index`` pushed multiple times counts as one sample.

    ``DistributedSampler(drop_last=False)`` pads the tail with
    duplicated samples so every rank gets an equal share; ``_gather``
    would otherwise concatenate the duplicates and skew accuracy.
    """
    metric = MCQAccuracy()
    metric.update({"pred": "A", "target": "A", "subject": "s", "index": 0})
    metric.update({"pred": "B", "target": "C", "subject": "s", "index": 1})
    metric.update({"pred": "C", "target": "C", "subject": "s", "index": 2})
    metric.update({"pred": "D", "target": "C", "subject": "s", "index": 2})
    metric.update({"pred": "D", "target": "C", "subject": "s", "index": 2})
    _, value = metric.get()
    assert math.isclose(value, 2 / 3, rel_tol=1e-4)
