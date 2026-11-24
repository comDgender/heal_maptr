# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from __future__ import annotations
import gc
import logging
import os
from datetime import datetime
from importlib.metadata import version
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from hat_engine.metrics.metric import EvalMetric
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.distributed import (
    all_gather_object,
    get_dist_info,
    rank_zero_only,
)
from hat_engine.utils.package_helper import require_packages

from heal.data.datasets.navsim.navsim_dataset import load_cfg
from heal.metrics.navsim.navsim_metric_impl import (
    NavsimMetricV1Impl,
    NavsimMetricV2Impl,
)

logger = logging.getLogger(__name__)
_navsim_pkg_version = "0"
try:
    _navsim_pkg_version = version("navsim")
except Exception as exc:
    logger.warning(
        "Could not read installed navsim version via importlib.metadata (%s); "
        "assuming major version 0 (v1-style behaviors).",
        exc,
    )
_navsim_major_version = _navsim_pkg_version.split(".")[0]
NAVSIM_VERSION = (
    "v2"
    if _navsim_major_version.isdigit() and int(_navsim_major_version) >= 2
    else "v1"
)


@OBJECT_REGISTRY.register
class NavsimMetric(EvalMetric):
    """Navsim PDM metric with NuscenesMetric-style per-token DDP gather."""

    @require_packages("navsim", "nuplan")
    def __init__(
        self,
        name: str = "NavsimMetric",
        metric_cache_path: str = "",
        future_seconds: float = 4.0,
        interval_seconds: float = 0.1,
        split_cfg_root: Optional[str] = None,
        split: str = "navtest",
        save_path: str = "./navsim_eval_results",
        eval_stage: str = "one_stage",
        verbose: bool = True,
        visualize: bool = False,
    ):
        """Initialize NavsimMetric.

        Args:
            name: Metric display name used in logs and output files.
            metric_cache_path: Path to navsim metric cache directory.
            future_seconds: Future horizon length for trajectory evaluation.
            interval_seconds: Time interval between trajectory waypoints.
            split_cfg_root: Root path of navsim split configuration files.
            split: Evaluation split, either "navtest" or "navhard_two_stage".
            save_path: Directory used to save intermediate evaluation results.
            eval_stage: Evaluation pipeline stage, "one_stage" or "two_stage".
            verbose: Whether to emit verbose progress logs.
            visualize: Whether to render visualization during evaluation.
        """
        super(NavsimMetric, self).__init__(name)
        self.metric_cache_path = metric_cache_path
        self.future_seconds = future_seconds
        self.interval_seconds = interval_seconds
        self.navsim_pkg_version = _navsim_pkg_version
        if split not in ("navtest", "navhard_two_stage"):
            raise ValueError(
                f"split must be 'navtest' or 'navhard_two_stage', got {split!r}"  # noqa: E501
            )
        self.split = split
        self.save_path = save_path
        self.eval_stage = eval_stage
        self.verbose = verbose
        self.visualize = visualize

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        if eval_stage == "two_stage" and NAVSIM_VERSION == "v2":
            self.reactive_all_mapping = load_cfg(
                split_cfg_root,
                split,
            ).reactive_all_mapping
        else:
            self.reactive_all_mapping = None
        self._initialized = False
        # token -> score_row (same shape as NuscenesMetric.nusc_annos)
        self.results: Dict[str, Dict[str, Any]] = {}
        self.ret = ["pdm_score", 0.0]
        self._navhard_v1_skip_warned = False
        self.metric_implement: Optional[NavsimMetricV1Impl] = None

    def compute(self):
        # Same stub as NuscenesMetric: marks EvalMetric DDP-ready so
        # ``get()`` does not warn about DistributedSampler.
        pass

    def _navhard_v1_skip(self) -> bool:
        """Return whether the current split must be skipped on navsim v1.

        Returns:
            True when split is "navhard_two_stage" and navsim major version is v1.
        """
        return self.split == "navhard_two_stage" and NAVSIM_VERSION == "v1"

    def _warn_navhard_v1_skip(self, where: str) -> None:
        """Log once that the current navhard split is skipped on navsim v1.

        Args:
            where: Caller phase name, used in the warning message.
        """
        if self._navhard_v1_skip_warned:
            return
        logger.warning(
            "NavsimMetric: split=%r with navsim v1 is not supported; skipping %s.",  # noqa: E501
            self.split,
            where,
        )
        self._navhard_v1_skip_warned = True

    def _init_resources(self):
        """Lazily create the version-matched metric implementation per worker."""
        if self._initialized:
            return
        if NAVSIM_VERSION == "v2":
            self.metric_implement = NavsimMetricV2Impl.from_metric(self)
        elif NAVSIM_VERSION == "v1":
            self.metric_implement = NavsimMetricV1Impl.from_metric(self)
        else:
            raise RuntimeError(
                f"Unsupported NAVSIM_VERSION={NAVSIM_VERSION!r}; expected 'v1' or 'v2'."  # noqa: E501
            )
        self._initialized = True

    def reset(self):
        """Reset the internal metric state between evaluation runs."""
        if NAVSIM_VERSION == "v2" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.results = {}
        self.ret = ["pdm_score", 0.0]
        self._navhard_v1_skip_warned = False

    def update(self, data: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Accumulate per-sample PDM rows into ``self.results`` by token.

        Args:
            data: Batched navsim input data for the current iteration.
            outputs: Model outputs containing predicted trajectories.
        """
        if self._navhard_v1_skip():
            self._warn_navhard_v1_skip("update")
            return

        self._init_resources()
        batch_rows: List[Dict[str, Any]] = []
        if self.metric_implement is not None:
            self.metric_implement.update(data, outputs, batch_rows)
        else:
            logger.error(
                f"Unsupported NAVSIM_VERSION={NAVSIM_VERSION!r}; "
                "expected 'v1' or 'v2'."
            )
            batch_rows.append({"token": "unknown", "valid": False})

        for row in batch_rows:
            token = row.get("token", "unknown")
            self.results[token] = row

    def _gather(self) -> List[Optional[Dict[str, Dict[str, Any]]]]:
        """Gather per-rank token→row dicts (NuscenesMetric-style)."""
        _rank, global_world_size = get_dist_info()
        total_results: List[Optional[Dict[str, Dict[str, Any]]]] = [
            None for _ in range(global_world_size)
        ]
        all_gather_object(total_results, self.results)
        return total_results

    def to_csv(self) -> None:
        """Write merged ``self.results`` to ``Navsim_eval/``."""
        df = pd.DataFrame(list(self.results.values()))
        time_mark = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            "Navsim_eval", f"{self.name}_{time_mark}.csv"
        )
        os.makedirs("Navsim_eval", exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved detailed metric results to {output_path}")

    def get(self) -> Tuple[str, float]:
        """Aggregate distributed results and return the PDM score.

        Returns:
            Tuple of metric name and aggregated PDM score.
        """
        if self._navhard_v1_skip():
            self._warn_navhard_v1_skip("get")
            self.ret = ["pdm_score", 0.0]
            self.compute()
            return self.ret[0], self.ret[1]

        logger.info(
            "NavsimMetric: local results=%d",
            len(self.results),
        )
        if len(self.results) != 0:
            gathered = self._gather()
            merged: Dict[str, Dict[str, Any]] = dict(gathered[0] or {})
            for part in gathered[1:]:
                if part:
                    merged.update(part)
            self.results = merged

            all_scores = list(self.results.values())
            self._get(all_scores)

            if NAVSIM_VERSION == "v2":
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                self.to_csv()

            # ``_get`` is rank0-only; broadcast so every rank sees the same
            # score. Otherwise DeepSpeed ``save_checkpoint(ds-best-ckpt)``
            # deadlocks when only rank0 decides ``is_best=True``.
            _rank, world_size = get_dist_info()
            if (
                world_size > 1
                and torch.distributed.is_available()
                and torch.distributed.is_initialized()
            ):
                payload = [self.ret]
                torch.distributed.broadcast_object_list(payload, src=0)
                self.ret = payload[0]

        self.compute()
        return self.ret[0], self.ret[1]

    @rank_zero_only
    def _get(self, all_scores: List[Any]) -> None:
        """Populate ``self.ret`` with the aggregated PDM score on rank 0.

        Args:
            all_scores: Flattened per-sample results gathered from all ranks.
        """
        if not all_scores:
            self.ret[1] = 0.0
            return

        if self.metric_implement is None:
            raise RuntimeError("Metric implement is not initialized.")

        if NAVSIM_VERSION == "v1":
            metrics_summary = self.metric_implement.summarize(all_scores)
            logger.info(f"Evaluated {len(all_scores)} samples.")
            logger.info(f"Averaged metric: {metrics_summary}")
            if "pdm_score" in metrics_summary:
                self.ret[1] = metrics_summary["pdm_score"]
            return

        if NAVSIM_VERSION == "v2":
            self.ret[0], self.ret[1] = self.metric_implement.summarize(
                all_scores
            )
            return

        logger.error(
            f"Unsupported NAVSIM_VERSION={NAVSIM_VERSION!r}; expected 'v1' or 'v2'."  # noqa: E501
        )
        self.ret = ["pdm_score", 0.0]
