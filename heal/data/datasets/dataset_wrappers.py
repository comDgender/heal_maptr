# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import hashlib
import json
import logging
import os
import pickle
import tempfile
import time

import numpy as np
from hat_engine.registry import OBJECT_REGISTRY

logger = logging.getLogger(__name__)

# Bump when the cached payload schema or fingerprint inputs change.
_CBGS_CACHE_VERSION = 1


@OBJECT_REGISTRY.register
class CBGSDataset(object):
    """Class-balanced grouping and sampling wrapper for point cloud datasets.

    Building class histograms calls ``dataset.get_cat_ids`` for every sample.
    On large NuScenes lidar LMDBs that can mean tens of GB of JuiceFS IO per
    training launch. The expensive histogram is therefore cached on disk;
    only the cheap random resample runs on every init.
    """

    def __init__(self, dataset, cache_dir=None):
        self.dataset = dataset
        self.CLASSES = dataset.CLASSES
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self._cache_dir_override = cache_dir
        self.sample_indices = self._get_sample_indices()
        if hasattr(self.dataset, "flag"):
            self.flag = np.array(
                [self.dataset.flag[ind] for ind in self.sample_indices],
                dtype=np.uint8,
            )

    def _dataset_root(self):
        return getattr(self.dataset, "root", None) or getattr(
            self.dataset, "data_path", None
        )

    def _cache_fingerprint(self) -> str:
        root = self._dataset_root()
        use_valid_flag = bool(getattr(self.dataset, "use_valid_flag", False))
        payload = {
            "v": _CBGS_CACHE_VERSION,
            "n": len(self.dataset),
            "classes": list(self.CLASSES),
            "use_valid_flag": use_valid_flag,
            "root": os.path.abspath(root) if root else None,
        }
        digest = hashlib.md5(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return digest[:16]

    def _cache_candidates(self):
        """Yield preferred cache file paths (first writable/readable wins)."""
        name = f".cbgs_class_indices_{self._cache_fingerprint()}.pkl"
        dirs = []
        if self._cache_dir_override:
            dirs.append(self._cache_dir_override)
        env_dir = os.environ.get("HAT_CBGS_CACHE_DIR")
        if env_dir:
            dirs.append(env_dir)
        root = self._dataset_root()
        if root:
            dirs.append(root)
        dirs.append(
            os.path.join(
                os.path.expanduser("~"),
                ".cache",
                "heal",
                "cbgs",
            )
        )
        seen = set()
        for directory in dirs:
            if not directory:
                continue
            directory = os.path.abspath(directory)
            if directory in seen:
                continue
            seen.add(directory)
            yield os.path.join(directory, name)

    def _load_class_sample_idxs(self):
        for path in self._cache_candidates():
            if not os.path.isfile(path):
                continue
            try:
                with open(path, "rb") as fid:
                    payload = pickle.load(fid)
                if payload.get("version") != _CBGS_CACHE_VERSION:
                    continue
                if payload.get("num_samples") != len(self.dataset):
                    continue
                class_sample_idxs = payload.get("class_sample_idxs")
                if not isinstance(class_sample_idxs, dict):
                    continue
                logger.info("Loaded CBGS class indices cache from %s", path)
                return class_sample_idxs
            except Exception as exc:  # noqa: BLE001 - corrupt cache is OK
                logger.warning(
                    "Failed to load CBGS cache %s (%s); rebuilding.",
                    path,
                    exc,
                )
        return None

    def _save_class_sample_idxs(self, class_sample_idxs):
        payload = {
            "version": _CBGS_CACHE_VERSION,
            "num_samples": len(self.dataset),
            "class_sample_idxs": class_sample_idxs,
        }
        for path in self._cache_candidates():
            directory = os.path.dirname(path)
            try:
                os.makedirs(directory, exist_ok=True)
                fd, tmp_path = tempfile.mkstemp(
                    prefix=".cbgs_cache_",
                    suffix=".pkl.tmp",
                    dir=directory,
                )
                try:
                    with os.fdopen(fd, "wb") as fid:
                        pickle.dump(
                            payload,
                            fid,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )
                    os.replace(tmp_path, path)
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                logger.info("Saved CBGS class indices cache to %s", path)
                return
            except OSError as exc:
                logger.warning(
                    "Cannot write CBGS cache to %s (%s)",
                    path,
                    exc,
                )

    def _build_class_sample_idxs(self):
        class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()}
        num_samples = len(self.dataset)
        log_every = max(1, num_samples // 20)
        t0 = time.time()
        logger.info(
            "Building CBGS class indices over %d samples "
            "(one-time scan; result is cached for later runs).",
            num_samples,
        )
        for idx in range(num_samples):
            sample_cat_ids = self.dataset.get_cat_ids(idx)
            for cat_id in sample_cat_ids:
                class_sample_idxs[cat_id].append(idx)
            if (idx + 1) % log_every == 0 or (idx + 1) == num_samples:
                elapsed = time.time() - t0
                logger.info(
                    "CBGS class indices: %d/%d (%.1fs)",
                    idx + 1,
                    num_samples,
                    elapsed,
                )
        return class_sample_idxs

    def _get_sample_indices(self):
        class_sample_idxs = self._load_class_sample_idxs()
        if class_sample_idxs is None:
            class_sample_idxs = self._build_class_sample_idxs()
            self._save_class_sample_idxs(class_sample_idxs)

        # Drop empty classes; keep insertion order from the histogram dict.
        nonempty = {k: v for k, v in class_sample_idxs.items() if len(v) > 0}
        if not nonempty:
            raise RuntimeError(
                "CBGSDataset found no category samples; check dataset labels."
            )

        duplicated_samples = sum(len(v) for v in nonempty.values())
        class_distribution = {
            k: len(v) / duplicated_samples for k, v in nonempty.items()
        }

        sample_indices = []
        frac = 1.0 / len(self.CLASSES)
        ratios = [frac / v for v in class_distribution.values()]
        for cls_inds, ratio in zip(nonempty.values(), ratios):
            sample_indices += np.random.choice(
                cls_inds, int(len(cls_inds) * ratio)
            ).tolist()
        return sample_indices

    def __getitem__(self, idx):
        return self.dataset[self.sample_indices[idx]]

    def __len__(self):
        return len(self.sample_indices)
