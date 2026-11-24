# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
"""Generic dataset that merges several index-aligned datasets by a join key."""

import logging
from typing import Callable, List, Optional

from hat_engine.registry import OBJECT_REGISTRY
from torch.utils.data import Dataset

from heal.data.datasets.nuscenes.nuscenes_occ_utils import get_lidar_occ_info

logger = logging.getLogger(__name__)

__all__ = ["MergeDataset"]


@OBJECT_REGISTRY.register
class MergeDataset(Dataset):
    """Merge several index-aligned datasets into one sample by a join key.

    Each child dataset returns a dict for the same integer index ``i``. This
    wrapper merges those dicts (later datasets override earlier keys on a key
    clash), validates that every child agrees on ``join_key`` for that index,
    and finally runs a single shared ``transforms`` pipeline on the merged
    dict. It is dataset agnostic: any datasets producing index-aligned dicts
    that share a join key can be combined (e.g. a base camera dataset + a
    separate occ dataset, or base + occ + another task's dataset).

    Children are aligned by integer index, so they must be produced from the
    same underlying samples in the same order (e.g. packed from the same
    NuScenes version + split); alignment is validated per sample via
    ``join_key``.

    Args:
        datasets: Child datasets (already built by the registry), all of equal
            length.
        join_key: Key whose value must match across all children for a given
            index. Resolved at the dict top level, falling back to
            ``sample["meta"][join_key]`` (some datasets nest it in ``meta``).
        transforms: Optional transform pipeline applied once to the merged
            sample. Child datasets should leave their own ``transforms`` unset.
        with_lidar_occ: When True, remap ego-frame occ GT to lidar frame after
            transforms (requires ``lidar2global`` and ``cam_ego2global`` from
            the base BEV dataset).
    """

    def __init__(
        self,
        datasets: List[Dataset],
        join_key: str = "sample_token",
        transforms: Optional[Callable] = None,
        with_lidar_occ: bool = False,
    ):
        assert len(datasets) >= 1, "MergeDataset needs at least one dataset."
        self.datasets = datasets
        self.join_key = join_key
        self.transforms = transforms
        self.with_lidar_occ = with_lidar_occ
        lengths = [len(d) for d in datasets]
        assert len(set(lengths)) == 1, (
            f"All merged datasets must have equal length, got {lengths}. "
            "Child datasets must come from the same samples + order."
        )
        self._length = lengths[0]

    def __len__(self):
        return self._length

    def _meta_source_dataset(self):
        for dataset in self.datasets:
            if hasattr(dataset, "get_meta"):
                return dataset
        raise AttributeError(
            "MergeDataset has no child with get_meta(); "
            "required by stream samplers (e.g. DistStreamBatchSampler)."
        )

    def get_meta(self):
        return self._meta_source_dataset().get_meta()

    def get_aug(self):
        aug_cfg = {}
        transform_pipelines = []
        if self.transforms is not None:
            transform_pipelines.append(self.transforms)
        else:
            for dataset in self.datasets:
                child_transforms = getattr(dataset, "transforms", None)
                if child_transforms is not None:
                    transform_pipelines.append(child_transforms)
        for transforms in transform_pipelines:
            if hasattr(transforms, "transforms"):
                for trans in transforms.transforms:
                    if hasattr(trans, "get_aug"):
                        name, cfg = trans.get_aug()
                        aug_cfg[name] = cfg
        return aug_cfg

    def set_epoch(self, epoch):
        for dataset in self.datasets:
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)

    def _lookup_join_key(self, sample):
        if self.join_key in sample:
            return sample[self.join_key]
        meta = sample.get("meta")
        if isinstance(meta, dict) and self.join_key in meta:
            return meta[self.join_key]
        return None

    def __getitem__(self, item):
        if isinstance(item, dict):
            idx = item["idx"]
            fetch_keys = [item] + [item["idx"]] * (len(self.datasets) - 1)
        else:
            idx = item
            fetch_keys = [idx] * len(self.datasets)

        merged = {}
        ref_key = None
        cam_ego2global = None
        for i, (dataset, fetch_key) in enumerate(
            zip(self.datasets, fetch_keys)
        ):
            sample = dataset[fetch_key]
            if i == 0:
                cam_ego2global = sample.get("cam_ego2global")
            cur_key = self._lookup_join_key(sample)
            if i == 0:
                ref_key = cur_key
            elif (
                ref_key is not None
                and cur_key is not None
                and cur_key != ref_key
            ):
                raise RuntimeError(
                    f"'{self.join_key}' mismatch at idx {idx}: "
                    f"datasets[0]={ref_key!r}, datasets[{i}]={cur_key!r}. "
                    "Merged datasets are misaligned; re-produce them from the "
                    "same samples + order (e.g. same version + split)."
                )
            merged.update(sample)
        if self.transforms is not None:
            merged = self.transforms(merged)
        if self.with_lidar_occ and "gt_occ_info" in merged:
            merged["gt_occ_info"] = get_lidar_occ_info(
                merged["gt_occ_info"],
                merged["lidar2global"],
                cam_ego2global,
            )
        return merged
