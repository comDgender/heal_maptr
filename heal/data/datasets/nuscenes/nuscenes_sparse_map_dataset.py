# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
"""NuScenes map dataset for sparse MapTR (single-frame)."""

from hat_engine.registry import OBJECT_REGISTRY

from heal.data.datasets.nuscenes.nuscenes_map_dataset import NuscenesMapDataset

__all__ = ["NuscenesSparseMapDataset"]


@OBJECT_REGISTRY.register
class NuscenesSparseMapDataset(NuscenesMapDataset):
    """Single-frame NuScenes map dataset for sparse MapTR.

    Lidar points for dense-depth aux loss should be merged via
    :class:`heal.data.datasets.merge_dataset.MergeDataset` together with
    :class:`heal.data.datasets.nuscenes.nuscenes_lidar.NuscenesLidarDataset`.
    """

    def __getitem__(self, item):
        if isinstance(item, dict):
            idx = item["idx"]
            aug_cfg = item["aug"]
        else:
            idx = item
            aug_cfg = None
        if self.test_mode:
            return self.prepare_test_data(idx)[0]
        while True:
            data = self.prepare_train_data(idx, aug_cfg)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data[0]
