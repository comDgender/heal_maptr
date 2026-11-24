# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
import math
from typing import Optional

import torch
from torch.utils.data.distributed import DistributedSampler

from hat_engine.data.datasets.dataset_wrappers import ConcatDataset
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.distributed import get_dist_info, get_local_process_group

__all__ = ["DistConcatSampler"]


@OBJECT_REGISTRY.register
class DistConcatSampler(DistributedSampler):  # noqa: D205,D400
    """
    The hook API is specifically distributed sampler for concat dataset.
    The results of the sampler will ensure that the same time period is
    only performed on the same subdataset.

    Args:
        dataset: compose dataset
        num_replicas: same as DistributedSampler
        rank: Same as DistributedSampler
        shuffle: if shuffle data
        seed: random seed
        drop_last: if drop last
    """

    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if num_replicas is None or rank is None:
            rank, num_replicas = get_dist_info(get_local_process_group())

        assert isinstance(dataset, ConcatDataset)
        super(DistConcatSampler, self).__init__(
            dataset, num_replicas, rank, shuffle, seed, drop_last
        )  # noqa

        self.total_size = []
        for dataset in self.dataset.datasets:
            if self.drop_last and len(dataset) % self.num_replicas != 0:
                num_samples = math.ceil(
                    (len(dataset) - self.num_replicas) / self.num_replicas
                )
            else:
                num_samples = math.ceil(len(dataset) / self.num_replicas)
            self.total_size.append(num_samples * self.num_replicas)

    def __iter__(self):
        merged = []
        for sub_idx, sub_len in enumerate(self.dataset.len_list):
            # Build the per-subdataset sample order for this epoch.
            if self.shuffle:
                rng = torch.Generator()
                rng.manual_seed(self.seed + self.epoch)
                sub_seq = torch.randperm(sub_len, generator=rng).tolist()
            else:
                sub_seq = list(range(sub_len))

            # Pad or truncate to align across replicas.
            target = self.total_size[sub_idx]
            if self.drop_last:
                sub_seq = sub_seq[:sub_len]
            else:
                shortage = target - len(sub_seq)
                if shortage > 0:
                    reps = math.ceil(shortage / len(sub_seq))
                    sub_seq = (sub_seq * (reps + 1))[: len(sub_seq) + shortage]

            # Pick this rank's share via strided slice.
            rank_seq = sub_seq[self.rank : sub_len : self.num_replicas]

            # Offset local indices by the subdataset's cumulative start.
            if sub_idx > 0:
                offset = self.dataset.cumulative_sizes[sub_idx - 1]
                rank_seq = [v + offset for v in rank_seq]

            merged += rank_seq
        return iter(merged)
