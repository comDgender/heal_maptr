# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
import copy
import itertools
import logging
import pickle
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.distributed import get_dist_info
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

__all__ = ["DistStreamBatchSampler", "DistStreamFilterBatchSampler"]

logger = logging.getLogger(__name__)


def sync_random_seed(seed: int = None, device: str = "cuda") -> int:
    """
    Synchronize a random seed across distributed training processes.

    Args:
    seed : Seed value to synchronize.
           If None, a random seed will be generated. Default is None.
    device : Device type for the seed tensor. Default is "cuda".

    Returns:
        Synchronized random seed across all processes.
    """
    if seed is None:
        seed = np.random.randint(2**31)
    assert isinstance(seed, int)

    rank, world_size = get_dist_info()

    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


@OBJECT_REGISTRY.register
class DistStreamBatchSampler(Sampler):
    """
    Distributed stream batch sampler.

    Args:
        dataset: The dataset from which to sample batches.
        batch_size: Batch size per GPU. Default is 1.
        seed : Random seed for synchronization across processes.
               Default is 0.
        skip_prob : Probability of skipping a sample.
                    Default is 0.5.
        max_skip_num : Maximum number of samples to skip.
                       Default is 1.
        sequence_flip_prob : Probability of flipping the sequence.
                             Default is 0.1.
        keep_consistent_seq_aug : Whether to keep consistent sequence
                                  augmentation across samples.
                                  Default is True.
    """

    def __init__(
        self,
        dataset,
        batch_size: Dataset = 1,
        seed: int = 0,
        skip_prob: float = 0.5,
        max_skip_num: int = 1,
        sequence_flip_prob: float = 0.1,
        keep_consistent_seq_aug: bool = True,
        reverse: bool = False,
    ) -> None:

        rank, world_size = get_dist_info()

        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.seed = sync_random_seed(seed)
        self.keep_consistent_seq_aug = keep_consistent_seq_aug

        self.size = len(self.dataset)
        meta = self.dataset.get_meta()
        self.scene_info = meta["scene_info"]

        self.groups_num = len(self.scene_info)
        self.global_batch_size = batch_size * world_size
        assert self.groups_num >= self.global_batch_size

        # Contiguous sample-index ranges for each scene group.
        self.group_idx_to_sample_idxs = {}
        cursor = 0
        for group_id, (_, length) in enumerate(self.scene_info.items()):
            nxt = cursor + length
            if reverse:
                self.group_idx_to_sample_idxs[group_id] = list(
                    range(cursor, nxt)
                )[::-1]
            else:
                self.group_idx_to_sample_idxs[group_id] = list(
                    range(cursor, nxt)
                )
            cursor = nxt

        # One interleaved group stream per local batch slot.
        self.group_indices_per_global_sample_idx = [
            self._group_indices_per_global_sample_idx(
                self.rank * self.batch_size + slot
            )
            for slot in range(self.batch_size)
        ]
        self.buffer_per_local_sample = [[] for _ in range(self.batch_size)]
        self.aug_per_local_sample = [None] * self.batch_size
        self.skip_prob = skip_prob
        self.sequence_flip_prob = sequence_flip_prob
        self.max_skip_num = max_skip_num

    def _infinite_group_indices(self):
        """
        Infinite generator yielding shuffled indices.

        Yields:
            Shuffled group index.
        """
        rng = torch.Generator()
        rng.manual_seed(self.seed)
        while True:
            order = torch.randperm(self.groups_num, generator=rng).tolist()
            yield from order

    def _group_indices_per_global_sample_idx(self, global_sample_idx: int):
        """
        Generate a stream of group indices for a specific global sample index.

        Args:
            global_sample_idx : Global index across all GPUs.

        Yields:
            Group index for batch sampling.
        """
        stream = self._infinite_group_indices()
        # Stride so each local slot sees a disjoint group stream.
        yield from itertools.islice(
            stream, global_sample_idx, None, self.global_batch_size
        )

    def __iter__(self):
        """
        Iterate indefinitely over the dataset to yield batches of samples.

        Yields:
            A batch of samples, each represented as a dictionary
            with keys 'idx' and 'aug'.
        """
        while True:
            batch_items = []
            for slot in range(self.batch_size):
                slot_buf = self.buffer_per_local_sample[slot]
                if not slot_buf:
                    # Refill from the next group assigned to this slot.
                    group_id = next(
                        self.group_indices_per_global_sample_idx[slot]
                    )
                    filled = copy.deepcopy(
                        self.group_idx_to_sample_idxs[group_id]
                    )
                    if np.random.uniform() < self.sequence_flip_prob:
                        filled = filled[::-1]
                    self.buffer_per_local_sample[slot] = filled
                    slot_buf = filled
                    if self.keep_consistent_seq_aug:
                        self.aug_per_local_sample[slot] = self.get_aug()

                if not self.keep_consistent_seq_aug:
                    self.aug_per_local_sample[slot] = self.get_aug()
                skip_cnt = 0
                while (
                    np.random.uniform() < self.skip_prob
                    and len(self.buffer_per_local_sample[slot]) > 1
                    and skip_cnt < self.max_skip_num
                ):
                    self.buffer_per_local_sample[slot].pop(0)
                    skip_cnt += 1

                batch_items.append(
                    {
                        "idx": self.buffer_per_local_sample[slot].pop(0),
                        "aug": self.aug_per_local_sample[slot],
                    }
                )
            yield batch_items

    def __len__(self) -> int:
        """Length of base dataset."""
        return self.size

    def set_epoch(self, epoch: int) -> None:
        """Set epoch."""
        self.epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def get_aug(self) -> Dict:
        """
        Return the augmentation parameters for the current sample.

        Returns:
            Augmentation parameters for the current sample.
        """
        return self.dataset.get_aug()


@OBJECT_REGISTRY.register
class DistStreamFilterBatchSampler(DistStreamBatchSampler):
    """Distributed stream batch sampler with token-based filtering.

    Extends ``DistStreamBatchSampler`` to filter out perception samples
    whose token does not appear in the planner dataset. This ensures
    only samples with both perception and planning GT are used in
    end-to-end joint training.

    Filtering logic:
    1. Load ``planner_token_to_idx`` pkl → set of planner tokens.
    2. Load ``perception_token_to_idx`` pkl → dict of
       ``{token: dataset_idx}``.
    3. Compute intersection: perception tokens that also exist in
       the planner set.
    4. Build a filtered ``scene_info`` that only contains valid
       samples, so the inherited stream sampling logic works
       transparently.
    """

    def __init__(
        self,
        dataset,
        batch_size: Dataset = 1,
        seed: int = 0,
        skip_prob: float = 0.5,
        max_skip_num: int = 1,
        sequence_flip_prob: float = 0.1,
        keep_consistent_seq_aug: bool = True,
        reverse: bool = False,
        planner_token_to_idx_path: Optional[str] = None,
        perception_token_to_idx_path: Optional[str] = None,
    ) -> None:

        rank, world_size = get_dist_info()

        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.seed = sync_random_seed(seed)
        self.keep_consistent_seq_aug = keep_consistent_seq_aug

        self.size = len(self.dataset)
        meta = self.dataset.get_meta()
        self.scene_info = meta["scene_info"]

        # Token-based filtering
        self.valid_sample_idxs = self._build_filtered_indices(
            planner_token_to_idx_path,
            perception_token_to_idx_path,
        )
        (
            self.scene_info,
            group_idx_to_original,
        ) = self._filter_scene_info(self.valid_sample_idxs)

        self.groups_num = len(self.scene_info.keys())
        self.global_batch_size = batch_size * world_size
        assert self.groups_num >= self.global_batch_size, (
            f"After filtering, {self.groups_num} groups remain, "
            f"but global_batch_size is {self.global_batch_size}. "
            f"Please check your token pkl files."
        )

        # Build group_idx → List[original dataset sample_idxs]
        self.group_idx_to_sample_idxs = {}
        for idx, original_idxs in group_idx_to_original.items():
            if reverse:
                self.group_idx_to_sample_idxs[idx] = original_idxs[::-1]
            else:
                self.group_idx_to_sample_idxs[idx] = original_idxs

        # Per-sample generators and buffers (same as parent)
        self.group_indices_per_global_sample_idx = [
            self._group_indices_per_global_sample_idx(
                self.rank * self.batch_size + local_sample_idx
            )
            for local_sample_idx in range(self.batch_size)
        ]
        self.buffer_per_local_sample = [[] for _ in range(self.batch_size)]
        self.aug_per_local_sample = [None for _ in range(self.batch_size)]
        self.skip_prob = skip_prob
        self.sequence_flip_prob = sequence_flip_prob
        self.max_skip_num = max_skip_num

    def _build_filtered_indices(
        self,
        planner_token_to_idx_path: Optional[str],
        perception_token_to_idx_path: Optional[str],
    ):
        """Compute the set of valid dataset indices via token intersection.

        Returns a sorted list of dataset indices whose perception token
        also exists in the planner token set, or ``None`` if either
        token file path is missing (falls back to no filtering).
        """
        if planner_token_to_idx_path is None:
            logger.warning(
                "planner_token_to_idx_path is None, "
                "no filtering will be applied."
            )
            return None

        with open(planner_token_to_idx_path, "rb") as f:
            planner_token_to_idx = pickle.load(f)
        planner_tokens = set(planner_token_to_idx.keys())

        if perception_token_to_idx_path is None:
            logger.warning(
                "perception_token_to_idx_path is None, "
                "no filtering will be applied."
            )
            return None

        with open(perception_token_to_idx_path, "rb") as f:
            perception_token_to_idx = pickle.load(f)

        valid_tokens = set(perception_token_to_idx.keys()) & planner_tokens
        valid_indices = sorted(
            [perception_token_to_idx[t] for t in valid_tokens]
        )

        logger.info(
            f"Token filtering: planner={len(planner_tokens)}, "
            f"perception={len(perception_token_to_idx)}, "
            f"intersection={len(valid_tokens)}, "
            f"filtered_indices={len(valid_indices)}"
        )
        return valid_indices

    def _filter_scene_info(self, valid_indices):
        """Rebuild scene_info + group mapping keeping only valid samples.

        Returns (new_scene_info, group_idx_to_original_sample_idxs).
        """
        if valid_indices is None:
            group_idx_to_original = {}
            start = 0
            for idx, (_, v) in enumerate(self.scene_info.items()):
                group_idx_to_original[idx] = list(range(start, start + v))
                start += v
            return self.scene_info, group_idx_to_original

        valid_set = set(valid_indices)
        new_scene_info = {}
        group_idx_to_original = {}
        start = 0
        new_group_idx = 0
        for name, count in self.scene_info.items():
            end = start + count
            valid_in_group = [i for i in range(start, end) if i in valid_set]
            if len(valid_in_group) > 0:
                new_scene_info[name] = len(valid_in_group)
                group_idx_to_original[new_group_idx] = valid_in_group
                new_group_idx += 1
            start = end

        return new_scene_info, group_idx_to_original
