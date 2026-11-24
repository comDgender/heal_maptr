# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import chain
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data as data

from hat_engine.data.datasets.dataset_wrappers import ConcatDataset
from hat_engine.registry import OBJECT_REGISTRY, build_from_registry
from hat_engine.utils.distributed import get_dist_info
from hat_engine.utils.global_var import get_value

__all__ = ["RankSplitDataset", "RankSplitDataLoader"]

logger = logging.getLogger(__name__)


def find_index(len_list, index):
    idx, offset = 0, index
    for i, ds_len in enumerate(len_list):
        if offset >= ds_len:
            offset -= ds_len
        else:
            idx = i
            break
    return idx, offset


@OBJECT_REGISTRY.register
class RankSplitDataset(data.Dataset):
    """RankSplitDataset is used for split concat dataset on different rank.

    RankSplitDataset must be used with RankSplitDataloader and it can
    avoids consuming excessive MEM and Virtual MEM resources,
    get more information from https://horizonrobotics.feishu.cn/wiki/wikcnALBu8Y6CkqQewtFIhANGGe.  # noqa

    Args:
        datasets: A list of dataset cfg.
        with_flag: Whether to concatenate datasets flags.
            If True, concatenate all datasets flag (
            all datasets must has flag attribute in this case).
            Default to False.
        max_build_threads: The max number of threads to build datasets.

    """

    def __init__(
        self,
        datasets: List[dict],
        with_flag: bool = False,
        max_build_threads: int = 10,
    ):
        self.len_list = []
        flag_list = []
        datasets_build = []
        tmp_len_list = []
        self.rank, self.world_size = get_dist_info()

        # re-range datasets to parallel build
        num_build_datasets = len(datasets)

        # avoid build redundantly in the same rank
        # ------------------------------------------------------------------
        # First: if num_build_datasets < max_build_threads * self.world_size
        # only use multi-thread to build datasets.
        # ------------------------------------------------------------------
        # Second: if num_build_datasets >= max_build_threads * self.world_size
        # use multi-thread and multi-process to build datasets.
        # ------------------------------------------------------------------
        world_size = (
            self.world_size
            if num_build_datasets >= max_build_threads * self.world_size
            else 1
        )
        rank = (
            self.rank
            if num_build_datasets >= max_build_threads * self.world_size
            else 0
        )
        chuck_size = (
            num_build_datasets // world_size
            if world_size > 1
            else num_build_datasets
        )
        left = rank * chuck_size
        right = min((rank + 1) * chuck_size, num_build_datasets)
        thread_chuck_datasets = datasets[left:right]
        already_built_index = list(range(left, right))

        # multithread build datasets
        thread_pool = ThreadPoolExecutor(max_workers=max_build_threads)
        results = thread_pool.map(
            partial(self._get_total_clip_len, with_flag=with_flag),
            thread_chuck_datasets,
            timeout=1800,
        )
        for clip_len, build_dataset, flag in results:
            tmp_len_list.append(clip_len)
            datasets_build.append(build_dataset)
            if with_flag:
                flag_list.append(flag)

        # sync all results
        if dist.is_initialized() and world_size > 1:
            global_len_output = [[] for _ in range(self.world_size)]
            dist.all_gather_object(global_len_output, tmp_len_list)
            self.len_list = list(chain(*global_len_output))
            if with_flag:
                global_flag_output = [[] for _ in range(self.world_size)]
                dist.all_gather_object(global_flag_output, flag_list)
                self.flag_list = np.concatenate(
                    list(chain(*global_flag_output))
                )
        else:
            self.len_list = tmp_len_list
            if with_flag:
                self.flag_list = np.concatenate(flag_list)

        self.total_len = sum(self.len_list)
        # Split datasets with world_size,
        # To balance the number of dataitem in different ranks, we split
        # datasets by len(dataset) instead of dataset instance.

        if self.total_len % self.world_size != 0:
            world_step = math.ceil(
                (self.total_len - self.world_size) / float(self.world_size)
            )

        else:
            world_step = math.ceil(self.total_len / float(self.world_size))

        if self.rank == self.world_size - 1:
            sub_index = list(range(sum(self.len_list)))[
                world_step * self.rank :
            ]
        else:
            sub_index = list(range(sum(self.len_list)))[
                world_step * self.rank : world_step * (self.rank + 1)
            ]

        # find datasets by split index
        ds_begin, _ = find_index(self.len_list, sub_index[0])
        ds_end, _ = find_index(self.len_list, sub_index[-1])
        ds_end += 1

        # avoid build redundantly
        truly_need_datasets = []
        for i in range(ds_begin, ds_end):
            if i in already_built_index:
                truly_need_datasets.append(
                    datasets_build[already_built_index.index(i)]
                )
            else:
                truly_need_datasets.append(build_from_registry(datasets[i]))
        self.datasets = ConcatDataset(truly_need_datasets)

    @staticmethod
    def _get_total_clip_len(
        dataset: Dict[str, Any], with_flag: bool = False
    ) -> Tuple[List[int], List[torch.utils.data.Dataset]]:
        dataset = build_from_registry(dataset)
        clip_len = len(dataset)

        if with_flag:
            assert hasattr(dataset, "flag"), "dataset must has group flag"
            assert isinstance(
                dataset.flag, np.ndarray
            ), "dataset flag must is numpy array instance"
            assert (
                len(dataset) == dataset.flag.shape[0]
            ), "dataset flag length at axis 0 must equal to the dataset length"  # noqa: E501
            flag = dataset.flag
        else:
            flag = None
        return clip_len, dataset, flag

    def __len__(self):
        # this is fake length, real length will be sampled in Sampler.
        return self.total_len

    def __getitem__(self, idx):
        return self.datasets[idx]


class RankSplitDataLoaderIter(data.dataloader._MultiProcessingDataLoaderIter):
    def __init__(self, loader):

        super(RankSplitDataLoaderIter, self).__init__(loader)

    def _try_put_index(self):
        assert (
            self._tasks_outstanding < self._prefetch_factor * self._num_workers
        )

        try:
            index = self._next_index()
        except StopIteration:
            return

        # hardcode batch_size, its error while batchsize < len(dataset)
        base_seed = 0

        shuffle = True
        try:
            shuffle = self._index_sampler.sampler.shuffle
        except BaseException:
            pass

        if self._send_idx == 0:
            self._real_send_idx = self._send_idx
            self._batch_size = len(index)
            self._generator = torch.Generator()
            self._generator.manual_seed(base_seed)
            # base_seed += 1

            # split dataset
            rank, world_size = self._dataset.rank, self._dataset.world_size

            total_len = self._dataset.total_len
            if total_len % world_size != 0:
                world_step = math.ceil(
                    (total_len - world_size) / float(world_size)
                )
            else:
                world_step = math.ceil(total_len / float(world_size))

            sub_index = range(sum(self._dataset.len_list))[
                world_step * rank : world_step * (rank + 1)
            ]
            cur_num = len(sub_index)

            global_index = world_step * rank
            _, offset = find_index(self._dataset.len_list, global_index)
            self._offset = offset

            if hasattr(self._dataset, "flag_list"):
                self._flag = self._dataset.flag_list[sub_index]
                group_sizes = np.bincount(self._flag)
                num_samples = 0
                for size in group_sizes:
                    num_samples += (
                        int(math.ceil(size * 1.0 / self._batch_size))
                        * self._batch_size
                    )

                indices = []
                for i, size in enumerate(group_sizes):
                    if size <= 0:
                        continue

                    indice = np.where(self._flag == i)[0]
                    assert len(indice) == size
                    # add .numpy() to avoid bug when selecting indice in parrots.  # noqa
                    if shuffle:
                        indice = indice[
                            list(
                                torch.randperm(
                                    int(size), generator=self._generator
                                ).numpy()
                            )
                        ].tolist()
                    else:
                        indice = indice.tolist()
                    extra = int(
                        math.ceil(size * 1.0 / self._batch_size)
                    ) * self._batch_size - len(indice)
                    # pad indice
                    tmp = indice.copy()
                    for _ in range(extra // size):
                        indice.extend(tmp)
                    indice.extend(tmp[: extra % size])
                    indices.extend(indice)

                if shuffle:
                    random_index = [
                        indices[j]
                        for i in list(
                            torch.randperm(
                                len(indices) // self._batch_size,
                                generator=self._generator,
                            )
                        )
                        for j in range(
                            i * self._batch_size, (i + 1) * self._batch_size
                        )
                    ]
                else:
                    random_index = [
                        indices[j]
                        for i in range(len(indices) // self._batch_size)
                        for j in range(
                            i * self._batch_size, (i + 1) * self._batch_size
                        )
                    ]
            else:
                if shuffle:
                    random_index = torch.randperm(
                        cur_num, generator=self._generator
                    ).tolist()
                else:
                    random_index = random_index.tolist()
            self._random_indexs = random_index

        for _ in range(
            self._num_workers
        ):  # find the next active worker, if any
            worker_queue_idx = next(self._worker_queue_idx_cycle)
            if self._workers_status[worker_queue_idx]:
                break
        else:
            # not found (i.e., didn't break)
            return

        # find worker idx of current idx
        cur_offset = self._send_idx

        begin = int(cur_offset * self._batch_size)

        # resume index
        if (
            get_value("dataloader_start_iter") is not None
            and get_value("dataloader_start_iter") > 0
        ):
            start_iter = get_value("dataloader_start_iter")
            begin = begin + int(start_iter * self._batch_size)

        end = begin + self._batch_size

        cur_index = [
            idx + self._offset for idx in self._random_indexs[begin:end]
        ]

        self._index_queues[worker_queue_idx].put((self._send_idx, cur_index))
        self._task_info[self._send_idx] = (worker_queue_idx,)
        self._tasks_outstanding += 1
        self._send_idx += 1


@OBJECT_REGISTRY.register
class RankSplitDataLoader(data.dataloader.DataLoader):
    """RankSplitDataset must be used with RankSplitDataloader.

    RankSplitDataloader only supports DistributedSampler
    and DistributedGroupSampler now.
    """

    def __init__(self, *args, **kwargs):
        super(RankSplitDataLoader, self).__init__(*args, **kwargs)
        assert self.num_workers > 0
        assert self.drop_last
        if hasattr(self.sampler, "drop_last"):
            assert self.sampler.drop_last

    def _get_iterator(self):
        self.check_worker_number_rationality()
        return RankSplitDataLoaderIter(self)
