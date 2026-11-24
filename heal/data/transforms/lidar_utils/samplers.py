# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import Any, Dict, List

import numpy as np


class BatchSampler:
    """Sample batch data.

    Args:
        sampled_list (list): The data list.
        name (str): Dataset name.
        epoch(int): Current epoch.
        shuffle (bool): Shuffle the dataset.
        drop_reminder (bool): Drop last reminder data.
    """

    def __init__(
        self,
        sampled_list: List[Any],
        name: str = None,
        epoch: int = 0,
        shuffle: bool = True,
        drop_reminder: bool = False,
    ):
        self._sampled_list = sampled_list
        n = len(sampled_list)
        order = np.arange(n)
        if shuffle:
            np.random.shuffle(order)
        self._indices = order
        self._idx = 0
        self._example_num = n
        self._name = name
        self._shuffle = shuffle
        self._epoch = epoch
        self._epoch_counter = 0
        self._drop_reminder = drop_reminder

    def _sample(self, num: int) -> Dict:
        cursor = self._idx
        total = self._example_num
        # Wrap when the request reaches or passes the end of the epoch.
        if cursor + num >= total:
            picked = np.array(self._indices[cursor:total], copy=True)
            self._reset()
            return picked
        picked = self._indices[cursor : cursor + num]
        self._idx = cursor + num
        return picked

    def _reset(self):
        if self._shuffle:
            np.random.shuffle(self._indices)
        self._idx = 0

    def sample(self, num: int) -> Dict:
        idxs = self._sample(num)
        items = self._sampled_list
        return [items[i] for i in idxs]


class DataBasePreprocessor:
    """
    The PreprocesserBase class.

    Args:
        preprocessors (class): preprocessors list.
    """

    def __init__(self, preprocessors):
        self._preprocessors = preprocessors

    def __call__(self, db_infos):
        for prepor in self._preprocessors:
            db_infos = prepor(db_infos)
        return db_infos
