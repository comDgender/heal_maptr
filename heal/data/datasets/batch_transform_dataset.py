# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import random
from typing import List

from hat_engine.registry import OBJECT_REGISTRY
from torch.utils.data import Dataset

__all__ = ["BatchTransformDataset"]


@OBJECT_REGISTRY.register
class BatchTransformDataset(Dataset):
    """Dataset wrapper that switches transforms at configured epochs.

    Args:
        dataset: Wrapped dataset providing samples.
        transforms_cfgs: Transform pipelines used at different stages.
        epoch_steps: Epoch numbers at which to switch to the next pipeline.
    """

    def __init__(
        self,
        dataset: Dataset,
        transforms_cfgs: List,
        epoch_steps: List,
    ):
        super(BatchTransformDataset, self).__init__()
        self.dataset = dataset
        assert hasattr(
            self.dataset, "__getitem__"
        ), "dataset must have __getitem__"
        self.transforms_steps = transforms_cfgs
        self.epoch_steps = epoch_steps
        self.current_epoch = 0
        self.current_transforms = self.transforms_steps[0]

    def set_epoch(self, epoch):
        """Update current epoch and switch transform pipeline if needed.

        Args:
            epoch: Current training epoch.
        """
        self.current_epoch = epoch
        for i, epoch_step in enumerate(self.epoch_steps):
            if self.current_epoch == epoch_step:
                print("Change transformer to {}".format(i + 1))
                self.current_transforms = self.transforms_steps[i + 1]

    def __len__(self):
        """Return wrapped dataset length."""
        return self.dataset.__len__()

    def __getitem__(self, item):
        """Get one sample and apply current batch-aware transform pipeline.

        Args:
            item: Sample index.

        Returns:
            Transformed dataset sample.
        """
        data = self.dataset.__getitem__(item)
        datalist = [data]
        image_needed = 1
        for trans in self.current_transforms:
            needs = getattr(trans, "images_needed", 1)
            image_needed *= needs
        indexes = [
            random.randint(0, self.__len__() - 1)
            for _ in range(image_needed - 1)
        ]
        for i in indexes:
            data_addition = self.dataset[i]
            datalist.append(data_addition)
        for trans in self.current_transforms:
            new_datalist = []
            needs = getattr(trans, "images_needed", 1)
            for i in range(0, len(datalist), needs):
                if needs == 1:
                    data = trans(datalist[i])
                else:
                    data = trans(datalist[i : i + needs])
                new_datalist.append(data)
            datalist = new_datalist
        return data
