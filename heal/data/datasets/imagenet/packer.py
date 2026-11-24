# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
"""Pack raw ImageNet into LMDB/msgpack for classification training."""

import logging
from typing import Optional

import cv2
import msgpack
import numpy as np
from hat_engine.data.datasets.data_packer import Packer
from hat_engine.utils.package_helper import require_packages

try:
    import torchvision
except ImportError:
    torchvision = None

logger = logging.getLogger(__name__)

__all__ = ["ImageNetPacker"]


def _loader(data_path):
    img = cv2.imread(data_path)
    img = cv2.imencode(".JPEG", img)[1]
    img = np.asarray(img).astype(np.uint8).tobytes()
    return img


class ImageNetPacker(Packer):
    """Pack raw ImageNet into LMDB/msgpack.

    For :class:`hat_engine.data.datasets.imagenet.ImageNet`.

    Args:
        src_data_dir: Raw ImageNet root directory.
        target_data_dir: Output pack directory.
        split_name: ``train`` or ``val``.
        num_workers: Parallel workers for packing.
        pack_type: Pack backend (e.g. ``lmdb``).
        num_samples: Optional cap on samples to pack; pack all if None.
        **kwargs: Extra kwargs for :class:`Packer`.
    """

    @require_packages("torchvision")
    def __init__(
        self,
        src_data_dir: str,
        target_data_dir: str,
        split_name: str,
        num_workers: int,
        pack_type: str,
        num_samples: Optional[int] = None,
        **kwargs,
    ):
        self.dataset = torchvision.datasets.ImageNet(
            root=src_data_dir, split=split_name, loader=_loader
        )
        if num_samples is None:
            num_samples = len(self.dataset)
        logger.info(
            "Packing ImageNet %s, total samples %d",
            split_name,
            num_samples,
        )
        super().__init__(
            target_data_dir, num_samples, pack_type, num_workers, **kwargs
        )

    def pack_data(self, idx):
        image, label = self.dataset[idx]
        label = np.uint16(label).tobytes()
        return msgpack.packb(image + label, use_bin_type=True)
