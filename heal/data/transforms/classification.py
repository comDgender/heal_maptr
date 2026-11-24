# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import logging

from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.package_helper import require_packages

try:
    import timm
except ImportError:
    timm = None
logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class TimmTransforms(object):
    """Apply timm transforms to ``data['img']``.

    Args:
        *args: Positional arguments passed to ``timm.data.create_transform``.
        **kwargs: Keyword arguments passed to
            ``timm.data.create_transform``.
    """

    @require_packages("timm")
    def __init__(self, *args, **kwargs):
        self.transform = timm.data.create_transform(*args, **kwargs)

    def __call__(self, data):
        """Apply the configured transform pipeline in-place.

        Args:
            data: Sample dict containing key ``img``.

        Returns:
            dict: Updated sample dict.
        """
        data["img"] = self.transform(data["img"])
        return data


@OBJECT_REGISTRY.register
class TimmMixup(object):
    """Apply timm mixup to ``data['img']`` and ``data['labels']``.

    Args:
        *args: Positional arguments passed to ``timm.data.Mixup``.
        **kwargs: Keyword arguments passed to ``timm.data.Mixup``.
    """

    @require_packages("timm")
    def __init__(self, *args, **kwargs):
        self.mixup = timm.data.Mixup(*args, **kwargs)

    def __call__(self, data):
        """Apply mixup when batch size is even.

        Args:
            data: Sample dict containing ``img`` and ``labels``.

        Returns:
            dict: Updated sample dict.
        """
        x, target = data["img"], data["labels"]
        if len(x) % 2 == 0:
            x, target = self.mixup(x, target)
            data["img"], data["labels"] = x, target
        else:
            logger.warning(
                "Batch size should be even when using Mixup)"
                f"get batchsize={len(x)}. Skip Mixup"
            )
        return data
