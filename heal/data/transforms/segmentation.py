# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from numbers import Real
from typing import Dict, Sequence, Tuple, Union

import numpy as np
import torch
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.package_helper import require_packages
from torch import Tensor
from torch.nn import functional as torchFunctional

from heal.data.transforms.functional_target import one_hot

try:
    from torchvision.transforms import InterpolationMode, RandomAffine, Resize
    from torchvision.transforms import functional as visionFunctional
except ImportError:
    RandomAffine = object
    Resize = object
    InterpolationMode = None
    visionFunctional = None

try:
    from torchvision.transforms.functional import get_image_size
except ImportError:
    from torchvision.transforms.functional import (
        _get_image_size as get_image_size,
    )

__all__ = [
    "LabelRemap",
    "SegOneHot",
    "SegResize",
    "SegRandomAffine",
    "Scale",
]


@OBJECT_REGISTRY.register
class LabelRemap(object):
    """Remap segmentation label indices.

    Args:
        mapping: Sequence mapping old label ids to new label ids.
    """

    def __init__(self, mapping: Sequence):
        if not isinstance(mapping, Sequence):
            raise TypeError(
                "mapping should be a sequence. Got {}".format(type(mapping))
            )
        self.mapping = mapping

    def __call__(self, data: Tensor):
        """Apply label remapping to ``gt_seg``.

        Args:
            data: Sample dict containing ``gt_seg``.

        Returns:
            dict: Sample dict with remapped segmentation labels.
        """
        label = data["gt_seg"]
        if isinstance(label, torch.Tensor):
            mapping = torch.tensor(
                self.mapping, dtype=label.dtype, device=label.device
            )
            data["gt_seg"] = mapping[label.to(dtype=torch.long)]
        else:
            mapping = np.array(self.mapping, dtype=label.dtype)
            data["gt_seg"] = mapping[label]
        return data


@OBJECT_REGISTRY.register
class SegOneHot(object):
    """Convert segmentation labels to one-hot format.

    Args:
        num_classes: Number of segmentation classes.
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def __call__(self, data):
        """Convert ``gt_seg`` label ids to one-hot tensors.

        Args:
            data: Sample dict containing ``gt_seg``.

        Returns:
            dict: Sample dict with one-hot encoded segmentation labels.
        """
        ndim = data["gt_seg"].ndim
        if ndim == 3 or ndim == 2:
            data["gt_seg"] = torch.unsqueeze(data["gt_seg"], 0)
        data["gt_seg"] = one_hot(data["gt_seg"], self.num_classes)
        if ndim == 3 or ndim == 2:
            data["gt_seg"] = data["gt_seg"][0]
        return data


@OBJECT_REGISTRY.register
class SegResize(Resize):
    """Resize image and segmentation label together.

    Args:
        size: Target spatial size.
        interpolation: Interpolation mode used for image resizing.
    """

    @require_packages("torchvision")
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        super().__init__(size, interpolation)

    def __call__(self, data):
        """Resize ``img`` and optional ``gt_seg``.

        Args:
            data: Sample dict containing image and optional segmentation label.

        Returns:
            dict: Sample dict with resized tensors.
        """
        data["img"] = super().__call__(data["img"])
        if "gt_seg" in data:
            data["gt_seg"] = super().__call__(data["gt_seg"])
        return data


@OBJECT_REGISTRY.register
class SegRandomAffine(RandomAffine):
    """Random affine for image and segmentation label.

    Args:
        degrees: Rotation degree range.
        translate: Optional translation range.
        scale: Optional scale range.
        shear: Optional shear range.
        interpolation: Interpolation mode for image transform.
        fill: Fill value for image transform.
        label_fill_value: Fill value used for segmentation labels.
        rotate_p: Probability of enabling random rotation.
        translate_p: Probability of enabling random translation.
        scale_p: Probability of enabling random scaling.
    """

    @require_packages("torchvision")
    def __init__(
        self,
        degrees: Union[Sequence, float] = 0,
        translate: Tuple = None,
        scale: Tuple = None,
        shear: Union[Sequence, float] = None,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Union[tuple, int] = 0,
        label_fill_value: Union[tuple, int] = -1,
        rotate_p: float = 1.0,
        translate_p: float = 1.0,
        scale_p: float = 1.0,
    ):
        super().__init__(degrees, translate, scale, shear, interpolation, fill)
        self.label_fill_value = label_fill_value
        self.rotate_p = rotate_p
        self.translate_p = translate_p
        self.scale_p = scale_p

    def __call__(self, data: Dict[str, Tensor]):
        """Apply one sampled affine transform to image and label.

        Args:
            data: Sample dict containing ``img`` and optional ``gt_seg``.

        Returns:
            dict: Sample dict after affine transform.
        """
        img = data["img"]
        img_size = get_image_size(img)

        rotate_flag = np.random.choice(
            [False, True], p=[1 - self.rotate_p, self.rotate_p]
        )
        translate_flag = np.random.choice(
            [False, True], p=[1 - self.translate_p, self.translate_p]
        )
        scale_flag = np.random.choice(
            [False, True], p=[1 - self.scale_p, self.scale_p]
        )
        params = [
            self.degrees,
            self.translate,
            self.scale,
            self.shear,
            img_size,
        ]
        if not rotate_flag:
            params[0] = (0.0, 0.0)
        if not translate_flag:
            params[1] = (0.0, 0.0)
        if not scale_flag:
            params[2] = (1.0, 1.0)
        ret = self.get_params(*params)

        data["img"] = visionFunctional.affine(
            img,
            *ret,
            interpolation=self.interpolation,
            fill=self.fill,
        )
        if "gt_seg" in data:
            need_unsqueeze = len(data["gt_seg"].shape) == 2
            if need_unsqueeze:
                data["gt_seg"] = data["gt_seg"].unsqueeze(0)
            data["gt_seg"] = visionFunctional.affine(
                data["gt_seg"],
                *ret,
                interpolation=InterpolationMode.NEAREST,
                fill=self.label_fill_value,
            )
            if need_unsqueeze:
                data["gt_seg"] = data["gt_seg"].squeeze(0)
        return data


@OBJECT_REGISTRY.register
class Scale(object):
    """Scale multi-scale segmentation targets.

    Args:
        scales: Scale factor or factors.
        mode: Interpolation mode used by ``torch.nn.functional.interpolate``.
        mul_scale: Whether to multiply scaled tensors by the scale factor.
    """

    def __init__(
        self,
        scales: Union[Real, Sequence],
        mode: str = "nearest",
        mul_scale: bool = False,
    ):
        if isinstance(scales, Real):
            self.scales = [scales]
        elif isinstance(scales, Sequence):
            self.scales = scales
        else:
            raise TypeError(
                "scales should be number or sequence. Got {}".format(
                    type(scales)
                )
            )
        self.mode = mode
        self.mul_scale = mul_scale

    def _scale(self, data: Tensor):
        """Generate scaled copies of a segmentation target.

        Args:
            data: Segmentation target tensor.

        Returns:
            list[torch.Tensor]: Scaled target tensors.
        """
        scaled_data = []
        for scale in self.scales:
            scaled_tmp_data = torchFunctional.interpolate(
                data.to(dtype=torch.float),
                scale_factor=scale,
                mode=self.mode,
                recompute_scale_factor=True,
            ).to(dtype=data.dtype)
            scaled_data.append(
                scaled_tmp_data * scale if self.mul_scale else scaled_tmp_data
            )
        return scaled_data

    def __call__(self, data: dict):
        """Apply multi-scale target generation to ``gt_seg``.

        Args:
            data: Sample dict containing optional ``gt_seg``.

        Returns:
            dict: Sample dict with scaled segmentation targets.
        """
        if "gt_seg" in data:
            data["gt_seg"] = self._scale(data["gt_seg"])
        return data
