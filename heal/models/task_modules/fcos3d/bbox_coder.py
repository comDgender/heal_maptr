# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import Optional, Tuple

import numpy as np
import torch
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.apply_func import limit_period


@OBJECT_REGISTRY.register_module
class FCOS3DBBoxCoder:
    """Bounding box coder for FCOS3D.

    Args:
        base_depths: Depth references for decode box depth.
        base_dims: Dimension references for decode box dimension.
        code_size: The dimension of boxes to be encoded.
        norm_on_bbox: Whether to apply normalization on the bounding
            box 2D attributes.
    """

    def __init__(
        self,
        base_depths: Optional[Tuple[Tuple[float]]] = None,
        base_dims: Optional[Tuple[Tuple[float]]] = None,
        code_size: int = 7,
        norm_on_bbox: bool = True,
    ):
        self.base_depths = base_depths
        self.base_dims = base_dims
        self.bbox_code_size = code_size
        self.norm_on_bbox = norm_on_bbox

    def decode(
        self,
        bbox: torch.Tensor,
        scale: Tuple,
        stride: int,
        training: bool,
        cls_score: Optional[torch.Tensor] = None,
    ):
        """Decode scaled residuals into camera-frame 3D box attributes.

        ``bbox`` is expected as ``(N, C, H, W)`` with offset / depth / size in
        the first six channels. ``scale`` holds three callables applied to
        offset, depth and size respectively.
        """
        scale_offset, scale_depth, scale_size = scale[:3]
        raw = bbox.clone()

        # Apply learned per-group scales, then materialize float residuals.
        bbox[:, :2] = scale_offset(raw[:, :2]).float()
        bbox[:, 2] = scale_depth(raw[:, 2]).float()
        bbox[:, 3:6] = scale_size(raw[:, 3:6]).float()

        # Depth: either plain exp, a single prior, or a per-class prior.
        if self.base_depths is None:
            bbox[:, 2] = bbox[:, 2].exp()
        elif len(self.base_depths) == 1:
            depth_mean, depth_std = self.base_depths[0]
            bbox[:, 2] = depth_mean + bbox[:, 2].clone() * depth_std
        else:
            n_cls = cls_score.shape[1]
            assert len(self.base_depths) == n_cls, (
                "The number of multi-class depth priors should be equal to "
                "the number of categories."
            )
            cls_ids = cls_score.argmax(dim=1)
            # (N, H, W, 2) -> (N, 2, H, W) for (mean, std).
            prior = cls_score.new_tensor(self.base_depths)[cls_ids].permute(
                0, 3, 1, 2
            )
            bbox[:, 2] = prior[:, 0] + bbox[:, 2].clone() * prior[:, 1]

        # Size: exp residual, optionally re-scaled by class-wise priors.
        bbox[:, 3:6] = bbox[:, 3:6].exp()
        if self.base_dims is not None:
            n_cls = cls_score.shape[1]
            assert len(self.base_dims) == n_cls, (
                "The number of anchor sizes should be equal to the number "
                "of categories."
            )
            cls_ids = cls_score.argmax(dim=1)
            # (N, H, W, 3) -> (N, 3, H, W).
            size_prior = cls_score.new_tensor(self.base_dims)[cls_ids].permute(
                0, 3, 1, 2
            )
            bbox[:, 3:6] = size_prior * bbox[:, 3:6].clone()

        if not self.norm_on_bbox:
            raise AssertionError(
                "Setting norm_on_bbox to False "
                "has not been thoroughly tested for FCOS3D."
            )
        # Inference: undo stride normalization on the 2D offset channels.
        if not training:
            bbox[:, :2] = bbox[:, :2] * stride

        return bbox

    @staticmethod
    def decode_yaw(
        bbox: torch.Tensor,
        centers2d: torch.Tensor,
        dir_cls: torch.Tensor,
        dir_offset: float,
        cam2img: torch.Tensor,
    ):
        if bbox.shape[0] > 0:
            dir_rot = limit_period(bbox[..., 6] - dir_offset, 0, np.pi)
            bbox[..., 6] = (
                dir_rot + dir_offset + np.pi * dir_cls.to(bbox.dtype)
            )

        bbox[:, 6] = (
            torch.atan2(centers2d[:, 0] - cam2img[0, 2], cam2img[0, 0])
            + bbox[:, 6]
        )

        return bbox
