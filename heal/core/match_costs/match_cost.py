# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

# Copyright (c) OpenMMLab. All rights reserved.
#
# Derived from open-mmlab/mmdetection
# (mmdet/core/bbox/match_costs/match_cost.py), licensed under the Apache
# License, Version 2.0. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# This file has been modified by Beijing Horizon Information Technology Co., Ltd. on 2026/08/03.  # noqa: E501
# Adapted mmdet's BBoxL1Cost / FocalLossCost for HEAL: re-registered the
# match costs from mmdet's MATCH_COST to HEAL's OBJECT_REGISTRY, renamed
# BBoxL1Cost to BBox3DL1Cost for 3D box matching, dropped the upstream
# ClassificationCost / IoUCost / DiceCost / CrossEntropyLossCost helpers
# (not used by HEAL), and added a mask-focal cost path
# (_mask_focal_loss_cost) for segmentation.

import torch
from hat_engine.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class BBox3DL1Cost(object):
    """BBox3DL1Cost.

    Args:
        weight: loss_weight
    """

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def __call__(
        self, bbox_pred: torch.Tensor, gt_bboxes: torch.Tensor
    ) -> torch.Tensor:
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight


@OBJECT_REGISTRY.register
class FocalLossCost:
    """FocalLossCost.

    Args:
        weight: loss_weight
        alpha: focal_loss alpha
        gamma: focal_loss gamma
        eps: default 1e-12
        binary_input: Whether the input is binary,
           default False.
    """

    def __init__(
        self,
        weight: float = 1.0,
        alpha: float = 0.25,
        gamma: int = 2,
        eps: float = 1e-12,
        binary_input: bool = False,
    ):
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.binary_input = binary_input

    def _focal_loss_cost(
        self, cls_pred: torch.Tensor, gt_labels: torch.Tensor
    ) -> torch.Tensor:
        cls_pred = cls_pred.sigmoid()
        neg_cost = (
            -(1 - cls_pred + self.eps).log()
            * (1 - self.alpha)
            * cls_pred.pow(self.gamma)
        )
        pos_cost = (
            -(cls_pred + self.eps).log()
            * self.alpha
            * (1 - cls_pred).pow(self.gamma)
        )
        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * self.weight

    def _mask_focal_loss_cost(
        self, cls_pred: torch.Tensor, gt_labels: torch.Tensor
    ) -> torch.Tensor:
        cls_pred = cls_pred.flatten(1)
        gt_labels = gt_labels.flatten(1).float()
        n = cls_pred.shape[1]
        cls_pred = cls_pred.sigmoid()
        neg_cost = (
            -(1 - cls_pred + self.eps).log()
            * (1 - self.alpha)
            * cls_pred.pow(self.gamma)
        )
        pos_cost = (
            -(cls_pred + self.eps).log()
            * self.alpha
            * (1 - cls_pred).pow(self.gamma)
        )
        cls_cost = torch.einsum(
            "nc,mc->nm", pos_cost, gt_labels
        ) + torch.einsum("nc,mc->nm", neg_cost, (1 - gt_labels))
        return cls_cost / n * self.weight

    def __call__(
        self, cls_pred: torch.Tensor, gt_labels: torch.Tensor
    ) -> torch.Tensor:
        if self.binary_input:
            return self._mask_focal_loss_cost(cls_pred, gt_labels)
        else:
            return self._focal_loss_cost(cls_pred, gt_labels)
