# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import torch
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.package_helper import require_packages
from scipy.optimize import linear_sum_assignment
from torch import nn

from heal.core.box_utils import box_center_to_corner

try:
    from torchvision.ops.boxes import box_area
except ImportError:
    box_area = None

__all__ = ["HungarianMatcher", "generalized_box_iou"]


@require_packages("torchvision")
def box_iou(boxes1, boxes2, eps):
    """Pairwise IoU for axis-aligned boxes in xyxy format."""
    a1 = box_area(boxes1)
    a2 = box_area(boxes2)
    # Intersection rectangle corners.
    corner_min = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    corner_max = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inter_wh = (corner_max - corner_min).clamp(min=0)
    inter = inter_wh[..., 0] * inter_wh[..., 1]
    union = a1[:, None] + a2 - inter + eps
    return inter / union, union


def generalized_box_iou(boxes1, boxes2, eps=1e-6):
    """Pairwise GIoU for xyxy boxes; see https://giou.stanford.edu/."""
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2, eps)

    enclose_min = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    enclose_max = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    enclose_wh = (enclose_max - enclose_min).clamp(min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1] + eps
    return iou - (enclose_area - union) / enclose_area


@OBJECT_REGISTRY.register
class HungarianMatcher(nn.Module):
    """Compute an assignment between targets and predictions.

    For efficiency reasons, the targets don't include the no_object.
    Because of this, in general, there are more predictions than targets.
    In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).

    Args:
        cost_class: weight of the classification error.
        cost_bbox: weight of the L1 error of the bbox coordinates.
        cost_giou: weight of the giou loss of the bounding box.
        use_focal: whether to use focal loss.
        alpha: A weighting factor for pos-sample, (1-alpha) is for
            neg-sample.
        gamma: Gamma used in focal loss to compress the contribution
            of easy examples.
    Returns:
        A list, containing tuples of (index_i, index_j) where:
            index_i is the indices of the selected predictions (in order)
            index_j is the indices of the selected targets (in order)
            For each batch element, it holds:
            len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
    """

    def __init__(
        self,
        cost_class: float = 1,
        cost_bbox: float = 1,
        cost_giou: float = 1,
        use_focal: bool = False,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.use_focal = use_focal
        self.alpha = alpha
        self.gamma = gamma
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, data):
        """Perform the matching.

        Args:
            outputs: a dict containing at least these entries:
                "pred_logits": Tensor of dim [bs, num_queries, num_classes]
                "pred_boxes": Tensor of dim [bs, num_queries, 4]

            data: a dict containing at least these entries:
                "gt_classes": Tensor of dim [num_target_boxes]
                "boxes": Tensor of dim [num_target_boxes, 4]
        """
        batch_size, n_query = outputs["pred_logits"].shape[:2]

        # Flatten batch so costs are (B*Q, ...) vs concatenated GTs.
        logits = outputs["pred_logits"].flatten(0, 1)
        probs = logits.sigmoid() if self.use_focal else logits.softmax(dim=-1)
        pred_boxes = outputs["pred_boxes"].flatten(0, 1)
        gt_labels = torch.cat(data["gt_classes"]).long()
        gt_boxes = torch.cat(data["gt_bboxes"]).float()

        # Class cost: focal form or -p_target (NLL without the constant 1).
        if self.use_focal:
            p = probs
            neg_term = (
                (1.0 - self.alpha)
                * (p**self.gamma)
                * (-(1.0 - p + 1e-8).log())
            )
            pos_term = (
                self.alpha * ((1.0 - p) ** self.gamma) * (-(p + 1e-8).log())
            )
            cls_cost = pos_term[:, gt_labels] - neg_term[:, gt_labels]
        else:
            cls_cost = -probs[:, gt_labels]

        l1_cost = torch.cdist(pred_boxes, gt_boxes, p=1)
        giou_cost = -generalized_box_iou(
            box_center_to_corner(pred_boxes).float(),
            box_center_to_corner(gt_boxes),
        )

        cost = (
            self.cost_bbox * l1_cost
            + self.cost_class * cls_cost
            + self.cost_giou * giou_cost
        )
        cost = cost.view(batch_size, n_query, -1).cpu()

        gt_sizes = [len(b) for b in data["gt_bboxes"]]
        indices = [
            linear_sum_assignment(cost_b[bi])
            for bi, cost_b in enumerate(cost.split(gt_sizes, dim=-1))
        ]
        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]
