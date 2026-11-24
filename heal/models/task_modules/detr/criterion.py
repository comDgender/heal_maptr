# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
# Source code reference to Facebook.

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.distributed import dist_initialized, get_dist_info

from heal.core.box_utils import box_center_to_corner, box_corner_to_center
from .matcher import HungarianMatcher, generalized_box_iou

__all__ = [
    "DetrCriterion",
    "accuracy",
]


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Compute the precision@k for the specified values of k."""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


@OBJECT_REGISTRY.register
class DetrCriterion(nn.Module):
    """This class computes the loss for DETR.

    Args:
        num_classes: number of object categories.
        dec_layers: number of the decoder layers.
        cost_class: weight of the classification error in the matching cost.
        cost_bbox: weight of the L1 error of the bbox in the matching cost.
        cost_giou: weight of the giou loss of the bbox in the matching cost.
        loss_class: weight of the classification loss.
        loss_bbox: weight of the L1 loss of the bbox.
        loss_giou: weight of the giou loss of the bbox.
        eos_coef: classification weight applied to the no-object category.
        losses: list of all the losses to be applied.
        aux_loss: True if auxiliary decoding losses are to be used.
    """

    def __init__(
        self,
        num_classes: int,
        dec_layers: int = 6,
        cost_class: float = 1.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        loss_ce: float = 1.0,
        loss_bbox: float = 5.0,
        loss_giou: float = 2.0,
        eos_coef: float = 0.1,
        losses: Sequence[str] = ("labels", "boxes", "cardinality"),
        aux_loss: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher(cost_class, cost_bbox, cost_giou)
        self.weight_dict = {
            "loss_ce": loss_ce,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
        }
        if aux_loss:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update(
                    {k + f"_{i}": v for k, v in self.weight_dict.items()}
                )
            self.weight_dict.update(aux_weight_dict)
        self.eos_coef = eos_coef
        self.losses = losses
        self.aux_loss = aux_loss
        # Down-weight the no-object / EOS class in CE.
        class_weight = torch.ones(self.num_classes + 1)
        class_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", class_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)."""
        assert "pred_logits" in outputs
        logits = outputs["pred_logits"]

        src_idx = self._get_src_permutation_idx(indices)
        matched_labels = torch.cat(
            [
                cls_ids[tgt_i]
                for cls_ids, (_, tgt_i) in zip(targets["gt_classes"], indices)
            ]
        )
        # Fill with EOS class, then overwrite matched queries.
        cls_target = torch.full(
            logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=logits.device,
        )
        cls_target[src_idx] = matched_labels.long()

        ce = F.cross_entropy(
            logits.transpose(1, 2), cls_target, self.empty_weight
        )
        out = {"loss_ce": ce}

        if log:
            out["class_error"] = (
                100 - accuracy(logits[src_idx], matched_labels)[0]
            )
        return out

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Absolute error of predicted vs GT object counts (logging only)."""
        logits = outputs["pred_logits"]
        eos_id = logits.shape[-1] - 1
        gt_counts = torch.as_tensor(
            [len(cls_ids) for cls_ids in targets["gt_classes"]],
            device=logits.device,
        )
        # Predictions whose argmax is not the EOS / no-object slot.
        pred_counts = (logits.argmax(dim=-1) != eos_id).sum(dim=1)
        err = F.l1_loss(pred_counts.float(), gt_counts.float())
        return {"cardinality_error": err}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """L1 + GIoU box losses on matched pairs (cxcywh, normalized)."""
        assert "pred_boxes" in outputs
        src_idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][src_idx]
        target_boxes = torch.cat(
            [
                boxes[tgt_i]
                for boxes, (_, tgt_i) in zip(targets["gt_bboxes"], indices)
            ],
            dim=0,
        )

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes  # normlize?

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_center_to_corner(src_boxes),
                box_center_to_corner(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # Flatten (batch_id, query_id) pairs from Hungarian match results.
        b_ids = torch.cat(
            [
                torch.full_like(src_i, batch_i)
                for batch_i, (src_i, _) in enumerate(indices)
            ]
        )
        q_ids = torch.cat([src_i for (src_i, _) in indices])
        return b_ids, q_ids

    def _get_tgt_permutation_idx(self, indices):
        # Flatten (batch_id, gt_id) pairs from Hungarian match results.
        b_ids = torch.cat(
            [
                torch.full_like(tgt_i, batch_i)
                for batch_i, (_, tgt_i) in enumerate(indices)
            ]
        )
        g_ids = torch.cat([tgt_i for (_, tgt_i) in indices])
        return b_ids, g_ids

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        handlers = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
        }
        assert loss in handlers, f"unsupported loss type: {loss}"
        return handlers[loss](outputs, targets, indices, num_boxes, **kwargs)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    def prepare_targets(self, targets):
        new_boxes = []
        boxes = targets["gt_bboxes"]
        if "before_pad_shape" in targets:
            shapes = targets["before_pad_shape"]
        else:
            shapes = targets["img_shape"]
        for shape, boxes_per_image in zip(shapes, boxes):
            h, w = shape[-2:]
            image_size_xyxy = torch.as_tensor(
                [w, h, w, h], dtype=torch.float, device=boxes_per_image.device
            )
            gt_boxes = boxes_per_image / image_size_xyxy
            gt_boxes = box_corner_to_center(gt_boxes)
            new_boxes.append(gt_boxes)
        targets["gt_bboxes"] = new_boxes
        return targets

    def forward(self, outs, targets):
        targets = self.prepare_targets(targets)

        outputs_class, outputs_coord = outs
        outputs_coord = outputs_coord.sigmoid()
        outputs = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord[-1],
        }
        if self.aux_loss:
            outputs["aux_outputs"] = self._set_aux_loss(
                outputs_class, outputs_coord
            )

        main_outputs = {
            key: val for key, val in outputs.items() if key != "aux_outputs"
        }

        # Match final-layer predictions to GT, then normalize by mean GT count.
        indices = self.matcher(main_outputs, targets)

        num_boxes = sum(len(cls_ids) for cls_ids in targets["gt_classes"])
        num_boxes = torch.as_tensor(
            [num_boxes],
            dtype=torch.float,
            device=next(iter(outputs.values())).device,
        )
        if dist_initialized():
            torch.distributed.all_reduce(num_boxes)
            _, world_size = get_dist_info()
            num_boxes = torch.clamp(num_boxes / world_size, min=1).item()

        losses = {}
        for name in self.losses:
            losses.update(
                self.get_loss(name, outputs, targets, indices, num_boxes)
            )

        # Aux decoder layers: rematch and accumulate with `_i` suffixes.
        if "aux_outputs" in outputs:
            for layer_i, layer_out in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(layer_out, targets)
                for name in self.losses:
                    extra = {"log": False} if name == "labels" else {}
                    layer_losses = self.get_loss(
                        name,
                        layer_out,
                        targets,
                        indices,
                        num_boxes,
                        **extra,
                    )
                    losses.update(
                        {f"{k}_{layer_i}": v for k, v in layer_losses.items()}
                    )

        for k in losses.keys():
            if k in self.weight_dict:
                losses[k] *= self.weight_dict[k]

        return losses
