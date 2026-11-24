# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.distributed import dist_initialized, get_dist_info

from heal.core.box_utils import bbox_overlaps, box_center_to_corner
from heal.models.task_modules.detr import HungarianMatcher, generalized_box_iou
from heal.models.task_modules.motr.motr_utils import select_instances


def sigmoid_focal_loss(
    inputs,
    targets,
    num_boxes,
    alpha: float = 0.25,
    gamma: float = 2,
    mean_in_dim1: bool = True,
):
    """Binary focal loss on sigmoid logits for MOTR classification heads.

    Easy examples are down-weighted by ``(1 - p_gt)^gamma``; when
    ``alpha >= 0``, positives/negatives are further rebalanced. The reduced
    scalar is divided by ``num_boxes``.

    Args:
        inputs: Raw logits of any shape.
        targets: Same shape as ``inputs``; soft/hard labels in ``[0, 1]``.
        num_boxes: Normalizer (typically matched-box count).
        alpha: Pos/neg balance factor; skipped when ``alpha < 0``.
        gamma: Focusing exponent on the easy-example weight.
        mean_in_dim1: If True, average over dim 1 before summing.
    """
    pred_prob = torch.sigmoid(inputs)
    bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # Probability assigned to the target label (supports soft targets).
    prob_gt = pred_prob * targets + (1.0 - pred_prob) * (1.0 - targets)
    focal = bce * (1.0 - prob_gt).pow(gamma)

    if alpha >= 0:
        balance = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        focal = focal * balance

    reduced = focal.mean(dim=1).sum() if mean_in_dim1 else focal.sum()
    return reduced / num_boxes


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Return precision@k percentages for each ``k`` in ``topk``.

    Empty ``target`` yields a single empty tensor on ``output``'s device so
    callers can still ``.item()``-safe aggregate.
    """
    if target.numel() == 0:
        return [output.new_zeros(())]

    k_max = max(topk)
    n = target.shape[0]
    # Rank predictions: [n, k_max] -> [k_max, n] for slicing by k.
    top_ids = output.topk(k_max, dim=1, largest=True, sorted=True).indices
    top_ids = top_ids.transpose(0, 1)
    hits = top_ids.eq(target.reshape(1, -1).expand_as(top_ids))

    precisions = []
    scale = 100.0 / float(n)
    for k in topk:
        num_correct = hits[:k].reshape(-1).to(dtype=torch.float32).sum()
        precisions.append(num_correct * scale)
    return precisions


@OBJECT_REGISTRY.register
class MotrCriterion(nn.Module):
    """This class computes the loss for Motr.

    Args:
        num_classes: number of object categories.
        num_dec_layers: number of the decoder layers.
        cost_class: weight of the classification error in the matching cost.
        cost_bbox: weight of the L1 error of the bbox in the matching cost.
        cost_giou: weight of the giou loss of the bbox in the matching cost.
        cls_loss_coef: weight of the classification loss.
        bbox_loss_coef: weight of the L1 loss of the bbox.
        giou_loss_coef: weight of the giou loss of the bbox.
        aux_loss: True if auxiliary decoding losses are to be used.
        max_frames_per_seq: The max num frame of seq data.
    """

    def __init__(
        self,
        num_classes,
        num_dec_layers: int = 6,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        cls_loss_coef: float = 2,
        bbox_loss_coef: float = 5,
        giou_loss_coef: float = 2,
        aux_loss: bool = True,
        max_frames_per_seq: int = 5,
    ):
        super(MotrCriterion, self).__init__()
        self.num_classes = num_classes

        self.losses = ["labels", "boxes"]
        self.losses_dict = {}
        self._current_frame_idx = 0
        self.aux_loss = aux_loss
        self.matcher = HungarianMatcher(cost_class, cost_bbox, cost_giou, True)

        weight_dict = {}
        for i in range(max_frames_per_seq):
            weight_dict.update(
                {
                    "frame_{}_loss_ce".format(i): cls_loss_coef,
                    "frame_{}_loss_bbox".format(i): bbox_loss_coef,
                    "frame_{}_loss_giou".format(i): giou_loss_coef,
                }
            )

        if aux_loss:
            for i in range(max_frames_per_seq):
                for j in range(num_dec_layers - 1):
                    weight_dict.update(
                        {
                            "frame_{}_aux{}_loss_ce".format(
                                i, j
                            ): cls_loss_coef,
                            "frame_{}_aux{}_loss_bbox".format(
                                i, j
                            ): bbox_loss_coef,
                            "frame_{}_aux{}_loss_giou".format(
                                i, j
                            ): giou_loss_coef,
                        }
                    )

        self.weight_dict = weight_dict

    def initialize_for_single_clip(self):
        self.num_samples = 0
        self.sample_device = None
        self._current_frame_idx = 0
        self.losses_dict = {}

    def _step(self):
        self._current_frame_idx += 1

    def get_num_boxes(self, num_samples):
        """All-reduce sample count and return the per-rank average (min 1)."""
        count = torch.as_tensor(
            num_samples, dtype=torch.float, device=self.sample_device
        )
        if dist_initialized():
            torch.distributed.all_reduce(count)
        _, world_size = get_dist_info()
        return torch.clamp(count / world_size, min=1).item()

    def get_loss(
        self, loss, outputs, gt_instances, indices, num_boxes, **kwargs
    ):
        handlers = {
            "labels": self.loss_labels,
            "boxes": self.loss_boxes,
        }
        if loss not in handlers:
            raise AssertionError(f"unsupported loss type: {loss}")
        return handlers[loss](
            outputs, gt_instances, indices, num_boxes, **kwargs
        )

    def loss_boxes(
        self,
        outputs,
        gt_instances: List[Dict],
        indices: List[tuple],
        num_boxes,
    ):
        """L1 + GIoU on matched boxes (cxcywh, image-normalized).

        Pairs with GT index ``-1`` (disappeared tracks) are dropped before
        gathering; rows whose ``gt_ids == -1`` are also skipped in the
        regression terms.
        """
        # Drop disappeared-track pairs (tgt idx == -1).
        kept_pairs = []
        for pred_idx, gt_idx in indices:
            alive = gt_idx != -1
            kept_pairs.append((pred_idx[alive], gt_idx[alive]))
        indices = kept_pairs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat(
            [
                gt_per_img.gt_bboxes[i]
                for gt_per_img, (_, i) in zip(gt_instances, indices)
            ],
            dim=0,
        )

        # for pad target, don't calculate regression loss,
        # judged by whether obj_id=-1
        target_obj_ids = torch.cat(
            [
                gt_per_img.gt_ids[i]
                for gt_per_img, (_, i) in zip(gt_instances, indices)
            ],
            dim=0,
        )  # size(16)
        mask = target_obj_ids != -1

        loss_bbox = F.l1_loss(
            src_boxes[mask], target_boxes[mask], reduction="none"
        )
        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_center_to_corner(src_boxes[mask]),
                box_center_to_corner(target_boxes[mask]),
            )
        )

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def loss_labels(
        self, outputs, gt_instances: List[Dict], indices, num_boxes, log=False
    ):
        """Focal classification loss over all queries (bg = num_classes)."""
        logits = outputs["pred_logits"]
        matched = self._get_src_permutation_idx(indices)
        # Default every query to the background class id.
        class_ids = logits.new_full(
            logits.shape[:2], self.num_classes, dtype=torch.long
        )

        # Matched labels: disappear slots (gt row == -1) keep placeholder 1.
        matched_labels = []
        for gt_i, (_, gt_rows) in zip(gt_instances, indices):
            labels_i = torch.ones_like(gt_rows)
            if len(gt_i.gt_classes) > 0:
                alive = gt_rows != -1
                labels_i[alive] = gt_i.gt_classes[gt_rows[alive]].to(
                    torch.int64
                )
            matched_labels.append(labels_i)
        matched_labels = torch.cat(matched_labels)
        class_ids[matched] = matched_labels

        # One-hot without the background channel.
        one_hot = F.one_hot(class_ids, num_classes=self.num_classes + 1)
        one_hot = one_hot[..., :-1].to(dtype=logits.dtype)
        loss_ce = sigmoid_focal_loss(
            logits.flatten(1),
            one_hot.flatten(1),
            alpha=0.25,
            gamma=2,
            num_boxes=num_boxes,
            mean_in_dim1=False,
        )
        loss_ce = loss_ce.sum()
        losses = {"loss_ce": loss_ce}

        if log:
            losses["class_error"] = (
                100 - accuracy(logits[matched], matched_labels)[0]
            )

        return losses

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    def match_for_single_frame(
        self,
        combine_instance: dict,
        outputs_classes,
        outputs_coords,
        out_hs,
        gt_instances_i: dict,
    ):

        combine_instance.pred_logits = outputs_classes[-1][0]
        combine_instance.pred_boxes = outputs_coords[-1][0].float().sigmoid()
        combine_instance.pred_boxes_unsigmoid = outputs_coords[-1][0]
        combine_instance.output_embedding = out_hs[0]
        with torch.no_grad():
            combine_instance.scores = (
                outputs_classes[0][0, :].sigmoid().max(dim=-1).values
            )
        keep_idxs = combine_instance.mask_query == 1

        track_instances = select_instances(combine_instance, keep_idxs)

        pred_logits_i = track_instances.pred_logits
        pred_boxes_i = track_instances.pred_boxes
        device = pred_logits_i.device

        # Map GT object id -> row index for this frame.
        gt_id_list = gt_instances_i.gt_ids.detach().cpu().tolist()
        id_to_gt_row = {obj_id: row for row, obj_id in enumerate(gt_id_list)}
        outputs_i = {
            "pred_logits": pred_logits_i.unsqueeze(0),
            "pred_boxes": pred_boxes_i.unsqueeze(0),
        }

        # Inherit previous tracks: refresh matched GT rows or mark disappeared.
        num_disappear_track = 0
        n_tracks = len(track_instances.query_pos)
        for track_i in range(n_tracks):
            obj_id = track_instances.obj_idxes[track_i].item()
            if obj_id < 0:
                track_instances.matched_gt_idxes[track_i] = -1
                continue
            if obj_id in id_to_gt_row:
                track_instances.matched_gt_idxes[track_i] = id_to_gt_row[
                    obj_id
                ]
            else:
                num_disappear_track += 1
                track_instances.matched_gt_idxes[track_i] = -1

        all_track_rows = torch.arange(
            n_tracks, dtype=torch.long, device=device
        )
        occupied = track_instances.obj_idxes >= 0
        prev_matched_indices = torch.stack(
            (
                all_track_rows[occupied],
                track_instances.matched_gt_idxes[occupied],
            ),
            dim=1,
        )

        # Free detect slots (obj_idxes == -1); FP (-2) stays excluded.
        unmatched_track_idxes = all_track_rows[track_instances.obj_idxes == -1]

        # GTs not claimed by inherited tracks -> new-match candidates.
        claimed_rows = track_instances.matched_gt_idxes
        claimed_rows = claimed_rows[claimed_rows != -1]
        claimed_mask = torch.zeros(
            len(gt_instances_i.gt_bboxes), device=device
        )
        claimed_mask[claimed_rows] = 1
        untracked_tgt_indexes = torch.arange(
            len(gt_instances_i.gt_bboxes), device=device
        )[claimed_mask == 0]

        untracked_gt_instances = EasyDict(
            gt_bboxes=[gt_instances_i.gt_bboxes[untracked_tgt_indexes]],
            gt_classes=[gt_instances_i.gt_classes[untracked_tgt_indexes]],
            gt_ids=[gt_instances_i.gt_ids[untracked_tgt_indexes]],
        )

        def match_for_single_decoder_layer(unmatched_outputs, matcher):
            """Match free slots to untracked GTs; remap to global indices."""
            pair = matcher(unmatched_outputs, untracked_gt_instances)[0]
            slot_local, gt_local = pair[0], pair[1]
            return torch.stack(
                (
                    unmatched_track_idxes[slot_local],
                    untracked_tgt_indexes[gt_local],
                ),
                dim=1,
            )

        # Hungarian match free slots against untracked GTs.
        unmatched_outputs = {
            "pred_logits": track_instances.pred_logits[
                unmatched_track_idxes
            ].unsqueeze(0),
            "pred_boxes": track_instances.pred_boxes[
                unmatched_track_idxes
            ].unsqueeze(0),
        }
        new_matched_indices = match_for_single_decoder_layer(
            unmatched_outputs, self.matcher
        )

        # Write newly matched object ids / GT rows back onto track slots.
        new_slots = new_matched_indices[:, 0]
        new_gt_rows = new_matched_indices[:, 1]
        track_instances.obj_idxes[new_slots] = gt_instances_i.gt_ids[
            new_gt_rows
        ].long()
        track_instances.matched_gt_idxes[new_slots] = new_gt_rows

        # Aligned IoU for currently active (obj_id + gt row both valid) tracks.
        active = (track_instances.obj_idxes >= 0) & (
            track_instances.matched_gt_idxes >= 0
        )
        active_pred = track_instances.pred_boxes[active]
        if active_pred.numel() > 0:
            active_gt = gt_instances_i.gt_bboxes[
                track_instances.matched_gt_idxes[active]
            ]
            track_instances.iou[active] = bbox_overlaps(
                box_center_to_corner(active_pred),
                box_center_to_corner(active_gt),
                is_aligned=True,
            )

        matched_indices = torch.cat(
            [new_matched_indices, prev_matched_indices], dim=0
        )

        # Accumulate per-frame losses (normalize later in forward).
        self.num_samples += len(gt_instances_i.gt_bboxes) + num_disappear_track
        self.sample_device = device
        frame_key = self._current_frame_idx
        pair_indices = [(matched_indices[:, 0], matched_indices[:, 1])]
        for loss_name in self.losses:
            frame_loss = self.get_loss(
                loss_name,
                outputs=outputs_i,
                gt_instances=[gt_instances_i],
                indices=pair_indices,
                num_boxes=1,
            )
            self.losses_dict.update(
                {
                    "frame_{}_{}".format(frame_key, name): val
                    for name, val in frame_loss.items()
                }
            )

        if self.aux_loss:
            aux_outputs = self._set_aux_loss(outputs_classes, outputs_coords)
            for i, aux_output in enumerate(aux_outputs):
                aux_output["pred_boxes"] = (
                    aux_output["pred_boxes"].float().sigmoid()
                )
                unmatched_outputs_layer = {
                    "pred_logits": aux_output["pred_logits"][
                        0, unmatched_track_idxes
                    ].unsqueeze(0),
                    "pred_boxes": aux_output["pred_boxes"][
                        0, unmatched_track_idxes
                    ].unsqueeze(0),
                }
                new_matched_indices_layer = match_for_single_decoder_layer(
                    unmatched_outputs_layer, self.matcher
                )
                matched_indices_layer = torch.cat(
                    [new_matched_indices_layer, prev_matched_indices], dim=0
                )
                for loss in self.losses:
                    if loss == "masks":
                        continue
                    l_dict = self.get_loss(
                        loss,
                        aux_output,
                        gt_instances=[gt_instances_i],
                        indices=[
                            (
                                matched_indices_layer[:, 0],
                                matched_indices_layer[:, 1],
                            )
                        ],
                        num_boxes=1,
                    )
                    self.losses_dict.update(
                        {
                            "frame_{}_aux{}_{}".format(
                                self._current_frame_idx, i, key
                            ): value
                            for key, value in l_dict.items()
                        }
                    )
        self._step()
        return track_instances

    def forward(self):
        losses = {}
        num_samples = self.get_num_boxes(self.num_samples)
        for loss_name, loss in self.losses_dict.items():
            losses[loss_name] = loss / num_samples
        return losses
