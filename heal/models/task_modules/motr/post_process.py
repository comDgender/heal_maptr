# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import torch
import torch.nn as nn
from hat_engine.registry import OBJECT_REGISTRY
from torch.amp import autocast

from heal.core.box_utils import bbox_overlaps, box_center_to_corner
from heal.models.task_modules.motr.motr_utils import (
    combine_instances,
    padding_tracks,
    random_drop_tracks,
    select_instances,
)

__all__ = ["MotrPostProcess"]


class RuntimeTrackerBase(object):
    def __init__(
        self, score_thresh=0.7, filter_score_thresh=0.6, miss_tolerance=5
    ):
        self.score_thresh = score_thresh
        self.filter_score_thresh = filter_score_thresh
        self.miss_tolerance = miss_tolerance
        self.max_obj_id = 0

    def clear(self):
        self.max_obj_id = 0

    def update(self, track_instances: dict):
        # High-score tracks are considered present this frame.
        alive = track_instances.scores >= self.score_thresh
        track_instances.disappear_time[alive] = 0

        n_tracks = len(track_instances.scores)
        for ti in range(n_tracks):
            obj_id = track_instances.obj_idxes[ti]
            score = track_instances.scores[ti]
            # New detection above score_thresh -> allocate an id.
            if obj_id == -1 and score >= self.score_thresh:
                track_instances.obj_idxes[ti] = self.max_obj_id
                self.max_obj_id += 1
                continue
            # Existing track below filter thresh -> accumulate miss count.
            if obj_id >= 0 and score < self.filter_score_thresh:
                track_instances.disappear_time[ti] += 1
                if track_instances.disappear_time[ti] >= self.miss_tolerance:
                    # Invalidate id; TrackEmbeddingLayer will drop it.
                    track_instances.obj_idxes[ti] = -1


@OBJECT_REGISTRY.register
class MotrPostProcess(nn.Module):
    def __init__(
        self,
        max_track: int = 256,
        area_threshold: int = 100,
        prob_threshold: float = 0.7,
        random_drop: float = 0.1,
        fp_ratio: float = 0.3,
        score_thresh: float = 0.7,
        filter_score_thresh: float = 0.6,
        miss_tolerance: int = 5,
    ):
        super(MotrPostProcess, self).__init__()
        self.track_base = RuntimeTrackerBase(
            score_thresh=score_thresh,
            filter_score_thresh=filter_score_thresh,
            miss_tolerance=miss_tolerance,
        )
        self.area_threshold = area_threshold
        self.prob_threshold = prob_threshold
        self.max_track = max_track
        self.random_drop = random_drop
        self.fp_ratio = fp_ratio

    def _random_drop_tracks(self, track_instances):
        return random_drop_tracks(track_instances, self.random_drop)

    def _add_fp_tracks(self, track_instances, active_track_instances):

        idx_tmp = track_instances.obj_idxes < 0
        inactive_instances = select_instances(track_instances, idx_tmp)

        # add fp for each active track in a specific probability.
        fp_prob = (
            torch.ones_like(active_track_instances.scores) * self.fp_ratio
        )

        idx_tmp_1 = torch.bernoulli(fp_prob).bool()
        selected_active_track_instances = select_instances(
            active_track_instances, idx_tmp_1
        )

        if (
            len(inactive_instances.query_pos) > 0
            and len(selected_active_track_instances.query_pos) > 0
        ):
            n_fp_needed = len(selected_active_track_instances.query_pos)
            n_inactive = len(inactive_instances.query_pos)
            if n_fp_needed >= n_inactive:
                # Not enough inactive slots: take all of them as FP.
                fp_track_instances = inactive_instances
            else:
                # For each selected active box, pick the inactive with max IoU.
                inactive_xyxy = box_center_to_corner(
                    inactive_instances.pred_boxes
                )
                active_xyxy = box_center_to_corner(
                    selected_active_track_instances.pred_boxes
                )
                iou_mat = bbox_overlaps(
                    inactive_xyxy, active_xyxy, is_aligned=False
                )
                best_inactive = torch.argmax(iou_mat, dim=0)
                best_inactive = torch.unique(best_inactive)
                fp_track_instances = select_instances(
                    inactive_instances, best_inactive
                )

            return combine_instances(
                active_track_instances, fp_track_instances
            )

        return active_track_instances

    def _select_active_tracks(self, track_instances: dict):
        if self.training:
            keep = (track_instances.obj_idxes >= 0) & (
                track_instances.iou > 0.5
            )
            active_track_instances = select_instances(track_instances, keep)
            active_track_instances = self._random_drop_tracks(
                active_track_instances
            )
            if self.fp_ratio > 0:
                active_track_instances = self._add_fp_tracks(
                    track_instances, active_track_instances
                )
        else:
            keep = track_instances.obj_idxes >= 0
            active_track_instances = select_instances(track_instances, keep)
        return active_track_instances

    def _head_out_reshape_permute(
        self, out_hs, outputs_classes_head, outputs_coords_head
    ):
        """Process head out from 4dim to 3dim."""
        out_hs = (
            out_hs.contiguous()
            .view(
                out_hs.shape[0],
                out_hs.shape[1],
                out_hs.shape[2] * out_hs.shape[3],
            )
            .permute(0, 2, 1)
        ).to(torch.float32)
        outputs_classes = []
        outputs_coords = []
        for i in range(len(outputs_classes_head)):
            outputs_classes.append(
                outputs_classes_head[i]
                .view(
                    outputs_classes_head[i].shape[0],
                    outputs_classes_head[i].shape[1],
                    outputs_classes_head[i].shape[2]
                    * outputs_classes_head[i].shape[3],
                )
                .permute(0, 2, 1)
                .to(torch.float32)
            )
            outputs_coords.append(
                outputs_coords_head[i]
                .view(
                    outputs_coords_head[i].shape[0],
                    outputs_coords_head[i].shape[1],
                    outputs_coords_head[i].shape[2]
                    * outputs_coords_head[i].shape[3],
                )
                .permute(0, 2, 1)
                .to(torch.float32)
            )
        return out_hs, outputs_classes, outputs_coords

    def _prepare_infer_process(
        self, track_instances, hs, outputs_classes, outputs_coords
    ):
        """Prepare for test process."""
        outputs_coord_unsigmoid = outputs_coords[-1]
        outputs_class = outputs_classes[-1]
        outputs_coord_sigmoid = outputs_coord_unsigmoid.sigmoid()
        track_scores = outputs_class[0].sigmoid()[:, 0]

        track_instances.scores = track_scores
        track_instances.pred_logits = outputs_class[0]
        track_instances.pred_boxes = outputs_coord_sigmoid[0]
        track_instances.output_embedding = hs[0]
        track_instances.pred_boxes_unsigmoid = outputs_coord_unsigmoid[0]
        return track_instances

    @autocast("cuda", enabled=False)
    def forward(
        self,
        track_instances,
        empty_track_instance,
        fake_track_instance,
        out_hs,
        outputs_classes_head,
        outputs_coords_head,
        criterion=None,
        targets=None,
        seq_data=None,
        frame_id=None,
        seq_frame_id=None,
        seq_name=None,
    ):
        (
            out_hs,
            outputs_classes,
            outputs_coords,
        ) = self._head_out_reshape_permute(
            out_hs, outputs_classes_head, outputs_coords_head
        )

        combine_instance = combine_instances(
            empty_track_instance, track_instances
        )

        if self.training:
            assert criterion is not None
            assert targets is not None
            output_instance = criterion.match_for_single_frame(
                combine_instance,
                outputs_classes,
                outputs_coords,
                out_hs,
                targets[frame_id],
            )
        else:
            output_instance = self._prepare_infer_process(
                combine_instance,
                out_hs,
                outputs_classes,
                outputs_coords,
            )

        keep_idxs = output_instance.mask_query == 1
        output_instance = select_instances(output_instance, keep_idxs)

        if not self.training:
            self.track_base.update(output_instance)

        active_track_instances = self._select_active_tracks(output_instance)

        if len(active_track_instances.scores) > self.max_track:
            keep_idxs = active_track_instances.scores.topk(self.max_track)[1]
            active_track_instances = select_instances(
                active_track_instances, keep_idxs
            )

        padding_track_instance, padd_len = padding_tracks(
            active_track_instances, fake_track_instance
        )

        if self.training:
            frame_outs = None
        else:
            frame_outs = self.get_ori_result(
                padding_track_instance,
                seq_data,
                frame_id,
                seq_frame_id,
                seq_name,
            )

        return padding_track_instance, padd_len, frame_outs

    def filter_dt_by_score(self, dt_instances: dict):
        keep = dt_instances.scores > self.prob_threshold
        return select_instances(dt_instances, keep)

    def filter_dt_by_area(self, dt_instances: dict):
        wh = dt_instances.pred_boxes[:, 2:4] - dt_instances.pred_boxes[:, 0:2]
        areas = wh[:, 0] * wh[:, 1]
        keep = areas > self.area_threshold
        return select_instances(dt_instances, keep)

    def get_ori_result(
        self,
        tracked_instance,
        seq_data,
        frame_id,
        seq_frame_id,
        seq_name,
    ):
        keep_idxs = tracked_instance.mask_query == 1
        tracked_instance = select_instances(tracked_instance, keep_idxs)

        scale_factor = seq_data["scale_factor"][frame_id]
        h, w = seq_data["img"][frame_id].shape[2:]
        image_size_xyxy = torch.as_tensor(
            [w, h, w, h],
            dtype=torch.float,
            device=seq_data["img"][frame_id].device,
        )
        for i in range(len(tracked_instance.pred_boxes)):
            tracked_instance.pred_boxes[i] = box_center_to_corner(
                tracked_instance.pred_boxes[i]
            )
            tracked_instance.pred_boxes[i] = (
                tracked_instance.pred_boxes[i] * image_size_xyxy
            )

            tracked_instance.pred_boxes[i] = (
                tracked_instance.pred_boxes[i] / scale_factor
            )

        dt_instances = self.filter_dt_by_score(tracked_instance)
        dt_instances = self.filter_dt_by_area(dt_instances)

        frame_outs = {
            "scores": dt_instances.scores,
            "pred_boxes": dt_instances.pred_boxes,
            "obj_idxes": dt_instances.obj_idxes,
            "seq_frame_id": seq_frame_id,
            "seq_name": seq_name,
        }
        return frame_outs
