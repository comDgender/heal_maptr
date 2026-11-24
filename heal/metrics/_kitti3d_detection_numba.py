# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

"""Numba JIT helpers for KITTI 3D detection metrics (import on demand only)."""

import warnings

import numba
import numpy as np

try:
    from numba.core.errors import (
        NumbaDeprecationWarning,
        NumbaPerformanceWarning,
        NumbaWarning,
    )

    warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
    warnings.filterwarnings("ignore", category=NumbaWarning)
except ImportError:
    pass


@numba.jit(nopython=True)
def _recall_at_rank(rank, total_gt, num_scores):
    recall = (rank + 1) / total_gt
    if rank + 1 < num_scores:
        return recall, (rank + 2) / total_gt
    return recall, recall


@numba.jit(nopython=True)
def _should_skip_threshold_sample(
    left_recall, right_recall, target_recall, has_next
):
    if not has_next:
        return False
    return (right_recall - target_recall) < (target_recall - left_recall)


@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    ordered = np.sort(scores)[::-1]
    recall_step = 1.0 / (num_sample_pts - 1.0)
    target_recall = 0.0
    picked = []
    for rank in range(ordered.shape[0]):
        left_recall, right_recall = _recall_at_rank(
            rank, num_gt, ordered.shape[0]
        )
        if _should_skip_threshold_sample(
            left_recall,
            right_recall,
            target_recall,
            rank + 1 < ordered.shape[0],
        ):
            continue
        picked.append(ordered[rank])
        target_recall += recall_step
    return picked


@numba.jit(nopython=True)
def _fused_image_slice(
    overlaps,
    gt_datas,
    dt_datas,
    ignored_gts,
    ignored_dets,
    dontcares,
    gt_off,
    dt_off,
    dc_off,
    gt_cnt,
    dt_cnt,
    dc_cnt,
):
    gt_end = gt_off + gt_cnt
    dt_end = dt_off + dt_cnt
    dc_end = dc_off + dc_cnt
    return (
        overlaps[dt_off:dt_end, gt_off:gt_end],
        gt_datas[gt_off:gt_end],
        dt_datas[dt_off:dt_end],
        ignored_gts[gt_off:gt_end],
        ignored_dets[dt_off:dt_end],
        dontcares[dc_off:dc_end],
    )


@numba.jit(nopython=True)
def fused_compute_statistics(
    overlaps,
    pr,
    gt_nums,
    dt_nums,
    dc_nums,
    gt_datas,
    dt_datas,
    dontcares,
    ignored_gts,
    ignored_dets,
    metric,
    min_overlap,
    thresholds,
    compute_aos=False,
):
    gt_off = 0
    dt_off = 0
    dc_off = 0
    for image_idx in range(gt_nums.shape[0]):
        gt_cnt = gt_nums[image_idx]
        dt_cnt = dt_nums[image_idx]
        dc_cnt = dc_nums[image_idx]
        (
            overlap_blk,
            gt_blk,
            dt_blk,
            ign_gt_blk,
            ign_dt_blk,
            dc_blk,
        ) = _fused_image_slice(
            overlaps,
            gt_datas,
            dt_datas,
            ignored_gts,
            ignored_dets,
            dontcares,
            gt_off,
            dt_off,
            dc_off,
            gt_cnt,
            dt_cnt,
            dc_cnt,
        )
        for thresh_idx, score_cut in enumerate(thresholds):
            tp, fp, fn, aos_sum, _ = compute_statistics_jit(
                overlap_blk,
                gt_blk,
                dt_blk,
                ign_gt_blk,
                ign_dt_blk,
                dc_blk,
                metric,
                min_overlap=min_overlap,
                thresh=score_cut,
                compute_fp=True,
                compute_aos=compute_aos,
            )
            pr[thresh_idx, 0] += tp
            pr[thresh_idx, 1] += fp
            pr[thresh_idx, 2] += fn
            if aos_sum != -1:
                pr[thresh_idx, 3] += aos_sum
        gt_off += gt_cnt
        dt_off += dt_cnt
        dc_off += dc_cnt


_NO_MATCH = -10000000


@numba.jit(nopython=True)
def _det_below_thresh_mask(dt_scores, thresh, det_size):
    below = [False] * det_size
    for det_id in range(det_size):
        below[det_id] = dt_scores[det_id] < thresh
    return below


@numba.jit(nopython=True)
def _det_unusable(ignored_det_val, already_used, below_thresh):
    return ignored_det_val == -1 or already_used or below_thresh


@numba.jit(nopython=True)
def _pick_det_for_gt(
    overlaps,
    dt_scores,
    ignored_det,
    assigned_detection,
    below_thresh,
    gt_id,
    min_overlap,
    compute_fp,
):
    best_det = -1
    best_signal = _NO_MATCH
    best_iou = 0.0
    use_ignored_fallback = False
    det_size = dt_scores.shape[0]

    for det_id in range(det_size):
        if _det_unusable(
            ignored_det[det_id],
            assigned_detection[det_id],
            below_thresh[det_id],
        ):
            continue
        iou = overlaps[det_id, gt_id]
        if iou <= min_overlap:
            continue
        if not compute_fp:
            if dt_scores[det_id] > best_signal:
                best_det = det_id
                best_signal = dt_scores[det_id]
            continue
        if ignored_det[det_id] == 0 and (
            iou > best_iou or use_ignored_fallback
        ):
            best_iou = iou
            best_det = det_id
            best_signal = 1
            use_ignored_fallback = False
        elif best_signal == _NO_MATCH and ignored_det[det_id] == 1:
            best_det = det_id
            best_signal = 1
            use_ignored_fallback = True

    return best_det, best_signal


@numba.jit(nopython=True)
def compute_statistics_jit(
    overlaps,
    gt_datas,
    dt_datas,
    ignored_gt,
    ignored_det,
    dc_bboxes,
    metric,
    min_overlap,
    thresh=0,
    compute_fp=False,
    compute_aos=False,
):
    det_size = dt_datas.shape[0]
    gt_size = gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]
    dt_alphas = dt_datas[:, 4]
    gt_alphas = gt_datas[:, 4]
    dt_bboxes = dt_datas[:, :4]

    assigned_detection = [False] * det_size
    below_thresh = (
        _det_below_thresh_mask(dt_scores, thresh, det_size)
        if compute_fp
        else [False] * det_size
    )

    tp = fp = fn = similarity = 0
    tp_score_buf = np.zeros((gt_size,))
    tp_score_len = 0
    alpha_delta = np.zeros((gt_size,))
    alpha_delta_len = 0

    for gt_id in range(gt_size):
        if ignored_gt[gt_id] == -1:
            continue
        det_id, match_signal = _pick_det_for_gt(
            overlaps,
            dt_scores,
            ignored_det,
            assigned_detection,
            below_thresh,
            gt_id,
            min_overlap,
            compute_fp,
        )
        if match_signal == _NO_MATCH and ignored_gt[gt_id] == 0:
            fn += 1
        elif match_signal != _NO_MATCH and (
            ignored_gt[gt_id] == 1 or ignored_det[det_id] == 1
        ):
            assigned_detection[det_id] = True
        elif match_signal != _NO_MATCH:
            tp += 1
            tp_score_buf[tp_score_len] = dt_scores[det_id]
            tp_score_len += 1
            if compute_aos:
                alpha_delta[alpha_delta_len] = (
                    gt_alphas[gt_id] - dt_alphas[det_id]
                )
                alpha_delta_len += 1
            assigned_detection[det_id] = True

    if compute_fp:
        for det_id in range(det_size):
            ign = ignored_det[det_id]
            if assigned_detection[det_id] or ign == -1 or ign == 1:
                continue
            if below_thresh[det_id]:
                continue
            fp += 1

        suppressed = 0
        if metric == 0 and dc_bboxes.shape[0] > 0:
            overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)
            for dc_id in range(dc_bboxes.shape[0]):
                for det_id in range(det_size):
                    ign = ignored_det[det_id]
                    if (
                        assigned_detection[det_id]
                        or ign == -1
                        or ign == 1
                        or below_thresh[det_id]
                    ):
                        continue
                    if overlaps_dt_dc[det_id, dc_id] > min_overlap:
                        assigned_detection[det_id] = True
                        suppressed += 1
        fp -= suppressed

        if compute_aos:
            aos_buf = np.zeros((fp + alpha_delta_len,))
            for k in range(alpha_delta_len):
                aos_buf[k + fp] = (1.0 + np.cos(alpha_delta[k])) / 2.0
            similarity = np.sum(aos_buf) if (tp > 0 or fp > 0) else -1

    return tp, fp, fn, similarity, tp_score_buf[:tp_score_len]


@numba.jit(nopython=True)
def _rect_intersect_1d(a0, a1, b0, b1):
    return min(a1, b1) - max(a0, b0)


@numba.jit(nopython=True)
def _box2d_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


_EPSILON = 1e-8


@numba.jit(nopython=True)
def _iou_from_areas(inter, area_a, area_b, criterion):
    if criterion == -1:
        union = area_a + area_b - inter
        if union <= _EPSILON:
            return 0.0
        return inter / union
    if criterion == 0:
        if area_a <= _EPSILON:
            return 0.0
        return inter / area_a
    if criterion == 1:
        if area_b <= _EPSILON:
            return 0.0
        return inter / area_b
    return inter


@numba.jit(nopython=True)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    num_boxes = boxes.shape[0]
    num_queries = query_boxes.shape[0]
    out = np.zeros((num_boxes, num_queries), dtype=boxes.dtype)
    for q in range(num_queries):
        qbox = query_boxes[q]
        q_area = _box2d_area(qbox)
        for b in range(num_boxes):
            box = boxes[b]
            w = _rect_intersect_1d(box[0], box[2], qbox[0], qbox[2])
            if w <= 0:
                continue
            h = _rect_intersect_1d(box[1], box[3], qbox[1], qbox[3])
            if h <= 0:
                continue
            inter = w * h
            out[b, q] = _iou_from_areas(
                inter, _box2d_area(box), q_area, criterion
            )
    return out


@numba.jit(nopython=True)
def _z_extent(box, z_axis, z_center):
    z = box[z_axis]
    depth = box[z_axis + 3]
    return z - depth * z_center, z + depth * (1.0 - z_center)


@numba.jit(nopython=True, parallel=True)
def box3d_overlap_kernel(
    boxes, qboxes, rinc, criterion=-1, z_axis=1, z_center=1.0
):
    """Fuse BEV overlap with vertical overlap for 3D IoU."""
    num_boxes = boxes.shape[0]
    num_queries = qboxes.shape[0]
    for bi in range(num_boxes):
        box = boxes[bi]
        vol_a = box[3] * box[4] * box[5]
        z_lo_a, z_hi_a = _z_extent(box, z_axis, z_center)
        for qj in range(num_queries):
            bev_iou = rinc[bi, qj]
            if bev_iou <= 0:
                continue
            qbox = qboxes[qj]
            z_lo_b, z_hi_b = _z_extent(qbox, z_axis, z_center)
            z_overlap = min(z_hi_a, z_hi_b) - max(z_lo_a, z_lo_b)
            if z_overlap <= 0:
                rinc[bi, qj] = 0.0
                continue
            vol_b = qbox[3] * qbox[4] * qbox[5]
            inter_vol = z_overlap * bev_iou
            rinc[bi, qj] = _iou_from_areas(inter_vol, vol_a, vol_b, criterion)
