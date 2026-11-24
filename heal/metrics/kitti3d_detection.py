# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import logging
from typing import List, Optional

import numpy as np
from hat_engine.metrics.metric import EvalMetric
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.apply_func import convert_numpy, limit_period

from heal.core.box_np_ops import box3d_to_bbox, box_lidar_to_camera

__all__ = ["Kitti3DMetricDet"]


logger = logging.getLogger(__name__)

_KITTI_CLASS_ID_TO_NAME = {
    0: "Car",
    1: "Pedestrian",
    2: "Cyclist",
    3: "Van",
    4: "Person_sitting",
}


def _build_kitti_overlap_thresholds() -> np.ndarray:
    profile_07 = np.tile(
        np.array([0.7, 0.5, 0.5, 0.7, 0.5, 0.7, 0.7, 0.7], dtype=np.float64),
        (3, 1),
    )
    profile_05 = np.array(
        [
            [0.7, 0.5, 0.5, 0.7, 0.5, 0.5, 0.5, 0.5],
            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5, 0.5, 0.5],
            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5, 0.5, 0.5],
        ],
        dtype=np.float64,
    )
    return np.stack((profile_07, profile_05), axis=0)


def _parse_eval_class_ids(raw_classes) -> List[int]:
    label_to_id = {name: idx for idx, name in _KITTI_CLASS_ID_TO_NAME.items()}
    entries = (
        raw_classes
        if isinstance(raw_classes, (list, tuple))
        else [raw_classes]
    )
    resolved = []
    for entry in entries:
        resolved.append(
            label_to_id[entry] if isinstance(entry, str) else entry
        )
    return resolved


@OBJECT_REGISTRY.register
class Kitti3DMetricDet(EvalMetric):
    def __init__(
        self,
        current_classes: List[str],
        compute_aos: bool = False,
        name: str = "kitti3dAPDet",
        difficultys: Optional[List] = None,  # noqa B006
    ):
        super().__init__(name)

        if difficultys is None:
            difficultys = [0, 1, 2]

        min_overlaps = _build_kitti_overlap_thresholds()
        class_to_name = _KITTI_CLASS_ID_TO_NAME
        current_classes = _parse_eval_class_ids(current_classes)
        min_overlaps = min_overlaps[:, :, current_classes]

        # TODO: move alpha-checking outside
        # check whether alpha is valid
        # compute_aos = False
        # for anno in dt_annos:
        #     if anno['alpha'].shape[0] != 0:
        #         if anno['alpha'][0] != -10:
        #             compute_aos = True
        #         break

        self.class_to_name = class_to_name
        self.min_overlaps = min_overlaps
        self.current_classes = current_classes
        self.compute_aos = compute_aos
        self.difficultys = difficultys
        self.mAPbbox = None
        self.mAPbev = None
        self.mAP3d = None
        self.mAPaos = None

        self.labels = []
        self.preds = []

    def reset(self):
        self.labels = []
        self.preds = []

    def update(self, preds, labels):

        preds_annos = self.convert_detection_to_kitti_annos(
            batch_detections=preds,
            batch_labels=labels,
        )
        gt_labels = self.convert_gt_annos(labels)

        self.preds += preds_annos
        self.labels += gt_labels

    def _update(self):
        assert len(self.preds) == len(
            self.labels
        ), f"{len(self.preds)} vs {len(self.labels)}"
        num_parts = min(len(self.preds), 50)

        mAPbbox, mAPbev, mAP3d, mAPaos = do_eval_v3(
            self.labels,
            self.preds,
            self.current_classes,
            self.min_overlaps,
            self.compute_aos,
            self.difficultys,
            num_parts=num_parts,
        )
        self.mAPbbox = mAPbbox
        self.mAPbev = mAPbev
        self.mAP3d = mAP3d
        self.mAPaos = mAPaos

    def get(self):
        logger.info("Calculate Kitti metric, please wait ...")
        self._update()
        names, values = ([], [])
        d3_names, d3_values = ([], [])
        for j, curcls in enumerate(self.current_classes):
            # mAP threshold array: [num_minoverlap, metric, class]
            # mAP result: [num_class, num_diff, num_minoverlap]
            for i in range(self.min_overlaps.shape[0]):
                names.append(f"{self.class_to_name[curcls]} AP@")
                overlap_str = ", ".join(
                    [str(x) for x in self.min_overlaps[i, :, j].tolist()]
                )
                values.append(f"{overlap_str}:")

                names.append("bbox AP:")
                bbox_ap_value = "%.2f, %.2f, %.2f" % (
                    self.mAPbbox[j, 0, i],
                    self.mAPbbox[j, 1, i],
                    self.mAPbbox[j, 2, i],
                )
                values.append(bbox_ap_value)
                names.append("bev  AP:")
                bev_ap_value = "%.2f, %.2f, %.2f" % (
                    self.mAPbev[j, 0, i],
                    self.mAPbev[j, 1, i],
                    self.mAPbev[j, 2, i],
                )
                values.append(bev_ap_value)
                names.append("3d   AP:")
                d3_ap_value = "%.2f, %.2f, %.2f" % (
                    self.mAP3d[j, 0, i],
                    self.mAP3d[j, 1, i],
                    self.mAP3d[j, 2, i],
                )
                values.append(d3_ap_value)
                if self.compute_aos:
                    names.append("aos  AP:")
                    bev_ap_value = "%.2f, %.2f, %.2f" % (
                        self.mAPaos[j, 0, i],
                        self.mAPaos[j, 1, i],
                        self.mAPaos[j, 2, i],
                    )
                    values.append(bev_ap_value)

                if i == 0:
                    d3_iou = str(self.min_overlaps[i, :, j].tolist()[1])
                    d3_names.append(
                        f"{self.class_to_name[curcls]}_3D_AP@{d3_iou}_moderate"
                    )
                    d3_mAP = self.mAP3d[j, 1, i]
                    d3_values.append(d3_mAP)

        d3_names.insert(0, "mAP_3D_moderate")
        overall_mAP = sum(d3_values) / len(self.current_classes)
        d3_values.insert(0, overall_mAP)

        log_info = "\n"
        for k, v in zip(names, values):
            if isinstance(v, (int, float)):
                log_info += "%s[%.2f] \n" % (k, v)
            else:
                log_info += "%s[%s] \n" % (str(k), str(v))
        logger.info(log_info)
        return (d3_names, d3_values)

    def convert_detection_to_kitti_annos(self, batch_detections, batch_labels):
        class_names = self.current_classes

        assert "calib" in batch_labels
        batch_meta_data = batch_labels["metadata"]
        batch_rect = batch_labels["calib"]["R0_rect"]
        batch_Trv2c = batch_labels["calib"]["Tr_velo_to_cam"]
        batch_P2 = batch_labels["calib"]["P2"]

        assert (
            len(batch_detections)
            == len(batch_rect)
            == len(batch_Trv2c)
            == len(batch_P2)
            == len(batch_meta_data)
        )

        predict_annos = []
        for detection, rect, Trv2c, P2, meta_data in zip(
            batch_detections,
            batch_rect,
            batch_Trv2c,
            batch_P2,
            batch_meta_data,
        ):
            (
                box3d_lidar_preds,
                labels_preds,
                scores_preds,
            ) = detection

            box3d_lidar_preds = convert_numpy(box3d_lidar_preds)
            labels_preds = convert_numpy(labels_preds)
            image_idx = convert_numpy(meta_data["image_idx"])
            image_shape = convert_numpy(meta_data["image_shape"])
            scores_preds = convert_numpy(scores_preds)

            rect = convert_numpy(rect)
            Trv2c = convert_numpy(Trv2c)
            P2 = convert_numpy(P2)

            box3d_lidar = box3d_lidar_preds
            box3d_lidar[:, -1] = limit_period(
                box3d_lidar[:, -1],
                offset=0.5,
                period=np.pi * 2,
            )
            box3d_lidar[:, 2] -= box3d_lidar[:, 5] / 2

            # aim: x, y, z, w, l, h, r -> -y, -z, x, h, w, l, r
            # (x, y, z, w, l, h r) in lidar -> (x', y', z', l, h, w, r) in camera  # noqa
            box3d_camera = box_lidar_to_camera(box3d_lidar, rect, Trv2c)
            box_2d = box3d_to_bbox(box3d_camera, rect, Trv2c, P2)

            anno = self._get_start_result_anno()
            num_example = 0
            for j in range(box3d_lidar_preds.shape[0]):
                if (
                    box_2d[j, 0] > image_shape[1]
                    or box_2d[j, 1] > image_shape[0]
                ):
                    continue
                if box_2d[j, 2] < 0 or box_2d[j, 3] < 0:
                    continue
                box_2d[j, 2:] = np.minimum(
                    box_2d[j, 2:], image_shape[::-1][1:]
                )
                box_2d[j, :2] = np.maximum(box_2d[j, :2], [0, 0])
                anno["bbox"].append(box_2d[j])
                # convert center format to kitti format
                # box3d_lidar[j, 2] -= box3d_lidar[j, 5] / 2
                anno["alpha"].append(
                    -np.arctan2(
                        -box3d_lidar_preds[j, 1], box3d_lidar_preds[j, 0]
                    )
                    + box3d_camera[j, 6]
                )
                anno["dimensions"].append(box3d_camera[j, 3:6])
                anno["location"].append(box3d_camera[j, :3])
                anno["rotation_y"].append(box3d_camera[j, 6])

                anno["name"].append(
                    self.class_to_name[class_names[int(labels_preds[j])]]
                )
                anno["truncated"].append(0.0)
                anno["occluded"].append(0)
                anno["score"].append(scores_preds[j])
                num_example += 1
            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                anno["image_idx"] = str(int(image_idx))
            else:
                anno = self._get_empty_result_anno()
                anno["image_idx"] = np.array([str(int(image_idx))])

            predict_annos.append(anno)

        return predict_annos

    def convert_gt_annos(self, batch_examples):
        labels = []
        for meta_data in batch_examples["metadata"]:
            annos = self._get_start_result_anno()
            for k in list(annos.keys()):
                if k in meta_data:
                    annos[k] = meta_data[k]
            annos["image_idx"] = str(int(meta_data["image_idx"]))
            labels.append(annos)
        return labels

    def _get_start_result_anno(self):
        return dict(
            {
                # 'image_idx': -10000,
                "name": [],
                "truncated": [],
                "occluded": [],
                "alpha": [],
                "bbox": [],
                "dimensions": [],
                "location": [],
                "rotation_y": [],
                "score": [],
            }
        )

    def _get_empty_result_anno(self):
        annotations = {}
        annotations.update(
            {
                "name": np.array([]),
                "truncated": np.array([]),
                "occluded": np.array([]),
                "alpha": np.array([]),
                "bbox": np.zeros([0, 4]),
                "dimensions": np.zeros([0, 3]),
                "location": np.zeros([0, 3]),
                "rotation_y": np.array([]),
                "score": np.array([]),
            }
        )
        return annotations


_EVAL_CLASS_NAMES = ("car", "pedestrian", "cyclist", "van", "person_sitting")
_MIN_BOX_HEIGHT = (40, 25, 25)
_MAX_OCCLUSION = (0, 1, 2)
_MAX_TRUNCATION = (0.15, 0.3, 0.5)


def _normalize_ann_name(label) -> str:
    return str(label).lower()


def _gt_semantic_match(gt_label: str, target_label: str) -> int:
    if gt_label == target_label:
        return 1
    if target_label == "pedestrian" and gt_label == "person_sitting":
        return 0
    if target_label == "car" and gt_label == "van":
        return 0
    return -1


def _gt_difficulty_ignore(bbox, occluded, truncated, diff_idx: int) -> bool:
    box_h = bbox[3] - bbox[1]
    return (
        occluded > _MAX_OCCLUSION[diff_idx]
        or truncated > _MAX_TRUNCATION[diff_idx]
        or box_h <= _MIN_BOX_HEIGHT[diff_idx]
    )


def clean_data(gt_anno, dt_anno, current_class, difficulty):
    target_cls = _EVAL_CLASS_NAMES[current_class].lower()
    ignored_gt, ignored_dt, dc_bboxes = [], [], []
    num_valid_gt = 0
    n_gt = len(gt_anno["name"])
    n_dt = len(dt_anno["name"])

    for gt_i in range(n_gt):
        gt_name = _normalize_ann_name(gt_anno["name"][gt_i])
        validity = _gt_semantic_match(gt_name, target_cls)
        ignore_flag = _gt_difficulty_ignore(
            gt_anno["bbox"][gt_i],
            gt_anno["occluded"][gt_i],
            gt_anno["truncated"][gt_i],
            difficulty,
        )
        if validity == 1 and not ignore_flag:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif validity == 0 or (ignore_flag and validity == 1):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)
        for scan_i in range(n_gt):
            tag = str(gt_anno["name"][scan_i])
            if tag in ("DontCare", "ignore"):
                dc_bboxes.append(gt_anno["bbox"][scan_i])

    for dt_i in range(n_dt):
        dt_name = _normalize_ann_name(dt_anno["name"][dt_i])
        dt_valid = 1 if dt_name == target_cls else -1
        det_h = abs(dt_anno["bbox"][dt_i, 3] - dt_anno["bbox"][dt_i, 1])
        if det_h < _MIN_BOX_HEIGHT[difficulty]:
            ignored_dt.append(1)
        elif dt_valid == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


def do_eval_v3(
    gt_annos,
    dt_annos,
    current_classes,
    min_overlaps,
    compute_aos=False,
    difficultys=(0, 1, 2),
    z_axis=1,
    z_center=1.0,
    num_parts=50,
):

    # min_overlaps: [num_minoverlap, metric, num_class]
    types = ["bbox", "bev", "3d"]
    metrics = {}
    for i in range(3):
        ret = eval_class_v3(
            gt_annos,
            dt_annos,
            current_classes,
            difficultys,
            i,
            min_overlaps,
            compute_aos,
            z_axis=z_axis,
            z_center=z_center,
            num_parts=num_parts,
        )
        metrics[types[i]] = ret

    mAP_bbox = get_mAP(metrics["bbox"]["precision"])
    mAP_aos = None
    if compute_aos:
        mAP_aos = get_mAP(metrics["bbox"]["orientation"])
    mAP_bev = get_mAP(metrics["bev"]["precision"])
    mAP_3d = get_mAP(metrics["3d"]["precision"])

    return mAP_bbox, mAP_bev, mAP_3d, mAP_aos


def get_mAP(prec):
    sums = 0
    for i in range(0, prec.shape[-1], 4):
        sums = sums + prec[..., i]
    return sums / 11 * 100


_numba_backend = None


def _lazy_numba():
    global _numba_backend
    if _numba_backend is None:
        from heal.metrics import _kitti3d_detection_numba

        _numba_backend = _kitti3d_detection_numba
    return _numba_backend


def get_thresholds(scores: np.ndarray, num_gt, num_sample_pts=41):
    return _lazy_numba().get_thresholds(scores, num_gt, num_sample_pts)


def fused_compute_statistics(*args, **kwargs):
    return _lazy_numba().fused_compute_statistics(*args, **kwargs)


def compute_statistics_jit(*args, **kwargs):
    return _lazy_numba().compute_statistics_jit(*args, **kwargs)


def image_box_overlap(*args, **kwargs):
    return _lazy_numba().image_box_overlap(*args, **kwargs)


N_SAMPLE_PTS = 41


def get_split_parts(num, num_part):
    same_part = num // num_part
    parts = [same_part] * num_part
    remain_num = num % num_part
    if remain_num:
        parts.append(remain_num)
    return parts


def _nonzero_split_parts(num_examples, num_parts):
    return [n for n in get_split_parts(num_examples, num_parts) if n != 0]


def _stack_dc_bboxes(dc_bboxes):
    if len(dc_bboxes) == 0:
        return np.zeros((0, 4), dtype=np.float64)
    return np.stack(dc_bboxes, 0).astype(np.float64)


def _concat_split_slice(data_list, start, num_part):
    return np.concatenate(data_list[start : start + num_part], 0)


def prepare_data(
    gt_annos, dt_annos, current_class, difficulty=None, clean_data=None
):
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dets, dontcares = [], [], []
    total_num_valid_gt = 0
    for gt_anno, dt_anno in zip(gt_annos, dt_annos):
        num_valid_gt, ignored_gt, ignored_det, dc_bboxes = clean_data(
            gt_anno, dt_anno, current_class, difficulty
        )
        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dets.append(np.array(ignored_det, dtype=np.int64))
        dc_bboxes = _stack_dc_bboxes(dc_bboxes)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        gt_datas_list.append(
            np.concatenate(
                [gt_anno["bbox"], gt_anno["alpha"][..., np.newaxis]], 1
            )
        )
        dt_datas_list.append(
            np.concatenate(
                [
                    dt_anno["bbox"],
                    dt_anno["alpha"][..., np.newaxis],
                    dt_anno["score"][..., np.newaxis],
                ],
                1,
            )
        )
    return (
        gt_datas_list,
        dt_datas_list,
        ignored_gts,
        ignored_dets,
        dontcares,
        np.stack(total_dc_num, axis=0),
        total_num_valid_gt,
    )


def _init_ap_metric_buffers(n_cls: int, n_diff: int, n_iou: int):
    buf_shape = (n_cls, n_diff, n_iou, N_SAMPLE_PTS)
    return (
        np.zeros(buf_shape),
        np.zeros(buf_shape),
        np.zeros(buf_shape),
        np.zeros(buf_shape),
    )


def _collect_overlap_thresholds(
    overlaps,
    gt_data,
    dt_data,
    ign_gt,
    ign_dt,
    dontcares,
    metric,
    iou_thresh: float,
    valid_gt_count: int,
) -> np.ndarray:
    score_pool = []
    for frame_idx in range(len(overlaps)):
        *_, frame_thresh = compute_statistics_jit(
            overlaps[frame_idx],
            gt_data[frame_idx],
            dt_data[frame_idx],
            ign_gt[frame_idx],
            ign_dt[frame_idx],
            dontcares[frame_idx],
            metric,
            min_overlap=iou_thresh,
            thresh=0.0,
            compute_fp=False,
        )
        score_pool.extend(frame_thresh.tolist())
    return np.array(get_thresholds(np.array(score_pool), valid_gt_count))


def _accumulate_partition_pr(
    parted_overlaps,
    split_parts,
    thresholds,
    total_gt,
    total_dt,
    total_dc,
    gt_data,
    dt_data,
    dontcares,
    ign_gt,
    ign_dt,
    metric,
    iou_thresh: float,
    use_aos: bool,
) -> np.ndarray:
    pr_stats = np.zeros((len(thresholds), 4))
    cursor = 0
    for chunk_id, chunk_sz in enumerate(split_parts):
        row_slice = slice(cursor, cursor + chunk_sz)
        fused_compute_statistics(
            parted_overlaps[chunk_id],
            pr_stats,
            total_gt[row_slice],
            total_dt[row_slice],
            total_dc[row_slice],
            _concat_split_slice(gt_data, cursor, chunk_sz),
            _concat_split_slice(dt_data, cursor, chunk_sz),
            _concat_split_slice(dontcares, cursor, chunk_sz),
            _concat_split_slice(ign_gt, cursor, chunk_sz),
            _concat_split_slice(ign_dt, cursor, chunk_sz),
            metric,
            min_overlap=iou_thresh,
            thresholds=thresholds,
            compute_aos=use_aos,
        )
        cursor += chunk_sz
    return pr_stats


def _assert_paired_annotations(gt_annos, dt_annos) -> None:
    if len(gt_annos) != len(dt_annos):
        raise AssertionError(
            "annotation count mismatch: "
            f"gt={len(gt_annos)} dt={len(dt_annos)}"
        )


def _partitioned_overlap_bundle(
    pred_annos,
    truth_annos,
    metric,
    num_parts,
    z_axis,
    z_center,
):
    """Build per-frame overlaps with SECOND/HAT (dt, gt) convention.

    ``calculate_iou_partly`` names its arguments ``(gt_annos, dt_annos)`` and
    returns counts in that same order. KITTI eval still calls it with
    detections first so the overlap matrix is ``(N_dt, N_gt)``, which
    ``compute_statistics_jit`` / ``fused_compute_statistics`` index as
    ``overlaps[det_id, gt_id]``. Swap the returned counts back to true
    GT/DT lengths for the fused PR accumulation.
    """
    overlaps, parted_overlaps, dt_as_gt_counts, gt_as_dt_counts = (
        calculate_iou_partly(
            pred_annos,
            truth_annos,
            metric,
            num_parts=num_parts,
            z_axis=z_axis,
            z_center=z_center,
        )
    )
    return overlaps, parted_overlaps, gt_as_dt_counts, dt_as_gt_counts


def eval_class_v3(
    gt_annos,
    dt_annos,
    current_classes,
    difficultys,
    metric,
    min_overlaps,
    compute_aos=False,
    z_axis=1,
    z_center=1.0,
    num_parts=50,
):
    """Run KITTI-style detection metrics across classes and difficulties.

    Supports 2D / BEV / 3D box metrics plus optional AOS, with COCO-style
    overlap sweeps (0.5:0.05:0.95).

    Returns:
        Dict with recall, precision, orientation (AOS), thresholds, and
        min_overlaps arrays.
    """
    _assert_paired_annotations(gt_annos, dt_annos)
    frame_chunks = _nonzero_split_parts(len(gt_annos), num_parts)

    (
        overlaps,
        parted_overlaps,
        total_gt_num,
        total_dt_num,
    ) = _partitioned_overlap_bundle(
        dt_annos,
        gt_annos,
        metric,
        num_parts,
        z_axis,
        z_center,
    )
    n_iou_levels = len(min_overlaps)
    n_classes = len(current_classes)
    n_diff_levels = len(difficultys)
    precision, recall, aos, all_thresholds = _init_ap_metric_buffers(
        n_classes, n_diff_levels, n_iou_levels
    )

    for cls_idx, eval_cls in enumerate(current_classes):
        for diff_idx, diff_level in enumerate(difficultys):
            (
                gt_datas_list,
                dt_datas_list,
                ignored_gts,
                ignored_dets,
                dontcares,
                total_dc_num,
                total_num_valid_gt,
            ) = prepare_data(
                gt_annos,
                dt_annos,
                eval_cls,
                difficulty=diff_level,
                clean_data=clean_data,
            )
            iou_row = min_overlaps[:, metric, cls_idx]
            for iou_idx, iou_thresh in enumerate(iou_row):
                thresholds = _collect_overlap_thresholds(
                    overlaps,
                    gt_datas_list,
                    dt_datas_list,
                    ignored_gts,
                    ignored_dets,
                    dontcares,
                    metric,
                    iou_thresh,
                    total_num_valid_gt,
                )
                all_thresholds[
                    cls_idx, diff_idx, iou_idx, : len(thresholds)
                ] = thresholds
                pr = _accumulate_partition_pr(
                    parted_overlaps,
                    frame_chunks,
                    thresholds,
                    total_gt_num,
                    total_dt_num,
                    total_dc_num,
                    gt_datas_list,
                    dt_datas_list,
                    dontcares,
                    ignored_gts,
                    ignored_dets,
                    metric,
                    iou_thresh,
                    compute_aos,
                )
                tp_fp = pr[:, 0] + pr[:, 1]
                prec_slice = precision[
                    cls_idx, diff_idx, iou_idx, : len(thresholds)
                ]
                prec_slice[:] = pr[:, 0] / tp_fp
                if compute_aos:
                    aos_slice = aos[
                        cls_idx, diff_idx, iou_idx, : len(thresholds)
                    ]
                    aos_slice[:] = pr[:, 3] / tp_fp
                precision[cls_idx, diff_idx, iou_idx, : len(thresholds)] = (
                    np.maximum.accumulate(prec_slice[::-1])[::-1]
                )
                if compute_aos:
                    aos[cls_idx, diff_idx, iou_idx, : len(thresholds)] = (
                        np.maximum.accumulate(aos_slice[::-1])[::-1]
                    )

    ret_dict = {
        "recall": recall,  # [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS]  # noqa
        "precision": precision,
        "orientation": aos,
        "thresholds": all_thresholds,
        "min_overlaps": min_overlaps,
    }
    return ret_dict


def get_mAP2(prec):
    sums = 0
    interval = 4
    for i in range(0, prec.shape[-1], interval):
        sums = sums + prec[..., i]
    return sums / int(prec.shape[-1] / interval) * 100


def _concat_bboxes(annos_part):
    return np.concatenate([anno["bbox"] for anno in annos_part], 0)


def _concat_loc_dim_rot_boxes(annos_part, bev_axes=None):
    if bev_axes is None:
        loc = np.concatenate([anno["location"] for anno in annos_part], 0)
        dims = np.concatenate([anno["dimensions"] for anno in annos_part], 0)
    else:
        loc = np.concatenate(
            [anno["location"][:, bev_axes] for anno in annos_part], 0
        )
        dims = np.concatenate(
            [anno["dimensions"][:, bev_axes] for anno in annos_part], 0
        )
    rots = np.concatenate([anno["rotation_y"] for anno in annos_part], 0)
    return np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)


def _compute_overlap_part(
    gt_annos_part, dt_annos_part, metric, bev_axes, z_axis, z_center
):
    if metric == 0:
        return image_box_overlap(
            _concat_bboxes(gt_annos_part),
            _concat_bboxes(dt_annos_part),
        )
    if metric == 1:
        return bev_box_overlap(
            _concat_loc_dim_rot_boxes(gt_annos_part, bev_axes),
            _concat_loc_dim_rot_boxes(dt_annos_part, bev_axes),
        ).astype(np.float64)
    if metric == 2:
        return box3d_overlap(
            _concat_loc_dim_rot_boxes(gt_annos_part),
            _concat_loc_dim_rot_boxes(dt_annos_part),
            z_axis=z_axis,
            z_center=z_center,
        ).astype(np.float64)
    raise ValueError("unknown metric")


def _per_frame_object_counts(annos) -> np.ndarray:
    return np.stack([len(entry["name"]) for entry in annos], axis=0)


def _plane_axes_excluding(height_axis: int) -> List[int]:
    axes = list(range(3))
    axes.pop(height_axis)
    return axes


def calculate_iou_partly(
    gt_annos,
    dt_annos,
    metric,
    num_parts: int = 50,
    z_axis: int = 1,
    z_center: float = 1.0,
):
    """Compute per-frame overlap matrices using chunked processing.

    May be called standalone for offline analysis.

    Args:
        gt_annos: Ground-truth annotation list from ``get_label_annos``.
        dt_annos: Detection annotation list from ``get_label_annos``.
        metric: 0=2D box, 1=BEV, 2=3D.
        num_parts: Chunk count for the fast partition path.
        z_axis: Height axis index (camera=1, lidar=2).
        z_center: Box center offset along ``z_axis``.
    """
    _assert_paired_annotations(gt_annos, dt_annos)

    total_dt_num = _per_frame_object_counts(dt_annos)
    total_gt_num = _per_frame_object_counts(gt_annos)
    chunk_sizes = _nonzero_split_parts(len(gt_annos), num_parts)
    plane_axes = _plane_axes_excluding(z_axis)

    parted_overlaps = []
    offset = 0
    for chunk_len in chunk_sizes:
        gt_chunk = gt_annos[offset : offset + chunk_len]
        dt_chunk = dt_annos[offset : offset + chunk_len]
        parted_overlaps.append(
            _compute_overlap_part(
                gt_chunk,
                dt_chunk,
                metric,
                plane_axes,
                z_axis,
                z_center,
            )
        )
        offset += chunk_len

    overlaps = []
    offset = 0
    for chunk_id, chunk_len in enumerate(chunk_sizes):
        gt_cursor = dt_cursor = 0
        for local_i in range(chunk_len):
            frame_gt = total_gt_num[offset + local_i]
            frame_dt = total_dt_num[offset + local_i]
            overlaps.append(
                parted_overlaps[chunk_id][
                    gt_cursor : gt_cursor + frame_gt,
                    dt_cursor : dt_cursor + frame_dt,
                ]
            )
            gt_cursor += frame_gt
            dt_cursor += frame_dt
        offset += chunk_len

    return overlaps, parted_overlaps, total_gt_num, total_dt_num


def box3d_overlap(boxes, qboxes, criterion=-1, z_axis=1, z_center=1.0):
    """Kitti camera format z_axis=1."""
    from heal.core.rotate_box_utils_v2 import rotate_iou_v2

    bev_axes = list(range(7))
    bev_axes.pop(z_axis + 3)
    bev_axes.pop(z_axis)
    rinc = rotate_iou_v2(boxes[:, bev_axes], qboxes[:, bev_axes], 2)
    _lazy_numba().box3d_overlap_kernel(
        boxes, qboxes, rinc, criterion, z_axis, z_center
    )
    return rinc


def bev_box_overlap(boxes, qboxes, criterion=-1, stable=False):
    from heal.core.rotate_box_utils_v2 import rotate_iou_v2

    if stable:
        # riou = box_np_ops.riou_cc(boxes, qboxes)
        raise ValueError("stable bev_box_overlap not implemented.")
    else:
        riou = rotate_iou_v2(boxes, qboxes, criterion)
    return riou
