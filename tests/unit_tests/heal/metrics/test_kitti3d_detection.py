"""Tests for heal.metrics.kitti3d_detection."""

import numpy as np


def _make_kitti_anno(name="Car", bbox=None, score=0.9, image_idx="0"):
    bbox = (
        bbox
        if bbox is not None
        else np.array([100.0, 150.0, 200.0, 250.0], dtype=np.float32)
    )
    return {
        "name": np.array([name]),
        "truncated": np.array([0.0]),
        "occluded": np.array([0]),
        "alpha": np.array([0.0]),
        "bbox": bbox.reshape(1, 4),
        "dimensions": np.array([[1.6, 1.5, 3.9]], dtype=np.float32),
        "location": np.array([[10.0, 1.5, 30.0]], dtype=np.float32),
        "rotation_y": np.array([0.0]),
        "score": np.array([score]),
        "image_idx": image_idx,
    }


def _cat_annos(*annos):
    merged = {key: [] for key in annos[0] if key != "image_idx"}
    for anno in annos:
        for key in merged:
            merged[key].append(anno[key])
    out = {key: np.concatenate(vals) for key, vals in merged.items()}
    out["image_idx"] = annos[0]["image_idx"]
    return out


def _car_min_overlaps():
    overlap_0_7 = np.tile(
        np.array([0.7, 0.5, 0.5, 0.7, 0.5, 0.7, 0.7, 0.7], dtype=np.float64),
        (3, 1),
    )
    overlap_0_5 = np.array(
        [
            [0.7, 0.5, 0.5, 0.7, 0.5, 0.5, 0.5, 0.5],
            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5, 0.5, 0.5],
            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5, 0.5, 0.5],
        ],
        dtype=np.float64,
    )
    return np.stack((overlap_0_7, overlap_0_5), axis=0)[:, :, [0]]


def test_kitti3d_metric_det_instantiation():
    from heal.metrics.kitti3d_detection import Kitti3DMetricDet

    metric = Kitti3DMetricDet(
        current_classes=["Car"],
        compute_aos=True,
        difficultys=[0, 1, 2],
    )
    assert metric.name == "kitti3dAPDet"


def test_kitti3d_eval_nonzero_when_pred_gt_counts_differ():
    from heal.metrics.kitti3d_detection import eval_class_v3

    gt = _make_kitti_anno()
    dt = _cat_annos(
        _make_kitti_anno(score=0.95),
        _make_kitti_anno(
            bbox=np.array([500.0, 550.0, 600.0, 650.0], dtype=np.float32),
            score=0.1,
        ),
    )
    result = eval_class_v3(
        [gt],
        [dt],
        [0],
        [0],
        0,
        _car_min_overlaps(),
        compute_aos=False,
        num_parts=1,
    )
    assert result["precision"].max() > 0.5


def test_partitioned_overlap_counts_match_true_gt_dt():
    """dt-first IoU must still report GT/DT lengths, not parameter names."""
    from heal.metrics.kitti3d_detection import _partitioned_overlap_bundle

    box_a = np.array([100.0, 150.0, 200.0, 250.0], dtype=np.float32)
    box_b = np.array([300.0, 160.0, 400.0, 260.0], dtype=np.float32)
    gt_frames = [
        _make_kitti_anno(bbox=box_a, image_idx="0"),
        _cat_annos(
            _make_kitti_anno(bbox=box_a, image_idx="1"),
            _make_kitti_anno(bbox=box_b, image_idx="1"),
        ),
    ]
    dt_frames = [
        _cat_annos(
            _make_kitti_anno(bbox=box_a, score=0.99, image_idx="0"),
            _make_kitti_anno(
                bbox=np.array([500.0, 550.0, 600.0, 650.0], dtype=np.float32),
                score=0.1,
                image_idx="0",
            ),
        ),
        _make_kitti_anno(bbox=box_a, score=0.98, image_idx="1"),
    ]
    _, _, total_gt_num, total_dt_num = _partitioned_overlap_bundle(
        dt_frames, gt_frames, metric=0, num_parts=2, z_axis=1, z_center=1.0
    )
    np.testing.assert_array_equal(total_gt_num, np.array([1, 2]))
    np.testing.assert_array_equal(total_dt_num, np.array([2, 1]))


def test_kitti3d_eval_high_ap_on_multi_frame_unequal_counts():
    """Fused PR slicing must use true GT/DT counts, not swapped IoU args."""
    from heal.metrics.kitti3d_detection import do_eval_v3, eval_class_v3

    box_a = np.array([100.0, 150.0, 200.0, 250.0], dtype=np.float32)
    box_b = np.array([300.0, 160.0, 400.0, 260.0], dtype=np.float32)
    box_fp = np.array([500.0, 550.0, 600.0, 650.0], dtype=np.float32)

    gt_frames = [
        _make_kitti_anno(bbox=box_a, image_idx="0"),
        _cat_annos(
            _make_kitti_anno(bbox=box_a, image_idx="1"),
            _make_kitti_anno(bbox=box_b, image_idx="1"),
        ),
    ]
    dt_frames = [
        _cat_annos(
            _make_kitti_anno(bbox=box_a, score=0.99, image_idx="0"),
            _make_kitti_anno(bbox=box_fp, score=0.1, image_idx="0"),
        ),
        _make_kitti_anno(bbox=box_a, score=0.98, image_idx="1"),
    ]
    min_overlaps = _car_min_overlaps()
    result = eval_class_v3(
        gt_frames,
        dt_frames,
        [0],
        [0],
        0,
        min_overlaps,
        compute_aos=False,
        num_parts=2,
    )
    assert result["precision"].max() > 0.5
    # Toy sets only fill a few of 41 recall bins; 11-point mAP is still > 0
    # once GT/DT counts used by fused PR are not swapped.
    mAP_bbox, _, _, _ = do_eval_v3(
        gt_frames,
        dt_frames,
        [0],
        min_overlaps,
        compute_aos=False,
        difficultys=[0],
        num_parts=2,
    )
    assert mAP_bbox[0, 0, 0] > 0.0
