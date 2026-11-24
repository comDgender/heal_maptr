"""Tests for heal.visualize.nuscenes."""

import numpy as np
import torch

from heal.visualize.nuscenes import NuscenesViz, _as_numpy, _image_hw


def test_as_numpy_from_tensor_and_array():
    arr = _as_numpy(torch.ones(2, 3))
    assert arr.shape == (2, 3)
    assert _as_numpy(np.zeros(4)).shape == (4,)


def test_image_hw_from_chw_tensor():
    h, w = _image_hw({"img": torch.zeros(3, 64, 128)})
    assert (h, w) == (64, 128)


def test_result_filename_includes_stage():
    assert NuscenesViz.result_filename(0, "float") == "nusc_pred_float_0.png"
    assert NuscenesViz.result_filename(1) == "nusc_pred_1.png"


def test_nuscenes_viz_call_no_plot():
    viz = NuscenesViz(is_plot=False, score_thresh=0.1)
    imgs = [{"name": "cam0", "img": np.zeros((32, 48, 3), dtype=np.uint8)}]
    preds = {"lidar_det": [np.zeros((0, 11), dtype=np.float32)]}
    meta = {"lidar2img": [np.eye(4, dtype=np.float32)]}
    viz(imgs, preds, meta, save_path=None, stage="hbir")
    assert viz.sample_idx == 1


def test_nuscenes_viz_saves_when_save_path_set_without_is_plot(tmp_path):
    viz = NuscenesViz(is_plot=False, score_thresh=0.1)
    imgs = [{"name": "cam0", "img": np.zeros((32, 48, 3), dtype=np.uint8)}]
    preds = {"lidar_det": [np.zeros((0, 11), dtype=np.float32)]}
    meta = {"lidar2img": [np.eye(4, dtype=np.float32)]}
    viz(imgs, preds, meta, save_path=str(tmp_path), stage="hbir")
    assert (tmp_path / "nusc_pred_hbir_0.png").is_file()
