# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import copy
import logging
import os
import shutil

import cv2
import numpy as np
import torch
from common import bev_size, resize_shape, task_name, use_bce, val_data_shape
from data import single_infer_dataset
from hat_engine.registry import OBJECT_REGISTRY
from model import hbir_model, hbir_model_convert_pipeline, hbm_model, model
from PIL import Image
from qconfig import (
    calibration_predict_model_convert_pipeline,
    qat_predict_model_convert_pipeline,
)
from torchvision.transforms.functional import pil_to_tensor

from heal.data.datasets.nuscenes.nuscenes_dataset import CAM_NAMES

logger = logging.getLogger(__name__)

try:
    from torchvision.transforms.functional_tensor import resize
except ImportError:
    from torchvision.transforms._functional_tensor import resize

try:
    import horizon_plugin_pytorch as horizon
except ImportError:
    horizon = None


def resize_homo(homo, scale):
    view = np.eye(4)
    view[0, 0] = scale[1]
    view[1, 1] = scale[0]
    return view @ homo


def crop_homo(homo, offset):
    view = np.eye(4)
    view[0, 2] = -offset[0]
    view[1, 2] = -offset[1]
    return view @ homo


def process_img(img_path, resize_size, crop_size):
    orig_img = cv2.imread(img_path)
    cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB, orig_img)
    orig_img = Image.fromarray(orig_img)
    orig_img = pil_to_tensor(orig_img)
    resize_hw = (int(resize_size[0]), int(resize_size[1]))
    orig_shape = (orig_img.shape[1], orig_img.shape[2])
    resized_img = resize(orig_img, resize_hw).unsqueeze(0)
    top = int(resize_hw[0] - crop_size[0])
    left = int((resize_hw[1] - crop_size[1]) / 2)
    resized_img = resized_img[:, :, top:, left:]
    return resized_img, orig_shape


def resolve_demo_sample_dir(data_path):
    """Resolve gen_demo output dir (``sample_idx_*``) from task demo root."""
    if os.path.isfile(os.path.join(data_path, "lidar2img.npy")):
        return data_path
    default_dir = os.path.join(data_path, "sample_idx_0")
    if os.path.isfile(os.path.join(default_dir, "lidar2img.npy")):
        return default_dir
    if os.path.isdir(data_path):
        for name in sorted(os.listdir(data_path)):
            sub_dir = os.path.join(data_path, name)
            if os.path.isfile(os.path.join(sub_dir, "lidar2img.npy")):
                return sub_dir
    return data_path


def data_loader(data_path):
    if not data_path or not str(data_path).strip():
        raise ValueError("`data_path` cannot be empty.")
    if not os.path.isdir(data_path):
        raise FileNotFoundError("Demo directory not found: %s" % data_path)
    return [{"data_path": resolve_demo_sample_dir(data_path)}]


def pre_processor(data, transforms=None):
    data_path = data["data_path"]
    resize_size = resize_shape[1:]
    input_size = val_data_shape[1:]

    file_list = sorted(
        [name for name in os.listdir(data_path) if name.endswith(".jpg")]
    )
    if not file_list:
        raise FileNotFoundError(
            "No .jpg images under %s. Run tools/gen_demo.py first." % data_path
        )

    orig_imgs = []
    orig_shape = None
    for i, img_name in enumerate(file_list):
        img_path = os.path.join(data_path, img_name)
        img, shape_hw = process_img(img_path, resize_size, input_size)
        if orig_shape is None:
            orig_shape = shape_hw
        orig_imgs.append(
            {
                "name": CAM_NAMES[i] if i < len(CAM_NAMES) else i,
                "img": img,
            }
        )

    if data.get("stage") == "hbm":
        input_imgs = torch.cat([item["img"] for item in orig_imgs])
    else:
        if horizon is None:
            raise ImportError(
                "horizon_plugin_pytorch is required for float/hbir BEV infer."
            )
        input_imgs = []
        for orig_img in orig_imgs:
            yuv = horizon.nn.functional.bgr_to_yuv444(orig_img["img"], True)
            input_imgs.append(yuv)
        input_imgs = torch.cat(input_imgs)
        input_imgs = (input_imgs - 128.0) / 128.0

    homo = np.load(os.path.join(data_path, "lidar2img.npy"))
    top = int(resize_size[0] - input_size[0])
    left = int((resize_size[1] - input_size[1]) / 2)
    scale = (resize_size[0] / orig_shape[0], resize_size[1] / orig_shape[1])
    homo = resize_homo(homo, scale)
    homo = crop_homo(homo, (left, top))

    model_input = {
        "img": input_imgs,
        "lidar2img": torch.tensor(homo),
    }
    if transforms is not None:
        model_input = transforms(model_input)

    model_input["vis_imgs"] = orig_imgs
    model_input["vis_meta"] = {"lidar2img": homo}
    if "stage" in data:
        model_input["stage"] = data["stage"]
    return model_input


def post_processor(model_outs, data):
    return model_outs, data


def _format_preds_for_viz(preds):
    """Convert CenterPoint decode dict to tensors expected by NuscenesViz."""
    if not isinstance(preds, dict) or "bev_det" not in preds:
        return preds
    det = preds["bev_det"]
    if not isinstance(det, list) or not det:
        return preds
    item = det[0]
    if not isinstance(item, dict) or "bboxes" not in item:
        return preds
    bboxes = item["bboxes"]
    scores = item["scores"]
    labels = item["labels"]
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.detach().cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if len(bboxes) == 0:
        formatted = np.zeros((0, 11), dtype=np.float32)
    else:
        formatted = np.concatenate(
            [bboxes, scores.reshape(-1, 1), labels.reshape(-1, 1)], axis=1
        )
        formatted = formatted.copy()
        formatted[:, [3, 4]] = formatted[:, [4, 3]]
    return {"lidar_det": [formatted]}


def _vis_inputs_from_data(data):
    imgs = []
    for item in data["vis_imgs"]:
        img = item["img"]
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
        imgs.append({"name": item["name"], "img": img})
    meta = copy.deepcopy(data["vis_meta"])
    homo = meta.get("lidar2img")
    if isinstance(homo, torch.Tensor):
        meta["lidar2img"] = homo.detach().cpu().numpy()
    return imgs, meta


@OBJECT_REGISTRY.register
class BevVtV2NuscenesInferViz(object):
    """Adapter from :class:`Inference` to :class:`NuscenesViz`."""

    def __init__(
        self,
        bev_size=bev_size,
        score_thresh=0.1,
        is_plot=True,
        use_bce=use_bce,
        save_path=None,
    ):
        from heal.visualize.nuscenes import NuscenesViz

        self.viz = NuscenesViz(
            bev_size=bev_size,
            score_thresh=score_thresh,
            is_plot=is_plot,
            use_bce=use_bce,
        )
        self.save_path = save_path

    def __call__(self, pred, data):
        preds = pred[1] if isinstance(pred, tuple) and len(pred) >= 2 else pred
        preds = _format_preds_for_viz(preds)
        num_boxes = 0
        if isinstance(preds, dict) and "lidar_det" in preds:
            det = preds["lidar_det"][0]
            num_boxes = len(det) if hasattr(det, "__len__") else 0
        logger.info("Visualizing %d detections", num_boxes)
        imgs, meta = _vis_inputs_from_data(data)
        stage = data.get("stage") if isinstance(data, dict) else None
        sample_idx = self.viz.sample_idx
        self.viz(imgs, preds, meta, save_path=self.save_path, stage=stage)
        if self.save_path:
            out_png = os.path.join(
                self.save_path,
                self.viz.result_filename(sample_idx, stage),
            )
            logger.info("Visualization saved to %s", out_png)


def data_saver(data, save_path):
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    for image_idx, img in enumerate(data["img"]):
        if isinstance(img, Image.Image):
            pil_img = img
        else:
            arr = np.asarray(img)
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = np.transpose(arr, (1, 2, 0))
            pil_img = Image.fromarray(arr.astype(np.uint8))
        save_name = "img%d_%s.jpg" % (
            image_idx,
            CAM_NAMES[image_idx] if image_idx < len(CAM_NAMES) else image_idx,
        )
        pil_img.save(os.path.join(save_path, save_name), "JPEG")
    np.save(
        os.path.join(save_path, "lidar2img.npy"), np.asarray(data["lidar2img"])
    )


demo_root = os.path.join(".", "demo", task_name)
demo_sample_dir = os.path.join(demo_root, "sample_idx_0")

gen_demo_cfg = dict(
    dataset=single_infer_dataset,
    sample_idx=[0],
    save_path=demo_root,
    data_saver=data_saver,
)

_infer_common = dict(
    data_path=demo_sample_dir,
    data_loader=data_loader,
    transforms=None,
    infer=dict(
        type="Inference",
        pre_processors=pre_processor,
        post_processors=post_processor,
        visualize_processors=dict(
            type="BevVtV2NuscenesInferViz",
            bev_size=bev_size,
            score_thresh=0.1,
            is_plot=True,
            use_bce=use_bce,
            save_path=demo_sample_dir,
        ),
    ),
)

float_infer = copy.deepcopy(_infer_common)
float_infer["infer"]["model"] = model
float_infer["infer"]["model_convert_pipeline"] = hbir_model_convert_pipeline

calibration_infer = copy.deepcopy(_infer_common)
calibration_infer["infer"]["model"] = model
calibration_infer["infer"][
    "model_convert_pipeline"
] = calibration_predict_model_convert_pipeline

qat_infer = copy.deepcopy(_infer_common)
qat_infer["infer"]["model"] = model
qat_infer["infer"][
    "model_convert_pipeline"
] = qat_predict_model_convert_pipeline

hbir_infer = copy.deepcopy(_infer_common)
hbir_infer["infer"]["model"] = hbir_model
hbir_infer["infer"].pop("model_convert_pipeline", None)

hbm_infer = copy.deepcopy(_infer_common)
hbm_infer["infer"]["model"] = hbm_model
hbm_infer["infer"].pop("model_convert_pipeline", None)
