"""Inference and gen_demo configuration."""

import copy
import logging
import os
import re
import shutil

import cv2
import numpy as np
import torch
from common import bev_size, march, resize_shape, task_name, use_bce
from data import single_infer_dataset
from hat_engine.registry import OBJECT_REGISTRY
from model import hbm_infer_model, single_hbir_infer_model, test_model
from PIL import Image
from qconfig import (
    calibration_predict_model_convert_pipeline,
    float_predict_model_convert_pipeline,
    qat_predict_model_convert_pipeline,
)
from torchvision.transforms.functional import pil_to_tensor

from heal.data.transforms.functional_img import image_pad

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


def process_img(img_path, resize_size, pad_divisor):
    orig_img = cv2.imread(img_path)
    cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB, orig_img)
    orig_img = Image.fromarray(orig_img)
    orig_img = pil_to_tensor(orig_img)
    resize_hw = (int(resize_size[0]), int(resize_size[1]))
    orig_shape = (orig_img.shape[1], orig_img.shape[2])
    resized_img = resize(orig_img, resize_hw)
    resized_img = image_pad(
        resized_img, "chw", None, pad_divisor, 0
    ).unsqueeze(0)
    return resized_img, orig_shape


def process_img_deploy(img_path, deploy_size):
    """Resize demo image to compiled HBM input shape."""
    orig_img = cv2.imread(img_path)
    cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB, orig_img)
    orig_img = Image.fromarray(orig_img)
    orig_img = pil_to_tensor(orig_img)
    deploy_hw = (int(deploy_size[0]), int(deploy_size[1]))
    orig_shape = (orig_img.shape[1], orig_img.shape[2])
    deploy_img = resize(orig_img, deploy_hw).unsqueeze(0)
    scale = (deploy_hw[0] / orig_shape[1], deploy_hw[1] / orig_shape[0])
    return deploy_img, orig_shape, scale


def prepare_inputs(infer_inputs):
    file_list = list(os.listdir(infer_inputs))
    image_dir_list = list(filter(lambda x: x.endswith(".png"), file_list))

    def extract_key(filename):
        match = re.search(r"img(\d+)_", filename)
        if match:
            return int(match.group(1))
        return float("inf")

    image_dir_list.sort(key=extract_key)
    ego2img = np.load(os.path.join(infer_inputs, "ego2img.npy"))
    ego2global = np.load(os.path.join(infer_inputs, "ego2global.npy"))
    lidar2img = np.load(os.path.join(infer_inputs, "lidar2img.npy"))
    frames_inputs = []
    num_cam = 6
    num_frame = len(image_dir_list) // num_cam
    for i in range(num_frame):
        frame_inputs = {}
        img_paths = image_dir_list[i * num_cam : (i + 1) * num_cam]
        img_paths = [os.path.join(infer_inputs, p) for p in img_paths]
        frame_inputs["img_paths"] = img_paths
        frame_inputs["ego2img"] = ego2img[i * num_cam : (i + 1) * num_cam]
        frame_inputs["lidar2img"] = lidar2img[i * num_cam : (i + 1) * num_cam]
        frame_inputs["ego2global"] = ego2global[:, i : i + 1]
        frames_inputs.append(frame_inputs)
    frames_inputs.reverse()
    return frames_inputs


def process_inputs(infer_inputs, transforms=None, stage=None):
    resize_size = resize_shape[1:]
    orig_imgs = []

    if stage == "hbm":
        # HBM: uint8 RGB/NV12, same 450+Pad geometry as float/hbir.
        pad_divisor = 32  # noqa: E501
        for i, img_path in enumerate(infer_inputs["img_paths"]):
            img, orig_shape = process_img(img_path, resize_size, pad_divisor)
            orig_imgs.append({"name": i, "img": img})
        input_imgs = torch.cat([item["img"] for item in orig_imgs])
        if input_imgs.dtype != torch.uint8:
            input_imgs = input_imgs.to(torch.uint8)
        ego2img = infer_inputs["ego2img"]
        scale = (
            resize_size[0] / orig_shape[0],
            resize_size[1] / orig_shape[1],
        )
        ego2img = resize_homo(ego2img, scale)
        ego2global = infer_inputs["ego2global"]
    else:
        pad_divisor = 32
        for i, img_path in enumerate(infer_inputs["img_paths"]):
            img, orig_shape = process_img(img_path, resize_size, pad_divisor)
            orig_imgs.append({"name": i, "img": img})

        input_imgs = []
        for orig_img in orig_imgs:
            if horizon is None:
                raise ImportError(
                    "horizon_plugin_pytorch is required for BevFormer infer."
                )
            input_img = horizon.nn.functional.bgr_to_yuv444(
                orig_img["img"], True
            )
            input_imgs.append(input_img)

        input_imgs = torch.cat(input_imgs)
        input_imgs = (input_imgs - 128.0) / 128.0
        ego2img = infer_inputs["ego2img"]
        scale = (
            resize_size[0] / orig_shape[0],
            resize_size[1] / orig_shape[1],
        )
        ego2img = resize_homo(ego2img, scale)
        ego2global = infer_inputs["ego2global"]

    model_input = {
        "img": input_imgs,
        "seq_meta": [
            {
                "meta": [{"scene": "test_infer"}],
                "ego2img": [ego2img],
                "ego2global": ego2global,
            }
        ],
    }
    if transforms is not None:
        model_input = transforms(model_input)

    vis_inputs = {"img": orig_imgs, "meta": {"ego2img": ego2img}}
    return model_input, vis_inputs


def resolve_demo_sample_dir(data_path):
    """Resolve gen_demo output dir containing ``ego2img.npy``."""
    if os.path.isfile(os.path.join(data_path, "ego2img.npy")):
        return data_path
    default_dir = os.path.join(data_path, "sample_idx_0")
    if os.path.isfile(os.path.join(default_dir, "ego2img.npy")):
        return default_dir
    if os.path.isdir(data_path):
        for name in sorted(os.listdir(data_path)):
            sub_dir = os.path.join(data_path, name)
            if os.path.isfile(os.path.join(sub_dir, "ego2img.npy")):
                return sub_dir
    return data_path


def data_loader(data_path):
    sample_dir = resolve_demo_sample_dir(data_path)
    if not os.path.isdir(sample_dir):
        raise FileNotFoundError("Demo directory not found: %s" % sample_dir)
    if not os.path.isfile(os.path.join(sample_dir, "ego2img.npy")):
        raise FileNotFoundError(
            "No ego2img.npy under %s. Run tools/gen_demo.py first."
            % sample_dir
        )
    frames = prepare_inputs(sample_dir)
    return [
        {
            "frame_input": frame,
            "save_path": sample_dir,
            "frame_idx": idx,
        }
        for idx, frame in enumerate(frames)
    ]


def pre_processor(data, transforms=None):
    model_input, vis_inputs = process_inputs(
        data["frame_input"], transforms, stage=data.get("stage")
    )
    model_input["vis_imgs"] = vis_inputs["img"]
    model_input["vis_meta"] = vis_inputs["meta"]
    model_input["save_path"] = data.get("save_path")
    model_input["frame_idx"] = data.get("frame_idx")
    if "stage" in data:
        model_input["stage"] = data["stage"]
    return model_input


def post_processor(pred, data):
    return {"bev_det": pred}, data


def _vis_inputs_from_data(data):
    imgs = []
    for item in data["vis_imgs"]:
        img = item["img"]
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
        imgs.append({"name": item["name"], "img": img})
    meta = copy.deepcopy(data["vis_meta"])
    homo = meta.get("ego2img", meta.get("lidar2img"))
    if isinstance(homo, torch.Tensor):
        homo = homo.detach().cpu().numpy()
    if homo is not None:
        if "ego2img" in meta:
            meta["ego2img"] = homo
        else:
            meta["lidar2img"] = homo
    return imgs, meta


@OBJECT_REGISTRY.register
class BevformerTinyNuscenesInferViz(object):
    """Adapter from :class:`Inference` to :class:`NuscenesViz`."""

    def __init__(
        self,
        bev_size=bev_size,
        is_plot=True,
        use_bce=use_bce,
        score_thresh=0.3,
        save_path=None,
    ):
        from heal.visualize.nuscenes import NuscenesViz

        self.viz = NuscenesViz(
            bev_size=bev_size,
            is_plot=is_plot,
            use_bce=use_bce,
            score_thresh=score_thresh,
        )
        self.save_path = save_path

    def __call__(self, pred, data):
        preds = (
            pred
            if isinstance(pred, dict) and "bev_det" in pred
            else {"bev_det": pred}
        )
        imgs, meta = _vis_inputs_from_data(data)
        stage = data.get("stage")
        save_path = data.get("save_path") or self.save_path
        sample_idx = self.viz.sample_idx
        self.viz(imgs, preds, meta, save_path=save_path, stage=stage)
        if save_path:
            out_png = os.path.join(
                save_path,
                self.viz.result_filename(sample_idx, stage),
            )
            logger.info("Visualization saved to %s", out_png)


def inputs_save_func(data, save_path):
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    lidar2imgs = []
    lidar2globals = []
    ego2imgs = []
    ego2globals = []
    for seq_idx, sample_data in enumerate(data):
        img_idx_start = seq_idx * len(sample_data["img_name"])
        for image_idx, (img_name, img_data) in enumerate(
            zip(sample_data["img_name"], sample_data["img"])
        ):
            image_idx_save = image_idx + img_idx_start
            save_name = (
                f"img{image_idx_save}_{os.path.basename(img_name)}".replace(
                    ".jpg", ".png"
                )
            )
            img_data.save(os.path.join(save_path, save_name), "PNG")
        lidar2imgs.append(sample_data["lidar2img"])
        lidar2globals.append(sample_data["lidar2global"])
        ego2imgs.append(sample_data["ego2img"])
        ego2globals.append(sample_data["ego2global"])
    lidar2img_path = os.path.join(save_path, "lidar2img.npy")
    lidar2imgs_np = np.array(lidar2imgs)
    lidar2imgs_np = lidar2imgs_np.reshape(
        -1, lidar2imgs_np.shape[-2], lidar2imgs_np.shape[-1]
    )
    np.save(lidar2img_path, lidar2imgs_np)
    lidar2global_path = os.path.join(save_path, "lidar2global.npy")
    np.save(lidar2global_path, np.array(lidar2globals)[None])
    ego2img_path = os.path.join(save_path, "ego2img.npy")
    ego2imgs_np = np.array(ego2imgs)
    ego2imgs_np = ego2imgs_np.reshape(
        -1, ego2imgs_np.shape[-2], ego2imgs_np.shape[-1]
    )
    np.save(ego2img_path, ego2imgs_np)
    ego2global_path = os.path.join(save_path, "ego2global.npy")
    np.save(ego2global_path, np.array(ego2globals)[None])


demo_root = os.path.join(".", "demo", task_name)

gen_demo_cfg = dict(
    dataset=single_infer_dataset,
    sample_idx=[2],
    save_path=demo_root,
    data_saver=inputs_save_func,
)

_infer_common = dict(
    data_path=demo_root,
    data_loader=data_loader,
    transforms=None,
    infer=dict(
        type="Inference",
        pre_processors=pre_processor,
        post_processors=post_processor,
        visualize_processors=dict(
            type="BevformerTinyNuscenesInferViz",
            is_plot=True,
            bev_size=bev_size,
            use_bce=use_bce,
            score_thresh=0.3,
            save_path=demo_root,
        ),
    ),
)

float_infer = copy.deepcopy(_infer_common)
float_infer["infer"]["model"] = test_model
float_infer["infer"]["march"] = march
float_infer["infer"][
    "model_convert_pipeline"
] = float_predict_model_convert_pipeline

calibration_infer = copy.deepcopy(_infer_common)
calibration_infer["infer"]["model"] = test_model
calibration_infer["infer"]["march"] = march
calibration_infer["infer"][
    "model_convert_pipeline"
] = calibration_predict_model_convert_pipeline

qat_infer = copy.deepcopy(_infer_common)
qat_infer["infer"]["model"] = test_model
qat_infer["infer"]["march"] = march
qat_infer["infer"][
    "model_convert_pipeline"
] = qat_predict_model_convert_pipeline

hbir_infer = copy.deepcopy(_infer_common)
hbir_infer["infer"]["model"] = single_hbir_infer_model
hbir_infer["infer"]["march"] = march
hbir_infer["infer"].pop("model_convert_pipeline", None)

hbm_infer = copy.deepcopy(_infer_common)
hbm_infer["infer"]["model"] = hbm_infer_model
hbm_infer["infer"]["march"] = march
hbm_infer["infer"].pop("model_convert_pipeline", None)
