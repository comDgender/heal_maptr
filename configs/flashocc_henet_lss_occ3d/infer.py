"""Inference (gen_demo + tools/inference.py) config for FlashOcc."""

import copy
import os

import numpy as np
import torch
from common import ckpt_dir, data_shape, march, resize_shape, task_name
from data import single_infer_dataset
from model import hbir_infer_model, hbm_infer_model
from model import model as train_model
from PIL import Image
from qconfig import calibration_predict_model_convert_pipeline

from heal.visualize.occ import OccViz

DEMO_DIR = f"./demo/{task_name}"
_SAMPLE_IDX = 1545
INFER_DATA_DIR = os.path.join(DEMO_DIR, f"sample_idx_{_SAMPLE_IDX}")


def inputs_save_func(data, save_path):
    """Save 6 multi-view images + ego2img.npy under ``save_path``."""
    os.makedirs(save_path, exist_ok=True)
    for image_idx, img_data in enumerate(data["img"]):
        save_name = f"img{image_idx}.png"
        if isinstance(img_data, np.ndarray):
            img_data = Image.fromarray(img_data.astype(np.uint8), mode="RGB")
        img_data.save(os.path.join(save_path, save_name), "PNG")
    np.save(os.path.join(save_path, "ego2img.npy"), np.array(data["ego2img"]))


gen_demo_cfg = dict(
    dataset=single_infer_dataset,
    sample_idx=[_SAMPLE_IDX],
    save_path=DEMO_DIR,
    data_saver=inputs_save_func,
)


def data_loader(data_path):
    """Load 6 png + ego2img.npy back from the demo dir."""
    if not os.path.isfile(os.path.join(data_path, "ego2img.npy")):
        data_path = os.path.join(data_path, f"sample_idx_{_SAMPLE_IDX}")
    files = sorted(
        f
        for f in os.listdir(data_path)
        if f.startswith("img") and f.endswith(".png")
    )
    imgs = [
        Image.open(os.path.join(data_path, f)).convert("RGB") for f in files
    ]
    ego2img = np.load(os.path.join(data_path, "ego2img.npy"))
    return [
        {
            "img": imgs,
            "ego2img": [ego2img[i] for i in range(ego2img.shape[0])],
            "img_name": files,
            "layout": "chw",
            "color_space": "rgb",
        }
    ]


def pre_processor(data, transforms=None):
    """Apply transforms and stack multi-view inputs into batched tensors."""
    if transforms is not None:
        data = transforms(data)
    if isinstance(data["img"], list):
        data["img"] = torch.stack([torch.as_tensor(x) for x in data["img"]])
    if isinstance(data["ego2img"], list):
        data["ego2img"] = torch.as_tensor(np.stack(data["ego2img"])).float()
    return data


def post_processor(pred, data):
    """Identity post-processor (model already returns final occ map)."""
    return pred, data


_occ_visualizer = OccViz(
    vcs_range=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
    vis_bev_2d=True,
)


def visualize_occ(pred, data):
    """Save BEV OCC visualization under INFER_DATA_DIR/<stage>/."""
    if isinstance(pred, (tuple, list)):
        occ_pre = pred[1]["occ_pre"]
    elif isinstance(pred, dict) and "occ_pre" in pred:
        occ_pre = pred["occ_pre"]
    else:
        occ_pre = pred
    stage = data.get("stage", "float")
    save_path = os.path.join(INFER_DATA_DIR, stage)
    occ_arr = occ_pre.squeeze().detach().cpu().numpy().astype(np.uint8)
    _occ_visualizer(occ_arr, save_path=save_path)


infer_transforms = [
    dict(type="MultiViewsImgResize", size=resize_shape[1:]),
    dict(type="MultiViewsImgCrop", size=data_shape[1:]),
    dict(
        type="MultiViewsImgTransformWrapper",
        transforms=[
            dict(type="PILToTensor"),
            dict(type="BgrToYuv444", rgb_input=True),
            dict(type="Normalize", mean=128.0, std=128.0),
        ],
    ),
]

hbm_infer_transforms = [
    dict(type="MultiViewsImgResize", size=resize_shape[1:]),
    dict(type="MultiViewsImgCrop", size=data_shape[1:]),
    dict(
        type="MultiViewsImgTransformWrapper",
        transforms=[
            dict(type="PILToTensor"),
        ],
    ),
]


float_infer = dict(
    data_path=INFER_DATA_DIR,
    data_loader=data_loader,
    transforms=infer_transforms,
    infer=dict(
        type="Inference",
        model=train_model,
        march=march,
        model_convert_pipeline=dict(
            type="ModelConvertPipeline",
            converters=[
                dict(
                    type="LoadCheckpoint",
                    checkpoint_path=os.path.join(
                        ckpt_dir, "float-checkpoint-best.pth.tar"
                    ),
                ),
            ],
        ),
        pre_processors=pre_processor,
        post_processors=post_processor,
        visualize_processors=visualize_occ,
    ),
)

# calibration: Float2QAT (cali qconfig) + calibration checkpoint
calibration_infer = copy.deepcopy(float_infer)
calibration_infer["infer"][
    "model_convert_pipeline"
] = calibration_predict_model_convert_pipeline

hbir_infer = copy.deepcopy(float_infer)
hbir_infer["infer"]["model"] = hbir_infer_model
hbir_infer["infer"].pop("model_convert_pipeline", None)

hbm_infer = copy.deepcopy(float_infer)
hbm_infer["infer"]["model"] = hbm_infer_model
hbm_infer["transforms"] = hbm_infer_transforms
hbm_infer["infer"].pop("model_convert_pipeline", None)
