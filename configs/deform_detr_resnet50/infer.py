import copy
import os

import numpy as np
import torch
from common import ckpt_dir, task_name
from data import inference_dataset
from model import deploy_model, hbir_model, hbm_model
from model import model as train_model
from PIL import Image
from qconfig import (
    calibration_predict_model_convert_pipeline,
    qat_predict_model_convert_pipeline,
)

from heal.data.collates.collates import collate_2d

float_deploy_model = copy.deepcopy(deploy_model)
float_deploy_model["post_process"] = train_model["post_process"]


def data_loader(data_path):
    if not data_path or not str(data_path).strip():
        raise ValueError(
            "`data_path` cannot be empty. "
            "Please provide a valid image directory path."
        )

    input_paths = []
    for root, _, file_names in os.walk(data_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            if file_name.lower().endswith(".jpg") and os.path.isfile(
                file_path
            ):
                input_paths.append(file_path)

    outputs = []
    for path in input_paths:
        img = Image.open(path).convert("RGB")
        image = np.array(img)
        data = {
            "img": image,
            "ori_img": image,
            "img_name": [os.path.basename(path)],
            "img_id": [0],
            "layout": "hwc",
            "color_space": "rgb",
            "img_shape": image.shape[0:2],
            "vis_img": img,
        }
        outputs.append(data)

    return outputs


def pre_processor(data, transforms=None):
    model_input = {
        "img": data["img"],
        "ori_img": data["ori_img"],
        "img_name": data["img_name"],
        "img_id": data["img_id"],
        "layout": data["layout"],
        "color_space": data["color_space"],
        "img_shape": data["img_shape"],
    }
    if transforms is not None:
        model_input = transforms(model_input)

    batched = collate_2d([model_input])
    batched["vis_img"] = data["vis_img"]
    if "stage" in data:
        batched["stage"] = data["stage"]
    return batched


def post_processor(model_outs, data):
    if isinstance(model_outs, dict) and "pred_bboxes" in model_outs:
        pred_bboxes = model_outs["pred_bboxes"]
        if isinstance(pred_bboxes, torch.Tensor) and pred_bboxes.dim() == 3:
            model_outs["pred_bboxes"] = [pred_bboxes[0]]
    data["ori_img"] = data.get("ori_img", data["img"])
    data["img"] = data.pop("vis_img")
    return model_outs, data


infer_transforms = [
    dict(
        type="Resize",
        img_scale=[(800, 1333)],
        multiscale_mode="value",
        keep_ratio=False,
    ),
    dict(
        type="ToTensor",
        to_yuv=True,
        use_yuv_v2=False,
    ),
    dict(
        type="Normalize",
        mean=128.0,
        std=128.0,
    ),
]

hbm_infer_transforms = [
    dict(
        type="Resize",
        img_scale=[(800, 1332)],
        multiscale_mode="value",
        keep_ratio=False,
    ),
    dict(
        type="ToTensor",
        to_yuv=False,
        use_yuv_v2=False,
    ),
]


def data_saver(data, save_path):
    if os.path.isdir(save_path):
        import shutil

        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    img = data["ori_img"]
    if isinstance(img, Image.Image):
        pil_img = img
    else:
        arr = np.asarray(img)
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        pil_img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
    pil_img.save(os.path.join(save_path, "img.jpg"), "JPEG")


gen_demo_cfg = dict(
    dataset=inference_dataset,
    sample_idx=[0],
    save_path=f"./demo/{task_name}",
    data_saver=data_saver,
)


float_infer = dict(
    data_path=f"./demo/{task_name}",
    data_loader=data_loader,
    transforms=infer_transforms,
    infer=dict(
        type="Inference",
        model=float_deploy_model,
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
        visualize_processors=dict(
            type="DetViz",
            is_plot=True,
            score_thresh=0.4,
            save_path=f"./demo/{task_name}",
        ),
    ),
)

calibration_infer = copy.deepcopy(float_infer)
calibration_infer["infer"][
    "model_convert_pipeline"
] = calibration_predict_model_convert_pipeline

qat_infer = copy.deepcopy(float_infer)
qat_infer["infer"][
    "model_convert_pipeline"
] = qat_predict_model_convert_pipeline

hbir_infer = copy.deepcopy(float_infer)
hbir_infer["infer"]["model"] = hbir_model
hbir_infer["infer"].pop("model_convert_pipeline", None)

hbm_infer = copy.deepcopy(float_infer)
hbm_infer["infer"]["model"] = hbm_model
hbm_infer["transforms"] = hbm_infer_transforms
hbm_infer["infer"].pop("model_convert_pipeline", None)
