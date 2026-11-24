import copy
import os
import shutil

import numpy as np
import torch
from common import ckpt_dir, data_shape, task_name
from data import inference_dataset
from model import deploy_model, hbir_model, hbm_model
from PIL import Image
from qconfig import (
    calibration_predict_model_convert_pipeline,
    qat_predict_model_convert_pipeline,
)


def data_loader(data_path):
    if not data_path or not str(data_path).strip():
        raise ValueError("`data_path` cannot be empty.")

    img_paths = []
    for root, _, file_names in os.walk(data_path):
        if "img.jpg" in file_names:
            img_paths.append(os.path.join(root, "img.jpg"))
    direct_path = os.path.join(data_path, "img.jpg")
    if os.path.isfile(direct_path):
        img_paths.append(direct_path)
    img_paths = sorted(set(img_paths))
    if not img_paths:
        raise FileNotFoundError(
            f"Missing demo image under {data_path}. "
            f"Run `python3 tools/gen_demo.py -c configs/{task_name}` first."
        )

    outputs = []
    for img_path in img_paths:
        image = Image.open(img_path).convert("RGB")
        ori_img = np.ascontiguousarray(np.array(image).transpose((2, 0, 1)))
        outputs.append(
            {
                "img": image,
                "ori_img": ori_img,
                "layout": "chw",
                "color_space": "rgb",
            }
        )
    return outputs


def pre_processor(data, transforms=None):
    model_input = {
        "img": data["img"],
        "ori_img": data["ori_img"],
        "layout": data.get("layout", "chw"),
        "color_space": data.get("color_space", "rgb"),
    }
    if transforms is not None:
        model_input = transforms(model_input)
    model_input["img"] = model_input["img"].unsqueeze(0)
    model_input["vis_img"] = data["img"]
    if "stage" in data:
        model_input["stage"] = data["stage"]
    return model_input


def post_processor(model_outs, data):
    preds = model_outs[0]
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
    if preds.ndim == 4:
        preds = torch.argmax(preds, dim=1, keepdim=False)
    data["pred"] = preds
    data["img"] = data.pop("vis_img")
    return model_outs, data


infer_transforms = [
    dict(type="PILToTensor"),
    dict(type="SegResize", size=data_shape[1:]),
    dict(type="BgrToYuv444", rgb_input=True),
    dict(
        type="TorchVisionAdapter",
        interface="Normalize",
        mean=128.0,
        std=128.0,
    ),
]

hbm_infer_transforms = [
    dict(type="PILToTensor"),
    dict(type="SegResize", size=data_shape[1:]),
    dict(type="BgrToYuv444", rgb_input=True),
    dict(
        type="TorchVisionAdapter",
        interface="Normalize",
        mean=128.0,
        std=128.0,
    ),
]


def data_saver(data, save_path):
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    data["img"].save(os.path.join(save_path, "img.jpg"), "JPEG")


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
        model=deploy_model,
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
            type="SegViz",
            is_plot=True,
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
