import copy
import os
import re
import shutil

import numpy as np
from common import ckpt_dir, task_name
from data import inference_dataset
from model import hbir_model, hbm_model
from model import model as base_model
from PIL import Image
from qconfig import (
    calibration_predict_model_convert_pipeline,
    qat_predict_model_convert_pipeline,
)

# float inference needs Motr.forward's else branch (calls track_embed
# and post_process). deploy_model omits track_embed, so use base_model.
float_deploy_model = copy.deepcopy(base_model)


def data_loader(data_path):
    if not data_path or not str(data_path).strip():
        raise ValueError(
            "`data_path` cannot be empty. "
            "Please provide a valid image directory path."
        )

    input_paths = []
    for root, _, file_names in os.walk(data_path):
        for file_name in file_names:
            if file_name.lower().endswith(".jpg") and os.path.isfile(
                os.path.join(root, file_name)
            ):
                input_paths.append(os.path.join(root, file_name))

    def extract_key(filename):
        # Sort by leading integer in basename (e.g. "0.jpg", "1.jpg").
        base = os.path.basename(filename)
        match = re.match(r"(\d+)", base)
        if match:
            return int(match.group(1))
        return float("inf")

    input_paths.sort(key=extract_key)
    return input_paths


def pre_processor(data, transforms=None):
    # `data` is a single image path; wrap into a single-frame
    # frame_data_list for MotrIrInfer.
    image = Image.open(data).convert("RGB")
    ori_img = np.array(image)
    model_input = {
        "frame_data_list": [
            {
                "img": ori_img,
                "ori_img": ori_img,
                "layout": "hwc",
                "color_space": "rgb",
                "img_name": [os.path.basename(data)],
                "seq_name": ["demo"],
            }
        ]
    }
    if transforms is not None:
        model_input = transforms(model_input)

    frame = model_input["frame_data_list"][0]
    frame["img"] = [frame["img"].unsqueeze(0)]
    return model_input


def post_processor(model_outs, data):
    # Restore ori_img so TrackViz can draw boxes on original frame.
    frame = data["frame_data_list"][0]
    data["ori_img"] = frame.get("ori_img", frame.get("img"))
    return model_outs, data


infer_transforms = [
    dict(
        type="SeqResize",
        img_scale=(800, 1422),
        keep_ratio=False,
    ),
    dict(
        type="SeqToTensor",
        to_yuv=False,
    ),
    dict(type="SeqBgrToYuv444", rgb_input=True),
    dict(
        type="SeqNormalize",
        mean=128.0,
        std=128.0,
    ),
]

hbm_infer_transforms = [
    dict(
        type="SeqResize",
        img_scale=(800, 1422),
        keep_ratio=False,
    ),
    dict(
        type="SeqToTensor",
        to_yuv=False,
    ),
]


def data_saver(data, save_path):
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    # Save each frame of the sequence as {idx}.jpg for multi-frame
    # tracking visualization (MOTR carries state across frames).
    frame_data_list = data["frame_data_list"]
    for img_idx, frame_data in enumerate(frame_data_list):
        img_data = frame_data["ori_img"]
        if isinstance(img_data, np.ndarray):
            img_data = Image.fromarray(img_data.astype(np.uint8), mode="RGB")
        img_data.save(os.path.join(save_path, f"{img_idx}.jpg"), "JPEG")


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
                    allow_miss=True,
                    ignore_extra=True,
                    verbose=True,
                ),
            ],
        ),
        pre_processors=pre_processor,
        post_processors=post_processor,
        visualize_processors=dict(
            type="TrackViz",
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
