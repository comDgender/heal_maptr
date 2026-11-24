import copy
import os

import numpy as np
import torch
from common import ckpt_dir, task_name
from data import inference_dataset
from model import deploy_model, hbir_model, hbm_model
from model import model as train_model
from PIL import Image
from qconfig import calibration_predict_model_convert_pipeline

# deploy_model has post_process=None (used for export). For float inference
# we need the decoder attached so forward returns decoded lanes instead of
# raw heatmaps. Pattern matches fcos3d_efficientnetb0/infer.py.
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
            "layout": "hwc",
            "color_space": "rgb",
            "img_shape": image.shape,
            "vis_img": img,
        }
        outputs.append(data)

    return outputs


def pre_processor(data, transforms=None):
    # Keep the original (un-transformed) image for visualization; HAT passes
    # this as vis_inputs to viz_func. transforms would otherwise resize/
    # crop ori_img, mismatching decoder outputs which are in original-image
    # coordinates.
    ori_img_for_viz = data["ori_img"]
    model_input = {
        "img": data["img"],
        "ori_img": data["ori_img"],
        "layout": data["layout"],
        "color_space": data["color_space"],
        "img_shape": data["img_shape"],
    }
    if transforms is not None:
        model_input = transforms(model_input)

    model_input["img"] = model_input["img"].unsqueeze(0)
    model_input["ori_img"] = [model_input["ori_img"]]
    model_input["crop_offset"] = [model_input["crop_offset"]]
    model_input["scale_factor"] = [torch.tensor(model_input["scale_factor"])]
    model_input["vis_img"] = data["vis_img"]
    model_input["ori_img_for_viz"] = ori_img_for_viz
    if "stage" in data:
        model_input["stage"] = data["stage"]
    return model_input


def post_processor(model_outs, data):
    data["img"] = data.pop("vis_img")
    # Restore the original image so viz draws on the full-size canvas that
    # matches the decoder's original-image coordinates.
    data["ori_img"] = data.pop("ori_img_for_viz")
    return model_outs, data


infer_transforms = [
    dict(
        type="FixedCrop",
        size=(0, 270, 1640, 320),
    ),
    dict(
        type="Resize",
        img_scale=(320, 800),
        multiscale_mode="value",
        keep_ratio=False,
    ),
    dict(
        type="ToTensor",
        to_yuv=False,
    ),
    dict(type="BgrToYuv444", rgb_input=True),
    dict(
        type="TorchVisionAdapter",
        interface="Normalize",
        mean=128.0,
        std=128.0,
    ),
]

hbm_infer_transforms = [
    dict(
        type="FixedCrop",
        size=(0, 270, 1640, 320),
    ),
    dict(
        type="Resize",
        img_scale=(320, 800),
        multiscale_mode="value",
        keep_ratio=False,
    ),
    dict(
        type="ToTensor",
        to_yuv=False,
    ),
]


def data_saver(data, save_path):
    save_name = "img.jpg"
    img_data = data["img"]
    if isinstance(img_data, np.ndarray):
        img_data = Image.fromarray(img_data.astype(np.uint8), mode="RGB")
    os.makedirs(save_path, exist_ok=True)
    img_data.save(os.path.join(save_path, save_name), "JPEG")


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
            type="LanelineViz",
            is_plot=True,
            save_path=f"./demo/{task_name}/float",
        ),
    ),
)

calibration_infer = copy.deepcopy(float_infer)
calibration_infer["infer"][
    "model_convert_pipeline"
] = calibration_predict_model_convert_pipeline
calibration_infer["infer"]["visualize_processors"][
    "save_path"
] = f"./demo/{task_name}/calibration"

hbir_infer = copy.deepcopy(float_infer)
hbir_infer["infer"]["model"] = hbir_model
hbir_infer["infer"].pop("model_convert_pipeline", None)
hbir_infer["infer"]["visualize_processors"][
    "save_path"
] = f"./demo/{task_name}/hbir"

hbm_infer = copy.deepcopy(float_infer)
hbm_infer["infer"]["model"] = hbm_model
hbm_infer["transforms"] = hbm_infer_transforms
hbm_infer["infer"].pop("model_convert_pipeline", None)
hbm_infer["infer"]["visualize_processors"][
    "save_path"
] = f"./demo/{task_name}/hbm"
