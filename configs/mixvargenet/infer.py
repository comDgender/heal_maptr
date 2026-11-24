import copy
import os

from common import ckpt_dir, task_name
from data import inference_dataset
from model import deploy_model, hbir_model, hbm_model
from PIL import Image
from qconfig import calibration_predict_model_convert_pipeline

from heal.data.datasets.imagenet.meta import Classes


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
        data = {
            "img": img,
            "classes": Classes,
        }
        outputs.append(data)

    return outputs


def pre_processor(data, transforms=None):
    model_input = {
        "img": data["img"],
    }
    if transforms is not None:
        model_input = transforms(model_input)

    model_input["img"] = model_input["img"].unsqueeze(0)
    model_input["classes"] = data["classes"]
    model_input["vis_img"] = data["img"]
    if "stage" in data:
        model_input["stage"] = data["stage"]
    return model_input


def post_processor(model_outs, data):
    data["img"] = data.pop("vis_img")
    return model_outs, data


infer_transforms = [
    dict(type="TorchVisionAdapter", interface="Resize", size=256),
    dict(type="TorchVisionAdapter", interface="CenterCrop", size=224),
    dict(type="TorchVisionAdapter", interface="PILToTensor"),
    dict(type="BgrToYuv444", rgb_input=True),
    dict(
        type="TorchVisionAdapter",
        interface="Normalize",
        mean=128.0,
        std=128.0,
    ),
]

hbm_infer_transforms = [
    dict(type="TorchVisionAdapter", interface="Resize", size=256),
    dict(type="TorchVisionAdapter", interface="CenterCrop", size=224),
    dict(type="TorchVisionAdapter", interface="PILToTensor"),
]


def data_saver(data, save_path):
    save_name = "img.jpg"
    img_data = data["img"]
    img_data.save(os.path.join(save_path, save_name), "JPEG")


gen_demo_cfg = dict(
    dataset=inference_dataset,
    sample_idx=[500],
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
            type="ClsViz",
            save_path=f"./demo/{task_name}",
        ),
    ),
)

calibration_infer = copy.deepcopy(float_infer)
calibration_infer["infer"][
    "model_convert_pipeline"
] = calibration_predict_model_convert_pipeline

hbir_infer = copy.deepcopy(float_infer)
hbir_infer["infer"]["model"] = hbir_model
hbir_infer["infer"].pop("model_convert_pipeline", None)

hbm_infer = copy.deepcopy(float_infer)
hbm_infer["infer"]["model"] = hbm_model
hbm_infer["transforms"] = hbm_infer_transforms
hbm_infer["infer"].pop("model_convert_pipeline", None)
