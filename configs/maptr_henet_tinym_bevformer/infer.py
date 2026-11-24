"""Inference and demo configuration for MapTR."""

import copy
import logging
import os
import re
import shutil

import cv2
import numpy as np
import torch
from common import (
    march,
    point_cloud_range,
    resize_shape,
    task_name,
    use_lidar_gt,
)
from data import single_infer_dataset
from hat_engine.registry import OBJECT_REGISTRY
from model import hbm_infer_model, single_hbir_infer_model, test_model
from PIL import Image
from qconfig import (
    calibration_predict_model_convert_pipeline,
    float_predict_model_convert_pipeline,
)
from torchvision.transforms.functional import pil_to_tensor

from heal.data.transforms.functional_img import image_pad
from heal.visualize.nuscenes_map import NuscenesMapViz

logger = logging.getLogger(__name__)

homo_key = "lidar2img" if use_lidar_gt else "ego2img"

try:
    from torchvision.transforms.functional_tensor import resize
except ImportError:
    from torchvision.transforms._functional_tensor import resize

try:
    import horizon_plugin_pytorch as horizon
except ImportError:
    horizon = None


def inputs_save_func(data, save_path):
    if isinstance(data, dict):
        data = [data]
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    homo_key = "lidar2img" if use_lidar_gt else "ego2img"
    homos = []
    osm_masks = []
    for seq_idx, sample_data in enumerate(data):
        img_idx_start = seq_idx * len(sample_data["img_name"])
        for image_idx, (img_name, img_data) in enumerate(
            zip(sample_data["img_name"], sample_data["img"])
        ):
            image_idx_save = image_idx + img_idx_start
            save_name = f"img{image_idx_save}_{os.path.basename(img_name)}"
            img_data.save(os.path.join(save_path, save_name), "JPEG")
        homos.append(sample_data[homo_key])
        osm_masks.append(sample_data["osm_mask"].numpy())
    np.save(os.path.join(save_path, f"{homo_key}.npy"), np.array(homos))
    np.save(os.path.join(save_path, "osm_mask.npy"), np.array(osm_masks))


def prepare_inputs(infer_inputs):
    file_list = list(os.listdir(infer_inputs))
    image_dir_list = sorted(
        filter(lambda x: x.endswith(".jpg"), file_list),
        key=lambda filename: (
            int(re.search(r"img(\d+)_", filename).group(1))
            if re.search(r"img(\d+)_", filename)
            else float("inf")
        ),
    )
    homo_key = "lidar2img" if use_lidar_gt else "ego2img"
    homos = np.load(os.path.join(infer_inputs, f"{homo_key}.npy"))
    osm_mask = np.load(os.path.join(infer_inputs, "osm_mask.npy"))
    frames_inputs = []
    num_cam = 6
    num_frame = len(image_dir_list) // num_cam
    for i in range(num_frame):
        frame_inputs = {
            "img_paths": [
                os.path.join(infer_inputs, p)
                for p in image_dir_list[i * num_cam : (i + 1) * num_cam]
            ],
            homo_key: homos[i],
            "osm_mask": osm_mask[i],
        }
        frames_inputs.append(frame_inputs)
    frames_inputs.reverse()
    return frames_inputs


def process_img(img_path, resize_size, pad_divisor):
    orig_img = cv2.imread(img_path)
    cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB, orig_img)
    orig_img = pil_to_tensor(Image.fromarray(orig_img))
    resize_hw = (int(resize_size[0]), int(resize_size[1]))
    orig_shape = (orig_img.shape[1], orig_img.shape[2])
    resized_img = image_pad(
        resize(orig_img, resize_hw), "chw", None, pad_divisor, 0
    ).unsqueeze(0)
    return resized_img, orig_shape


def resize_homo(homo, scale):
    view = np.eye(4)
    view[0, 0] = scale[1]
    view[1, 1] = scale[0]
    return view @ homo


def process_inputs(infer_inputs, transforms=None, stage=None):
    pad_divisor = 32
    resize_size = resize_shape[1:]
    orig_imgs = []
    for i, img_path in enumerate(infer_inputs["img_paths"]):
        img, orig_shape = process_img(img_path, resize_size, pad_divisor)
        orig_imgs.append({"name": i, "img": img})

    if stage == "hbm":
        input_imgs = torch.cat([item["img"] for item in orig_imgs])
        if input_imgs.dtype != torch.uint8:
            input_imgs = input_imgs.to(torch.uint8)
    else:
        if horizon is None:
            raise ImportError(
                "horizon_plugin_pytorch is required for float/hbir MapTR infer."  # noqa: E501
            )
        input_imgs = []
        for orig_img in orig_imgs:
            input_imgs.append(
                horizon.nn.functional.bgr_to_yuv444(orig_img["img"], True)
            )
        input_imgs = (torch.cat(input_imgs) - 128.0) / 128.0

    homo_key = "lidar2img" if use_lidar_gt else "ego2img"
    homo = infer_inputs[homo_key]
    osm_mask = torch.tensor(infer_inputs["osm_mask"]).unsqueeze(0)
    scale = (resize_size[0] / orig_shape[0], resize_size[1] / orig_shape[1])
    homo = resize_homo(homo, scale)

    model_input = {
        "img": input_imgs,
        "osm_mask": osm_mask,
        "seq_meta": [
            {
                "meta": [{"scene": "test_infer"}],
                homo_key: [homo],
            }
        ],
    }
    if transforms is not None:
        model_input = transforms(model_input)
    return model_input, {"img": orig_imgs}


def process_outputs(model_outs, viz_func, vis_inputs):
    viz_func(vis_inputs["img"], preds=model_outs)
    return None


def resolve_demo_sample_dir(data_path):
    """Resolve gen_demo output dir containing homography and osm mask."""
    if os.path.isfile(os.path.join(data_path, f"{homo_key}.npy")):
        return data_path
    default_dir = os.path.join(data_path, "sample_idx_0")
    if os.path.isfile(os.path.join(default_dir, f"{homo_key}.npy")):
        return default_dir
    if os.path.isdir(data_path):
        for name in sorted(os.listdir(data_path)):
            sub_dir = os.path.join(data_path, name)
            if os.path.isfile(os.path.join(sub_dir, f"{homo_key}.npy")):
                return sub_dir
    return data_path


def data_loader(data_path):
    sample_dir = resolve_demo_sample_dir(data_path)
    if not os.path.isdir(sample_dir):
        raise FileNotFoundError("Demo directory not found: %s" % sample_dir)
    if not os.path.isfile(os.path.join(sample_dir, f"{homo_key}.npy")):
        raise FileNotFoundError(
            "No %s under %s. Run tools/gen_demo.py first."
            % (homo_key, sample_dir)
        )
    if not os.path.isfile(os.path.join(sample_dir, "osm_mask.npy")):
        raise FileNotFoundError(
            "No osm_mask.npy under %s. Run tools/gen_demo.py first."
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
        data["frame_input"],
        transforms=transforms,
        stage=data.get("stage"),
    )
    model_input["vis_imgs"] = vis_inputs["img"]
    model_input["save_path"] = data.get("save_path")
    model_input["frame_idx"] = data.get("frame_idx")
    if "stage" in data:
        model_input["stage"] = data["stage"]
    return model_input


def post_processor(pred, data):
    return pred, data


def _vis_imgs_from_data(data):
    imgs = []
    for item in data["vis_imgs"]:
        img = item["img"]
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
        imgs.append({"name": item["name"], "img": img})
    return imgs


@OBJECT_REGISTRY.register
class MaptrBevformerNuscenesMapInferViz(object):
    """Adapter from :class:`Inference` to :class:`NuscenesMapViz`."""

    def __init__(
        self,
        pc_range=point_cloud_range,
        is_plot=True,
        car_img_path=None,
        use_lidar=use_lidar_gt,
        save_path=None,
    ):
        self.viz = NuscenesMapViz(
            is_plot=is_plot,
            pc_range=pc_range,
            car_img_path=car_img_path,
            use_lidar=use_lidar,
        )
        self.save_path = save_path

    def __call__(self, pred, data):
        stage = data.get("stage") if isinstance(data, dict) else None
        base_dir = data.get("save_path") or self.save_path or demo_sample_dir
        sample_idx = self.viz.sample_idx
        logger.info(
            "Visualizing map predictions (stage=%s, save_path=%s)",
            stage,
            base_dir,
        )
        self.viz(
            _vis_imgs_from_data(data),
            preds=pred,
            save_path=base_dir,
            stage=stage,
        )
        if base_dir:
            out_png = os.path.join(
                base_dir,
                self.viz.result_filename(sample_idx, stage),
            )
            logger.info("Visualization saved to %s", out_png)


car_img_path = (
    "tmp_orig_data/nuscenes/infer_imgs/single_frame_lidarego/cars/lidar_car.png"  # noqa: E501
    if use_lidar_gt
    else "tmp_orig_data/nuscenes/infer_imgs/single_frame_lidarego/cars/car.png"
)

demo_root = os.path.join(".", "demo", task_name)
demo_sample_dir = os.path.join(demo_root, "sample_idx_0")

gen_demo_cfg = dict(
    dataset=single_infer_dataset,
    sample_idx=[0],
    save_path=demo_root,
    data_saver=inputs_save_func,
)

infer_cfg = dict(
    model=single_hbir_infer_model,
    input_path=demo_root,
    gen_inputs_cfg=gen_demo_cfg,
    process_inputs=process_inputs,
    viz_func=dict(
        type="NuscenesMapViz",
        is_plot=True,
        pc_range=point_cloud_range,
        car_img_path=car_img_path,
        use_lidar=use_lidar_gt,
    ),
    process_outputs=process_outputs,
    prepare_inputs=prepare_inputs,
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
            type="MaptrBevformerNuscenesMapInferViz",
            is_plot=True,
            pc_range=point_cloud_range,
            car_img_path=car_img_path,
            use_lidar=use_lidar_gt,
            save_path=demo_sample_dir,
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

hbir_infer = copy.deepcopy(_infer_common)
hbir_infer["infer"]["model"] = single_hbir_infer_model
hbir_infer["infer"]["march"] = march
hbir_infer["infer"].pop("model_convert_pipeline", None)

hbm_infer = copy.deepcopy(_infer_common)
hbm_infer["infer"]["model"] = hbm_infer_model
hbm_infer["infer"]["march"] = march
hbm_infer["infer"].pop("model_convert_pipeline", None)

gen_ref_type = "bevformer"
