"""Inference and demo configuration for sparse MapTR."""

import copy
import logging
import os
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
    val_data_shape,
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

from heal.visualize.nuscenes_map import NuscenesMapViz

logger = logging.getLogger(__name__)

try:
    import horizon_plugin_pytorch as horizon
except ImportError:
    horizon = None

try:
    from torchvision.transforms.functional_tensor import resize
except ImportError:
    from torchvision.transforms._functional_tensor import resize


def inputs_save_func(data, save_path):
    if isinstance(data, dict):
        data = [data]
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    for idx_, sample_data in enumerate(data):
        save_dir = os.path.join(save_path, f"frame{idx_}")
        os.makedirs(save_dir, exist_ok=True)
        for image_idx, (img_name, img_data) in enumerate(
            zip(sample_data["img_name"], sample_data["img"])
        ):
            save_name = f"img{image_idx}_{os.path.basename(img_name)}"
            img_data.save(os.path.join(save_dir, save_name), "JPEG")
        lidar2img_path = os.path.join(save_dir, "lidar2img.npy")
        np.save(lidar2img_path, np.array(sample_data["lidar2img"]))


def prepare_inputs(infer_inputs):
    dir_list = os.listdir(infer_inputs)
    dir_list.sort()
    input_datas = []
    for _, frame in enumerate(dir_list):
        data = {}
        frame_path = os.path.join(infer_inputs, frame)
        file_list = list(os.listdir(frame_path))
        image_dir_list = list(filter(lambda x: x.endswith(".jpg"), file_list))
        image_dir_list.sort()

        data["imgs"] = [
            os.path.join(frame_path, tmpdir) for tmpdir in image_dir_list
        ]
        lidar2img_path = os.path.join(frame_path, "lidar2img.npy")
        data["lidar2img"] = np.load(lidar2img_path)
        input_datas.append(data)
    return input_datas


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


def process_inputs(data, transforms=None):
    resize_size = resize_shape[1:]
    input_size = val_data_shape[1:]

    orig_imgs = []
    for i, img_path in enumerate(data["imgs"]):
        img, orig_shape = process_img(img_path, resize_size, input_size)
        orig_imgs.append({"name": i, "img": img})

    input_imgs = []
    for orig_img in orig_imgs:
        input_imgs.append(
            horizon.nn.functional.bgr_to_yuv444(orig_img["img"], True)
        )
    input_imgs = torch.cat(input_imgs)
    input_imgs = (input_imgs - 128.0) / 128.0

    homo = data["lidar2img"]
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

    vis_inputs = {"img": orig_imgs}
    return model_input, vis_inputs


def process_outputs(model_outs, viz_func, vis_inputs):
    viz_func(vis_inputs["img"], preds=model_outs)
    return None


def resolve_demo_sample_dir(data_path):
    """Resolve gen_demo output dir (``sample_idx_*``) containing ``frame*`` dirs."""  # noqa: E501
    if os.path.isdir(os.path.join(data_path, "frame0")):
        return data_path
    default_dir = os.path.join(data_path, "sample_idx_0")
    if os.path.isdir(os.path.join(default_dir, "frame0")):
        return default_dir
    if os.path.isdir(data_path):
        for name in sorted(os.listdir(data_path)):
            if not name.startswith("sample_idx_"):
                continue
            sub_dir = os.path.join(data_path, name)
            if os.path.isdir(os.path.join(sub_dir, "frame0")):
                return sub_dir
    return data_path


def data_loader(data_path):
    sample_dir = resolve_demo_sample_dir(data_path)
    if not os.path.isdir(sample_dir):
        raise FileNotFoundError("Demo directory not found: %s" % sample_dir)
    frames = sorted(
        name
        for name in os.listdir(sample_dir)
        if os.path.isdir(os.path.join(sample_dir, name))
        and name.startswith("frame")
    )
    if not frames:
        raise FileNotFoundError(
            "No frame* dirs under %s. Run tools/gen_demo.py first."
            % sample_dir
        )
    return [
        {"data_path": os.path.join(sample_dir, frame), "save_path": sample_dir}
        for frame in frames
    ]


def pre_processor(data, transforms=None):
    frame_path = data["data_path"]
    image_dir_list = sorted(
        name for name in os.listdir(frame_path) if name.endswith(".jpg")
    )
    if not image_dir_list:
        raise FileNotFoundError("No .jpg images under %s" % frame_path)

    resize_size = resize_shape[1:]
    input_size = val_data_shape[1:]
    orig_imgs = []
    for i, img_name in enumerate(image_dir_list):
        img_path = os.path.join(frame_path, img_name)
        img, orig_shape = process_img(img_path, resize_size, input_size)
        orig_imgs.append({"name": i, "img": img})

    if data.get("stage") == "hbm":
        input_imgs = torch.cat([item["img"] for item in orig_imgs])
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
        input_imgs = torch.cat(input_imgs)
        input_imgs = (input_imgs - 128.0) / 128.0

    homo = np.load(os.path.join(frame_path, "lidar2img.npy"))
    top = int(resize_size[0] - input_size[0])
    left = int((resize_size[1] - input_size[1]) / 2)
    scale = (resize_size[0] / orig_shape[0], resize_size[1] / orig_shape[1])
    homo = crop_homo(resize_homo(homo, scale), (left, top))

    model_input = {
        "img": input_imgs,
        "lidar2img": torch.tensor(homo),
    }
    if transforms is not None:
        model_input = transforms(model_input)

    model_input["vis_imgs"] = orig_imgs
    model_input["save_path"] = data.get("save_path")
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
class MaptrSparseNuscenesMapInferViz(object):
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
    sample_idx=[0, 1],
    save_path=demo_root,
    data_saver=inputs_save_func,
)

infer_cfg = dict(
    model=single_hbir_infer_model,
    input_path=demo_root,
    gen_inputs_cfg=gen_demo_cfg,
    prepare_inputs=prepare_inputs,
    process_inputs=process_inputs,
    viz_func=dict(
        type="NuscenesMapViz",
        is_plot=True,
        pc_range=point_cloud_range,
        car_img_path=car_img_path,
        use_lidar=use_lidar_gt,
    ),
    process_outputs=process_outputs,
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
            type="MaptrSparseNuscenesMapInferViz",
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
