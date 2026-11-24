"""Offline inference helpers for SparseBEV lidar fusion."""

import copy
import os
import shutil

import cv2
import horizon_plugin_pytorch as horizon
import numpy as np
import torch
from common import march, resize_shape, task_name, val_data_shape
from data import single_infer_dataset
from hat_engine.registry import OBJECT_REGISTRY
from model import hbir_infer_model, hbm_infer_model
from model import model as train_model
from PIL import Image
from qconfig import (
    calibration_predict_model_convert_pipeline,
    float_checkpoint,
    qat_val_model_convert_pipeline,
)
from torchvision.transforms.functional import pil_to_tensor

try:
    from torchvision.transforms.functional_tensor import resize
except ImportError:
    from torchvision.transforms._functional_tensor import resize

from heal.visualize.nuscenes import NuscenesMultitaskViz


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


def prepare_inputs(infer_inputs):
    dir_list = sorted(os.listdir(infer_inputs))
    input_datas = []
    for frame in dir_list:
        data = {}
        frame_path = os.path.join(infer_inputs, frame)
        image_dir_list = sorted(
            [f for f in os.listdir(frame_path) if f.endswith(".jpg")]
        )
        data["imgs"] = [os.path.join(frame_path, p) for p in image_dir_list]
        data["timestamp"] = np.load(
            os.path.join(frame_path, "timestamp.npy")
        ).reshape(1)
        data["lidar2global"] = np.load(
            os.path.join(frame_path, "lidar2global.npy")
        ).reshape(1, 4, 4)
        data["lidar2img"] = np.load(os.path.join(frame_path, "lidar2img.npy"))
        data["points"] = np.load(os.path.join(frame_path, "lidar.npy"))
        input_datas.append(data)
    return input_datas


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
    homo = crop_homo(resize_homo(homo, scale), (left, top))

    model_input = {
        "img": input_imgs,
        "lidar2img": torch.tensor(homo),
        "lidar2global": torch.tensor(data["lidar2global"]),
        "timestamp": torch.tensor(data["timestamp"]),
        "points": [torch.tensor(data["points"])],
    }
    if transforms is not None:
        model_input = transforms(model_input)
    vis_inputs = {
        "img": orig_imgs,
        "meta": {"lidar2img": homo},
        "points": model_input["points"],
    }
    return model_input, vis_inputs


def process_outputs(model_outs, viz_func, vis_inputs):
    outs = torch.cat(
        [
            model_outs[0]["bboxes"][..., :9].view(1, -1, 9),
            model_outs[0]["scores"].view(1, -1, 1),
            model_outs[0]["labels"].view(1, -1, 1),
        ],
        dim=-1,
    )
    outs[..., 3], outs[..., 4] = outs[..., 4], outs[..., 3]
    viz_func(
        vis_inputs["img"],
        vis_inputs["points"],
        {"lidar_det": outs},
        vis_inputs["meta"],
    )
    return None


def inputs_save_func(data, save_path):
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
        np.save(
            os.path.join(save_dir, "lidar2global.npy"),
            np.array(sample_data["lidar2global"]),
        )
        np.save(
            os.path.join(save_dir, "lidar2img.npy"),
            np.array(sample_data["lidar2img"]),
        )
        np.save(
            os.path.join(save_dir, "timestamp.npy"),
            np.array(sample_data["timestamp"]),
        )
        if "points" in sample_data:
            np.save(os.path.join(save_dir, "lidar.npy"), sample_data["points"])
        elif "lidar" in sample_data and "points" in sample_data["lidar"]:
            np.save(
                os.path.join(save_dir, "lidar.npy"),
                sample_data["lidar"]["points"],
            )


# Single-frame saver used by tools/gen_demo.py: ``dataset[idx]`` returns one
# sample dict, so save its 6 multi-view images + lidar/nuscenes meta straight
# into ``sample_path`` (``demo/<task>/sample_idx_<i>``). The resulting dir
# matches what ``prepare_inputs`` expects (one frame dir = img*.jpg + npy).
def data_saver(sample, save_path):
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    for image_idx, (img_name, img_data) in enumerate(
        zip(sample["img_name"], sample["img"])
    ):
        save_name = f"img{image_idx}_{os.path.basename(img_name)}"
        img_data.save(os.path.join(save_path, save_name), "JPEG")
    np.save(
        os.path.join(save_path, "lidar2global.npy"),
        np.array(sample["lidar2global"]),
    )
    np.save(
        os.path.join(save_path, "lidar2img.npy"),
        np.array(sample["lidar2img"]),
    )
    np.save(
        os.path.join(save_path, "timestamp.npy"),
        np.array(sample["timestamp"]),
    )
    if "points" in sample:
        np.save(os.path.join(save_path, "lidar.npy"), sample["points"])
    elif "lidar" in sample and "points" in sample["lidar"]:
        np.save(
            os.path.join(save_path, "lidar.npy"),
            sample["lidar"]["points"],
        )


demo_root = os.path.join(".", "demo", task_name)

gen_demo_cfg = dict(
    dataset=single_infer_dataset,
    sample_idx=[0],
    save_path=demo_root,
    data_saver=data_saver,
)


infer_cfg = dict(
    model=hbir_infer_model,
    input_path=f"./demo/{task_name}",
    gen_inputs_cfg=dict(
        dataset=single_infer_dataset,
        sample_idx=[0, 3],
        inputs_save_func=inputs_save_func,
    ),
    prepare_inputs=prepare_inputs,
    process_inputs=process_inputs,
    viz_func=dict(type="NuscenesMultitaskViz", is_plot=True),
    process_outputs=process_outputs,
)


# ---------------------------------------------------------------------------
# tools/inference.py stage configs ({stage}_infer), reusing the helpers above.
# inference.py builds an Inference object and feeds it frames via
# data_loader -> pre_processor -> model -> post_processor -> visualize.
# ---------------------------------------------------------------------------
def data_loader(data_path):
    """List frame dirs under ``data_path`` (gen_demo output: sample_idx_*).

    Each entry feeds one frame through ``pre_processor -> model -> viz``.
    ``data_path`` may be the task demo root (``demo/<task>`` holding
    ``sample_idx_*`` dirs) or a single ``sample_idx_*`` dir directly.
    """
    if not data_path or not str(data_path).strip():
        raise ValueError("`data_path` cannot be empty.")
    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            "Demo directory not found: %s. "
            "Run `python3 tools/gen_demo.py -c configs/%s` first."
            % (data_path, task_name)
        )
    # If data_path is itself a frame dir (has img*.jpg), treat as single frame.
    if any(
        f.endswith(".jpg")
        for f in os.listdir(data_path)
        if os.path.isfile(os.path.join(data_path, f))
    ):
        frames = [""]
    else:
        frames = sorted(
            f
            for f in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, f))
        )
        if not frames:
            raise FileNotFoundError(
                "No frame dirs or .jpg images under %s. "
                "Run `python3 tools/gen_demo.py -c configs/%s` first."
                % (data_path, task_name)
            )
    return [
        {
            "data_path": os.path.join(data_path, f),
            "save_path": os.path.join(data_path, f) if f else data_path,
        }
        for f in frames
    ]


def pre_processor(data, transforms=None):
    """Build model_input for one frame dir; stash viz artifacts on data."""
    frame_path = data["data_path"]
    image_dir_list = sorted(
        f for f in os.listdir(frame_path) if f.endswith(".jpg")
    )
    if not image_dir_list:
        raise FileNotFoundError(
            "No .jpg images under %s. Run tools/gen_demo.py first."
            % frame_path
        )
    imgs = [os.path.join(frame_path, p) for p in image_dir_list]
    sample = {
        "imgs": imgs,
        "timestamp": np.load(
            os.path.join(frame_path, "timestamp.npy")
        ).reshape(1),
        "lidar2global": np.load(
            os.path.join(frame_path, "lidar2global.npy")
        ).reshape(1, 4, 4),
        "lidar2img": np.load(os.path.join(frame_path, "lidar2img.npy")),
        "points": np.load(os.path.join(frame_path, "lidar.npy")),
    }
    model_input, vis_inputs = process_inputs(sample, transforms=None)
    model_input["vis_imgs"] = vis_inputs["img"]
    model_input["vis_points"] = vis_inputs["points"]
    model_input["vis_meta"] = vis_inputs["meta"]
    if "save_path" in data:
        model_input["save_path"] = data["save_path"]
    if "stage" in data:
        model_input["stage"] = data["stage"]
    return model_input


def post_processor(pred, data):
    """Identity; viz consumes pred + data["vis_*"] directly."""
    return pred, data


@OBJECT_REGISTRY.register
class NuscenesLidarDetInferViz(object):
    """Adapter from :class:`Inference` to :class:`NuscenesMultitaskViz`."""

    def __init__(
        self,
        score_thresh=0.4,
        is_plot=True,
        bev_size=None,
        use_bce=False,
        save_path=None,
    ):
        self.viz = NuscenesMultitaskViz(
            score_thresh=score_thresh,
            is_plot=is_plot,
            bev_size=bev_size,
            use_bce=use_bce,
        )
        self.save_path = save_path

    def __call__(self, pred, data):
        def _to_cpu(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu()
            return x

        outs = torch.cat(
            [
                pred[0]["bboxes"][..., :9].view(1, -1, 9),
                pred[0]["scores"].view(1, -1, 1),
                pred[0]["labels"].view(1, -1, 1),
            ],
            dim=-1,
        )
        outs = outs.detach().cpu()
        outs[..., 3], outs[..., 4] = outs[..., 4], outs[..., 3]
        # to_cuda in Inference.forward may have moved viz artifacts to GPU;
        # pull them back to host before matplotlib rendering.
        vis_imgs = [
            {"name": i["name"], "img": _to_cpu(i["img"])}
            for i in data["vis_imgs"]
        ]
        vis_points = [_to_cpu(p) for p in data["vis_points"]]
        vis_meta = _to_cpu(data["vis_meta"])
        self.viz(
            vis_imgs,
            vis_points,
            {"lidar_det": outs},
            vis_meta,
            save_path=self.save_path or data.get("save_path"),
            stage=data.get("stage"),
        )


_infer_common = dict(
    data_path=f"./demo/{task_name}",
    data_loader=data_loader,
    transforms=None,
    infer=dict(
        type="Inference",
        pre_processors=pre_processor,
        post_processors=post_processor,
        visualize_processors=dict(
            type="NuscenesLidarDetInferViz",
            score_thresh=0.3,
            is_plot=True,
            save_path=demo_root,
        ),
    ),
)

float_infer = copy.deepcopy(_infer_common)
float_infer["infer"]["model"] = train_model
float_infer["infer"]["march"] = march
float_infer["infer"]["model_convert_pipeline"] = dict(
    type="ModelConvertPipeline",
    converters=[
        dict(
            type="LoadCheckpoint",
            checkpoint_path=float_checkpoint,
            load_ema_model=True,
            allow_miss=True,
            ignore_extra=True,
        ),
    ],
)

# QAT inference head: disable denoising (DN) like predict.py's qat_model.
qat_model = copy.deepcopy(train_model)
qat_model["head"]["enable_dn"] = False

# calibration: Float2QAT (val) + calibration checkpoint.
calibration_infer = copy.deepcopy(_infer_common)
calibration_infer["infer"]["model"] = train_model
calibration_infer["infer"]["march"] = march
calibration_infer["infer"][
    "model_convert_pipeline"
] = calibration_predict_model_convert_pipeline

# qat: Float2QAT (val) + qat checkpoint, DN disabled.
qat_infer = copy.deepcopy(_infer_common)
qat_infer["infer"]["model"] = qat_model
qat_infer["infer"]["march"] = march
qat_infer["infer"]["model_convert_pipeline"] = qat_val_model_convert_pipeline

# hbir: run the compiled IR model (quantized.bc); no convert pipeline needed
# since the model (SparseBEVFusionIrInfer) already wraps the HbirModule.
hbir_infer = copy.deepcopy(_infer_common)
hbir_infer["infer"]["model"] = hbir_infer_model
hbir_infer["infer"]["march"] = march
hbir_infer["infer"].pop("model_convert_pipeline", None)

# hbm: run the on-board BPU runtime (compile/model.hbm) after compile_perf.
hbm_infer = copy.deepcopy(_infer_common)
hbm_infer["infer"]["model"] = hbm_infer_model
hbm_infer["infer"]["march"] = march
hbm_infer["infer"].pop("model_convert_pipeline", None)
