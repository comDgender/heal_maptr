"""Local inference configs for BevFusion multisensor multitask."""

import copy
import os
import re
import shutil

import cv2
import horizon_plugin_pytorch as horizon
import numpy as np
import torch
from common import bev_size, ckpt_dir, march, task_name
from data import single_infer_dataset
from hat_engine.registry import OBJECT_REGISTRY
from model import hbir_infer_model, hbm_infer_model
from model import model as train_model
from PIL import Image
from qconfig import calibration_predict_model_convert_pipeline
from torchvision.transforms.functional import pil_to_tensor

try:
    from torchvision.transforms.functional_tensor import resize
except ImportError:
    from torchvision.transforms._functional_tensor import resize

from heal.data.transforms.functional_img import image_pad

resize_shape = (3, 512, 960)
val_data_shape = (3, 512, 960)
orig_shape = (3, 900, 1600)


def resize_homo(homo, scale):
    view = np.eye(4)
    view[0, 0] = scale[1]
    view[1, 1] = scale[0]
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


def prepare_inputs(infer_inputs):
    file_list = list(os.listdir(infer_inputs))
    image_dir_list = list(filter(lambda x: x.endswith(".jpg"), file_list))

    def extract_key(filename):
        match = re.search(r"img(\d+)_", filename)
        if match:
            return int(match.group(1))
        return float("inf")

    image_dir_list.sort(key=extract_key)
    ego2img = np.load(os.path.join(infer_inputs, "ego2img.npy"))
    lidar2img = np.load(os.path.join(infer_inputs, "lidar2img.npy"))
    frame_inputs = {}
    num_cam = 6
    frame_inputs["img_paths"] = [
        os.path.join(infer_inputs, p) for p in image_dir_list
    ]
    frame_inputs["ego2img"] = ego2img[:num_cam]
    frame_inputs["lidar2img"] = lidar2img[:num_cam]
    frame_inputs["points"] = os.path.join(infer_inputs, "lidar.npy")
    return [frame_inputs]


def process_inputs(infer_inputs, transforms=None):
    pad_divisor = 32
    resize_size = resize_shape[1:]
    orig_imgs = []
    for i, img_path in enumerate(infer_inputs["img_paths"]):
        img, orig_shape = process_img(img_path, resize_size, pad_divisor)
        orig_imgs.append({"name": i, "img": img})

    input_imgs = []
    for orig_img in orig_imgs:
        input_img = horizon.nn.functional.bgr_to_yuv444(orig_img["img"], True)
        input_imgs.append(input_img)
    input_imgs = torch.cat(input_imgs)
    input_imgs = (input_imgs - 128.0) / 128.0

    lidar2img = infer_inputs["lidar2img"]
    ego2img = infer_inputs["ego2img"]
    points = [torch.tensor(np.load(infer_inputs["points"]))]

    scale = (resize_size[0] / orig_shape[0], resize_size[1] / orig_shape[1])
    lidar2img = resize_homo(lidar2img, scale)
    ego2img = resize_homo(ego2img, scale)
    model_input = {
        "img": input_imgs,
        "points": points,
        "seq_meta": [
            {
                "meta": [{"scene": "test_infer"}],
                "lidar2img": [lidar2img],
                "ego2img": [ego2img],
            }
        ],
    }
    if transforms is not None:
        model_input = transforms(model_input)

    vis_inputs = {
        "img": orig_imgs,
        "points": points,
        "meta": {
            "ego2img": ego2img,
            "lidar2img": copy.deepcopy(lidar2img),
        },
    }
    return model_input, vis_inputs


def hbm_process_inputs(infer_inputs, transforms=None):
    """Build model_input for the HBM (BPU, use_nv12) path.

    Unlike ``process_inputs`` (float/hbir), the HBM NV12 path must NOT convert
    to YUV444 or normalize: BevFusionHbirInfer.nv12_runtime expects raw RGB
    uint8 images (NCHW, [0,255]) and converts them to NV12 internally. Only the
    img handling differs; points / seq_meta are built the same way.
    """
    pad_divisor = 32
    resize_size = resize_shape[1:]
    orig_imgs = []
    input_imgs = []
    for i, img_path in enumerate(infer_inputs["img_paths"]):
        img, orig_shape = process_img(img_path, resize_size, pad_divisor)
        orig_imgs.append({"name": i, "img": img})
        input_imgs.append(img)
    input_imgs = torch.cat(input_imgs)  # NCHW uint8 RGB, no yuv / no normalize

    lidar2img = infer_inputs["lidar2img"]
    ego2img = infer_inputs["ego2img"]
    points = [torch.tensor(np.load(infer_inputs["points"]))]

    scale = (resize_size[0] / orig_shape[0], resize_size[1] / orig_shape[1])
    lidar2img = resize_homo(lidar2img, scale)
    ego2img = resize_homo(ego2img, scale)
    model_input = {
        "img": input_imgs,
        "points": points,
        "seq_meta": [
            {
                "meta": [{"scene": "test_infer"}],
                "lidar2img": [lidar2img],
                "ego2img": [ego2img],
            }
        ],
    }
    if transforms is not None:
        model_input = transforms(model_input)

    vis_inputs = {
        "img": orig_imgs,
        "points": points,
        "meta": {
            "ego2img": ego2img,
            "lidar2img": copy.deepcopy(lidar2img),
        },
    }
    return model_input, vis_inputs


def process_outputs(model_outs, viz_func, vis_inputs):
    semantics_pred = model_outs[1][0].squeeze().numpy().astype(np.uint8)
    preds = {"bev_det": model_outs[0], "occ_det": semantics_pred}
    viz_func(
        vis_inputs["img"], vis_inputs["points"], preds, vis_inputs["meta"]
    )
    return None


single_hbir_infer_model = copy.deepcopy(hbir_infer_model)
single_hbir_infer_model["deploy_model"]["bev_decoders"][0]["post_process"][
    "score_threshold"
] = 0.1


def inputs_save_func(data, save_path):
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    if "seq_meta" in data:
        frame = data["seq_meta"][0]
        img_names = frame["img_name"]
        if img_names and isinstance(img_names[0], (list, tuple)):
            img_names = img_names[0]
        imgs = frame["img"]
        if imgs and hasattr(imgs[0], "save"):
            img_items = imgs
        else:
            img_items = imgs[0] if imgs else []
        ego2img = frame["ego2img"]
        lidar2img = frame["lidar2img"]
        if isinstance(ego2img[0], (list, tuple, np.ndarray)):
            ego2img = ego2img[0]
        if isinstance(lidar2img[0], (list, tuple, np.ndarray)):
            lidar2img = lidar2img[0]
        points = data["points"][0]
    else:
        img_names = data[0]["img_name"]
        img_items = data[0]["img"]
        ego2img = data[0]["ego2img"]
        lidar2img = data[0]["lidar2img"]
        points = data[0]["points"]

    for image_idx, (img_name, img_data) in enumerate(
        zip(img_names, img_items)
    ):
        save_name = f"img{image_idx}_{os.path.basename(img_name)}"
        img_data.save(os.path.join(save_path, save_name), "JPEG")
    np.save(os.path.join(save_path, "ego2img.npy"), np.array(ego2img))
    np.save(os.path.join(save_path, "lidar2img.npy"), lidar2img)
    np.save(os.path.join(save_path, "lidar.npy"), points)


# Saver used by tools/gen_demo.py. ``single_infer_dataset`` (a MergeDataset
# with transforms=None) returns one flat merged sample dict per index, so
# save its 6 multi-view images + lidar/nuscenes meta straight into
# ``sample_path`` (``demo/<task>/sample_idx_<i>``). The resulting dir
# matches what ``prepare_inputs`` expects (one frame dir = img*.jpg + npy).
def gen_demo_data_saver(sample, save_path):
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    img_names = sample["img_name"]
    img_items = sample["img"]
    ego2img = sample["ego2img"]
    lidar2img = sample["lidar2img"]
    if "points" in sample:
        points = sample["points"]
    elif "lidar" in sample and "points" in sample["lidar"]:
        points = sample["lidar"]["points"]
    else:
        points = sample.get("lidar", {}).get("points")

    for image_idx, (img_name, img_data) in enumerate(
        zip(img_names, img_items)
    ):
        save_name = f"img{image_idx}_{os.path.basename(img_name)}"
        img_data.save(os.path.join(save_path, save_name), "JPEG")
    np.save(os.path.join(save_path, "ego2img.npy"), np.array(ego2img))
    np.save(os.path.join(save_path, "lidar2img.npy"), np.array(lidar2img))
    np.save(os.path.join(save_path, "lidar.npy"), points)


demo_root = os.path.join(".", "demo", task_name)

gen_demo_cfg = dict(
    dataset=single_infer_dataset,
    sample_idx=[0],
    save_path=demo_root,
    data_saver=gen_demo_data_saver,
)

infer_cfg = dict(
    model=single_hbir_infer_model,
    input_path=demo_root,
    gen_inputs_cfg=dict(
        dataset=single_infer_dataset,
        sample_idx=[0],
        inputs_save_func=inputs_save_func,
    ),
    process_inputs=process_inputs,
    viz_func=dict(
        type="NuscenesMultitaskViz",
        occ_viz=dict(
            type="OccViz",
            vcs_range=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
            vis_bev_2d=True,
        ),
        score_thresh=0.3,
        is_plot=True,
        bev_size=bev_size,
        use_bce=False,
    ),
    process_outputs=process_outputs,
    prepare_inputs=prepare_inputs,
)


# ---------------------------------------------------------------------------
# tools/inference.py stage configs ({stage}_infer). inference.py builds an
# Inference object and feeds each frame (from data_loader) through
# pre_processor -> model -> post_processor -> visualize_processor. Each frame
# dir (gen_demo output: demo/<task>/sample_idx_<i>/) holds img*.jpg +
# ego2img.npy + lidar2img.npy + lidar.npy. The model returns
# (bev_results, [occ_res]); the viz adapter builds NuscenesMultitaskViz preds
# (bev_det + occ_det) from it.
# Run: python3 tools/gen_demo.py -c configs/<this> then
#      python3 tools/inference.py -c configs/<this> \
#        -s {float|calibration|hbir|hbm}
# ---------------------------------------------------------------------------
def data_loader(data_path):
    """List frame dirs under ``data_path`` (gen_demo output: sample_idx_*).

    gen_demo writes demo/<task>/sample_idx_<i>/img*.jpg + ego2img/lidar2img/
    lidar.npy directly (no frame0 nesting). Each sample_idx_* feeds one frame.
    """
    if not data_path or not str(data_path).strip():
        raise ValueError("`data_path` cannot be empty.")
    if not os.path.isdir(data_path):
        raise FileNotFoundError(
            "Demo directory not found: %s. "
            "Run `python3 tools/gen_demo.py -c configs/%s` first."
            % (data_path, task_name)
        )

    def _has_jpg(path):
        return any(
            f.endswith(".jpg")
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        )

    # data_path itself is a frame dir.
    if _has_jpg(data_path):
        return [{"data_path": data_path, "save_path": data_path}]

    frames = sorted(
        f
        for f in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, f))
        and _has_jpg(os.path.join(data_path, f))
    )
    if not frames:
        raise FileNotFoundError(
            "No frame dirs with .jpg images under %s. "
            "Run `python3 tools/gen_demo.py -c configs/%s` first."
            % (data_path, task_name)
        )
    return [
        {
            "data_path": os.path.join(data_path, f),
            "save_path": os.path.join(data_path, f),
        }
        for f in frames
    ]


def pre_processor(data, transforms=None):
    """Build model_input for one frame dir; stash viz artifacts on data."""
    frame_inputs = prepare_inputs(data["data_path"])[0]
    model_input, vis_inputs = process_inputs(frame_inputs, transforms=None)
    model_input["vis_imgs"] = vis_inputs["img"]
    model_input["vis_points"] = vis_inputs["points"]
    model_input["vis_meta"] = vis_inputs["meta"]
    if "save_path" in data:
        model_input["save_path"] = data["save_path"]
    if "stage" in data:
        model_input["stage"] = data["stage"]
    return model_input


def hbm_pre_processor(data, transforms=None):
    """Build model_input for the HBM (BPU, use_nv12) path: raw RGB uint8 img
    (no YUV444 / no normalize) — BevFusionHbirInfer.nv12_runtime converts to
    NV12 internally."""
    frame_inputs = prepare_inputs(data["data_path"])[0]
    model_input, vis_inputs = hbm_process_inputs(frame_inputs, transforms=None)
    model_input["vis_imgs"] = vis_inputs["img"]
    model_input["vis_points"] = vis_inputs["points"]
    model_input["vis_meta"] = vis_inputs["meta"]
    if "save_path" in data:
        model_input["save_path"] = data["save_path"]
    if "stage" in data:
        model_input["stage"] = data["stage"]
    return model_input


def post_processor(pred, data):
    """Identity; the viz processor consumes pred + data['vis_*'] directly."""
    return pred, data


@OBJECT_REGISTRY.register
class BevFusionMultitaskInferViz(object):
    """Adapter from :class:`Inference` to :class:`NuscenesMultitaskViz`.

    The model returns ``(bev_results, [occ_res])`` (det predictions_list +
    occ argmax); build the bev_det/occ_det preds dict the same way
    ``process_outputs`` does.
    """

    def __init__(self, score_thresh=0.3, is_plot=True, save_path=None):
        from heal.visualize.nuscenes import NuscenesMultitaskViz
        from heal.visualize.occ import OccViz

        self.viz = NuscenesMultitaskViz(
            occ_viz=OccViz(
                vcs_range=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
                vis_bev_2d=True,
            ),
            score_thresh=score_thresh,
            is_plot=is_plot,
            bev_size=bev_size,
            use_bce=False,
        )
        self.save_path = save_path

    def __call__(self, pred, data):
        def _to_cpu(x):
            return x.detach().cpu() if isinstance(x, torch.Tensor) else x

        model_outs = pred
        semantics_pred = (
            model_outs[1][0].squeeze().detach().cpu().numpy().astype(np.uint8)
        )
        preds = {"bev_det": model_outs[0], "occ_det": semantics_pred}
        vis_imgs = [
            {"name": i["name"], "img": _to_cpu(i["img"])}
            for i in data["vis_imgs"]
        ]
        vis_points = [_to_cpu(p) for p in data["vis_points"]]
        vis_meta = _to_cpu(data["vis_meta"])
        self.viz(
            vis_imgs,
            vis_points,
            preds,
            vis_meta,
            save_path=self.save_path or data.get("save_path"),
            stage=data.get("stage"),
        )


_infer_common = dict(
    data_path=demo_root,
    data_loader=data_loader,
    transforms=None,
    infer=dict(
        type="Inference",
        march=march,
        pre_processors=pre_processor,
        post_processors=post_processor,
        visualize_processors=dict(
            type="BevFusionMultitaskInferViz",
            score_thresh=0.3,
            is_plot=True,
            save_path=demo_root,
        ),
    ),
)

# float inference: float BevFusion model + float-checkpoint-last.
float_infer = copy.deepcopy(_infer_common)
float_infer["infer"]["model"] = train_model
float_infer["infer"]["model_convert_pipeline"] = dict(
    type="ModelConvertPipeline",
    converters=[
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "float-checkpoint-last.pth.tar"
            ),
            allow_miss=True,
            ignore_extra=True,
            verbose=True,
        ),
    ],
)

# calibration inference: float model converted to calibration quantized graph
# (RepModel2Deploy + Float2QAT) + calibration-checkpoint-last.
calibration_infer = copy.deepcopy(_infer_common)
calibration_infer["infer"]["model"] = train_model
calibration_infer["infer"][
    "model_convert_pipeline"
] = calibration_predict_model_convert_pipeline

# hbir inference: compiled IR model (quantized.bc). BevFusionHbirInfer wraps
# the HbirModule + deploy_model; no convert pipeline needed.
hbir_infer = copy.deepcopy(_infer_common)
hbir_infer["infer"]["model"] = hbir_infer_model
hbir_infer["infer"].pop("model_convert_pipeline", None)

# hbm inference: on-board BPU runtime (compile/model.hbm) after compile_perf.
# BevFusionHbirInfer accepts use_nv12 (stored) and ir_model (HbmModule here).
# Override `host` with your board IP (see model.py hbm_infer_model).
hbm_infer = copy.deepcopy(_infer_common)
hbm_infer["infer"]["model"] = hbm_infer_model
hbm_infer["infer"].pop("model_convert_pipeline", None)
# HBM (use_nv12) needs raw RGB uint8 img (no YUV444 / no normalize);
# BevFusionHbirInfer.nv12_runtime converts to NV12 internally.
hbm_infer["infer"]["pre_processors"] = hbm_pre_processor
