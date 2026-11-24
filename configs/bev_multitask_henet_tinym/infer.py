# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import copy
import logging
import os
import shutil

import cv2
import numpy as np
import torch
from PIL import Image

try:
    from torchvision.transforms.functional_tensor import resize
except ImportError:
    # torchvision 0.18
    from torchvision.transforms._functional_tensor import resize

from common import bev_size, ckpt_dir, map_point_cloud_range, march, task_name
from data import val_data_loader
from hat_engine.registry import OBJECT_REGISTRY
from model import hbir_infer_model, hbm_infer_model, test_model
from qconfig import calibration_predict_model_convert_pipeline
from torchvision.transforms.functional import pil_to_tensor

from heal.data.transforms.functional_img import image_pad
from heal.visualize.nuscenes_map import NuscenesMapViz
from heal.visualize.occ import OccViz

try:
    import horizon_plugin_pytorch as horizon
except ImportError:
    horizon = None


logger = logging.getLogger(__name__)


def resize_homo(homo, scale):
    view = np.eye(4)
    view[0, 0] = scale[1]
    view[1, 1] = scale[0]
    homo = view @ homo
    return homo


def process_img(img_path, resize_size, pad_divisor):
    orig_img = cv2.imread(img_path)
    cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB, orig_img)
    orig_img = Image.fromarray(orig_img)
    orig_img = pil_to_tensor(orig_img)
    resize_hw = (
        int(resize_size[0]),
        int(resize_size[1]),
    )

    orig_shape = (orig_img.shape[1], orig_img.shape[2])
    resized_img = resize(orig_img, resize_hw)
    resized_img = image_pad(
        resized_img, "chw", None, pad_divisor, 0
    ).unsqueeze(0)
    return resized_img, orig_shape


def prepare_inputs(infer_inputs):
    """Collect per-frame demo inputs from a sample directory.

    Each sample dir holds one ``frame{idx}`` sub-dir per temporal frame,
    containing ``img{i}_*.jpg`` plus ``lidar2img.npy`` / ``lidar2global.npy`` /
    ``timestamp.npy``.
    """
    dir_list = os.listdir(infer_inputs)
    dir_list.sort()
    input_datas = []
    for _, frame in enumerate(dir_list):
        frame_path = os.path.join(infer_inputs, frame)
        if not os.path.isdir(frame_path):
            continue
        data = {}
        file_list = list(os.listdir(frame_path))
        image_dir_list = list(filter(lambda x: x.endswith(".jpg"), file_list))
        image_dir_list.sort()

        data["imgs"] = [
            os.path.join(frame_path, tmpdir) for tmpdir in image_dir_list
        ]
        timestamp_path = os.path.join(frame_path, "timestamp.npy")
        data["timestamp"] = np.load(timestamp_path).reshape(1)
        lidar2global_path = os.path.join(frame_path, "lidar2global.npy")
        data["lidar2global"] = np.load(lidar2global_path).reshape(1, 4, 4)
        lidar2img_path = os.path.join(frame_path, "lidar2img.npy")
        data["lidar2img"] = np.load(lidar2img_path)
        input_datas.append(data)
    return input_datas


def process_inputs(infer_inputs, transforms=None, stage=None):
    pad_divisor = 32
    resize_size = (512, 960)
    orig_shape = (900, 1600)
    orig_imgs = []
    for i, img_path in enumerate(infer_inputs["imgs"]):
        img, orig_shape = process_img(img_path, resize_size, pad_divisor)
        orig_imgs.append({"name": i, "img": img})

    # HBM (use_nv12): BevMultitaskIrInfer.img2nv12 expects raw RGB uint8.
    if stage == "hbm":
        input_imgs = torch.cat([item["img"] for item in orig_imgs])
        if input_imgs.dtype != torch.uint8:
            input_imgs = input_imgs.to(torch.uint8)
    else:
        if horizon is None:
            raise ImportError(
                "horizon_plugin_pytorch is required for bev-multitask infer."
            )
        input_imgs = []
        for orig_img in orig_imgs:
            input_imgs.append(
                horizon.nn.functional.bgr_to_yuv444(orig_img["img"], True)
            )
        input_imgs = (torch.cat(input_imgs) - 128.0) / 128.0

    lidar2img = infer_inputs["lidar2img"]
    scale = (resize_size[0] / orig_shape[0], resize_size[1] / orig_shape[1])
    lidar2img = resize_homo(lidar2img, scale)

    model_input = {
        "img": input_imgs,
        "lidar2img": torch.tensor(lidar2img),
        "lidar2global": torch.tensor(infer_inputs["lidar2global"]),
        "timestamp": torch.tensor(infer_inputs["timestamp"]),
        "seq_meta": [
            {
                "lidar2img": [
                    [np.array(lidar2img[i]) for i in range(len(lidar2img))]
                ]
            }
        ],
    }
    if transforms is not None:
        model_input = transforms(model_input)

    vis_inputs = {}
    vis_inputs["img"] = orig_imgs
    vis_inputs["meta"] = {
        "lidar2img": copy.deepcopy(lidar2img),
    }

    return model_input, vis_inputs


def convert_outputs(model_outs):
    """Convert multi-task model outputs into the viz ``preds`` layout."""
    preds = {}

    if "dynamic_objects_det" in model_outs:
        outs = torch.cat(
            [
                model_outs["dynamic_objects_det"][0]["bboxes"][..., :9].view(
                    1, -1, 9
                ),
                model_outs["dynamic_objects_det"][0]["scores"].view(1, -1, 1),
                model_outs["dynamic_objects_det"][0]["labels"].view(1, -1, 1),
            ],
            dim=-1,
        )
        outs[..., 3], outs[..., 4] = outs[..., 4], outs[..., 3]
        preds["dynamic_objects_det"] = outs
    if "static_objects_det" in model_outs:
        outs = torch.cat(
            [
                model_outs["static_objects_det"][0]["bboxes"][..., :9].view(
                    1, -1, 9
                ),
                model_outs["static_objects_det"][0]["scores"].view(1, -1, 1),
                model_outs["static_objects_det"][0]["labels"].view(1, -1, 1),
            ],
            dim=-1,
        )
        outs[..., 3], outs[..., 4] = outs[..., 4], outs[..., 3]
        preds["static_objects_det"] = outs

    if "occ" in model_outs:
        occ_out = model_outs["occ"]
        if isinstance(occ_out, (list, tuple)):
            occ_out = occ_out[0]
        if isinstance(occ_out, torch.Tensor):
            occ_out = occ_out.squeeze().detach().cpu()
            semantics_pred = occ_out.numpy().astype(np.uint8)
        else:
            semantics_pred = np.asarray(occ_out).astype(np.uint8)
        preds["occ_det"] = semantics_pred
    if "maptr" in model_outs:
        preds["om_det"] = model_outs["maptr"]
    if "discrete_map" in model_outs:
        outs = model_outs["discrete_map"][0]
        new_outs = torch.zeros(
            (outs.shape[0], 11), device=outs.device, dtype=outs.dtype
        )
        new_outs[:, :2] = outs[:, :2]
        new_outs[:, 3] = outs[:, 3]
        new_outs[:, 4] = outs[:, 2]
        new_outs[:, 6] = outs[:, 4]
        new_outs[:, -2] = outs[:, -2]
        new_outs[:, -1] = outs[:, -1]
        preds["discrete_map_det"] = [new_outs]

    return preds


def _vis_inputs_from_data(data):
    imgs = []
    for item in data["vis_imgs"]:
        img = item["img"]
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
        imgs.append({"name": item["name"], "img": img})
    meta = copy.deepcopy(data["vis_meta"])
    homo = meta.get("lidar2img", meta.get("ego2img"))
    if isinstance(homo, torch.Tensor):
        homo = homo.detach().cpu().numpy()
    if homo is not None:
        meta["lidar2img"] = homo
    return imgs, meta


# --- data_saver (gen_demo): saves a single dataset sample to disk ----------


def _to_pil_image(img_data):
    """Convert PIL / Tensor / ndarray to a saveable ``PIL.Image``."""
    if isinstance(img_data, Image.Image):
        return img_data
    if isinstance(img_data, torch.Tensor):
        img_data = img_data.detach().cpu().numpy()
    arr = np.asarray(img_data)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        # Normalized / float tensors: map back to uint8 for demo dump.
        if np.issubdtype(arr.dtype, np.floating):
            a_min, a_max = float(arr.min()), float(arr.max())
            if a_max <= 1.5 and a_min >= -1.5:
                arr = (arr * 128.0 + 128.0).clip(0, 255)
            elif a_max <= 1.0 and a_min >= 0.0:
                arr = (arr * 255.0).clip(0, 255)
            else:
                arr = arr.clip(0, 255)
        arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(arr, mode="RGB")


def inputs_save_func(sample, save_path):
    """Save a single ``dataset[idx]`` sample as demo ``frame*`` dirs.

    ``tools/gen_demo.py`` calls ``data_saver(sample, sample_path)`` per index.
    NuScenes multitask returns ``img`` as a list of 6 camera images (not
    temporal frames). Temporal packs are nested lists-of-lists.
    """
    if os.path.isdir(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)

    imgs = sample["img"]
    names = sample["img_name"]
    # Temporal: [[cam0, ...], [cam0, ...], ...]; single frame: [cam0, ...]
    if imgs and isinstance(imgs[0], (list, tuple)):
        frames = imgs
        frame_names = names
    else:
        frames = [imgs]
        frame_names = [names]

    for frame_idx, (img_data_list, img_name_list) in enumerate(
        zip(frames, frame_names)
    ):
        save_dir = os.path.join(save_path, f"frame{frame_idx}")
        os.makedirs(save_dir, exist_ok=True)
        if not isinstance(img_data_list, (list, tuple)):
            img_data_list = [img_data_list]
            img_name_list = [img_name_list]
        for image_idx, (img_name, img_data) in enumerate(
            zip(img_name_list, img_data_list)
        ):
            save_name = f"img{image_idx}_{os.path.basename(str(img_name))}"
            _to_pil_image(img_data).save(
                os.path.join(save_dir, save_name), "JPEG"
            )

        def _frame_value(key, default=None):
            val = sample.get(key, default)
            if (
                isinstance(val, (list, tuple))
                and len(val) == len(frames)
                and frame_idx < len(val)
            ):
                return val[frame_idx]
            return val

        np.save(
            os.path.join(save_dir, "lidar2global.npy"),
            np.array(_frame_value("lidar2global")),
        )
        np.save(
            os.path.join(save_dir, "lidar2img.npy"),
            np.array(_frame_value("lidar2img")),
        )
        np.save(
            os.path.join(save_dir, "timestamp.npy"),
            np.array(_frame_value("timestamp")),
        )


# --- demo sample discovery + per-sample wrapping ---------------------------


def resolve_demo_sample_dir(data_path):
    """Resolve the demo dir that holds ``frame{idx}`` sub-dirs."""
    if os.path.isdir(os.path.join(data_path, "frame0")):
        return data_path
    default_dir = os.path.join(data_path, "sample_idx_0")
    if os.path.isdir(os.path.join(default_dir, "frame0")):
        return default_dir
    if os.path.isdir(data_path):
        for name in sorted(os.listdir(data_path)):
            sub_dir = os.path.join(data_path, name)
            if os.path.isdir(os.path.join(sub_dir, "frame0")):
                return sub_dir
    return data_path


def data_loader(data_path):
    sample_dir = resolve_demo_sample_dir(data_path)
    if not os.path.isdir(sample_dir):
        raise FileNotFoundError("Demo directory not found: %s" % sample_dir)
    frames = prepare_inputs(sample_dir)
    if not frames:
        raise FileNotFoundError(
            "No frame* sub-dirs under %s. Run tools/gen_demo.py first."
            % sample_dir
        )
    return [
        {
            "frame_input": frame,
            "save_path": sample_dir,
            "frame_idx": idx,
        }
        for idx, frame in enumerate(frames)
    ]


# --- Inference three-stage adapters ----------------------------------------


def pre_processor(data, transforms=None):
    model_input, vis_inputs = process_inputs(
        data["frame_input"],
        transforms=transforms,
        stage=data.get("stage"),
    )
    model_input["vis_imgs"] = vis_inputs["img"]
    model_input["vis_meta"] = vis_inputs["meta"]
    model_input["save_path"] = data.get("save_path")
    model_input["frame_idx"] = data.get("frame_idx")
    if "stage" in data:
        model_input["stage"] = data["stage"]
    return model_input


def post_processor(pred, data):
    return pred, data


@OBJECT_REGISTRY.register
class NuscenesMultitaskInferViz(object):
    """Adapter from :class:`Inference` to :class:`MultitaskViz`."""

    def __init__(
        self,
        bev_size=bev_size,
        is_plot=True,
        use_bce=False,
        score_thresh=0.1,
        num_pics=11,
        car_img_path=(
            "tmp_orig_data/nuscenes/infer_imgs/single_frame_lidarego/"
            "cars/lidar_car.png"
        ),
        save_path=None,
    ):
        from heal.visualize.perception_multitask import MultitaskViz

        self.viz = MultitaskViz(
            occ_viz=OccViz(
                vcs_range=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
                vis_bev_2d=True,
            ),
            om_viz=NuscenesMapViz(
                pc_range=map_point_cloud_range,
                car_img_path=car_img_path,
            ),
            bev_size=bev_size,
            score_thresh=score_thresh,
            is_plot=is_plot,
            use_bce=use_bce,
            num_pics=num_pics,
        )
        self.save_path = save_path

    def __call__(self, pred, data):
        preds = convert_outputs(pred)
        imgs, meta = _vis_inputs_from_data(data)
        stage = data.get("stage")
        save_path = data.get("save_path") or self.save_path
        sample_idx = self.viz.sample_idx
        self.viz(
            imgs, preds=preds, meta=meta, save_path=save_path, stage=stage
        )
        if save_path:
            out_png = os.path.join(
                save_path,
                self.viz.result_filename(sample_idx, stage),
            )
            logger.info("Visualization saved to %s", out_png)


# --- config assembly --------------------------------------------------------

# Drop all image transforms so gen_demo gets raw PIL frames (not YUV tensors).
single_infer_dataset = copy.deepcopy(val_data_loader["dataset"])
single_infer_dataset["transforms"] = None
if "datasets" in single_infer_dataset:
    for _ds in single_infer_dataset["datasets"]:
        _ds["transforms"] = None

demo_root = os.path.join(".", "demo", task_name)

gen_demo_cfg = dict(
    dataset=single_infer_dataset,
    sample_idx=[0, 2],
    save_path=demo_root,
    data_saver=inputs_save_func,
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
            type="NuscenesMultitaskInferViz",
            is_plot=True,
            bev_size=bev_size,
            use_bce=False,
            score_thresh=0.1,
            num_pics=11,
            save_path=demo_root,
        ),
    ),
)

# float: float_stage3 best ckpt
float_infer = copy.deepcopy(_infer_common)
float_infer["infer"]["model"] = test_model
float_infer["infer"]["march"] = march
float_infer["infer"]["model_convert_pipeline"] = dict(
    type="ModelConvertPipeline",
    converters=[
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "float_stage3-checkpoint-best.pth.tar"
            ),
            allow_miss=True,
            ignore_extra=True,
            verbose=True,
        ),
    ],
)

# calibration: Float2Calibration + calibration-checkpoint-last
calibration_infer = copy.deepcopy(_infer_common)
calibration_infer["infer"]["model"] = test_model
calibration_infer["infer"]["march"] = march
calibration_infer["infer"][
    "model_convert_pipeline"
] = calibration_predict_model_convert_pipeline

# hbir: quantized_infer.bc via BevMultitaskIrInfer (no convert pipeline)
hbir_infer = copy.deepcopy(_infer_common)
hbir_infer["infer"]["model"] = hbir_infer_model
hbir_infer["infer"]["march"] = march
hbir_infer["infer"].pop("model_convert_pipeline", None)

# hbm: compile/model.hbm on board (use_nv12 → raw RGB in pre_processor)
hbm_infer = copy.deepcopy(_infer_common)
hbm_infer["infer"]["model"] = hbm_infer_model
hbm_infer["infer"]["march"] = march
hbm_infer["infer"].pop("model_convert_pipeline", None)
