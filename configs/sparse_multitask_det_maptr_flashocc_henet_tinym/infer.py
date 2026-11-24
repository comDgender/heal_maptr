"""Inference and demo for sparse multitask det + MapTR + FlashOcc.

Two entry points share this config:

* ``tools/gen_demo.py -c configs/<this>`` — dumps raw demo samples
  (6 multi-view images + ego2img/lidar2img + lidar points) from
  ``single_infer_dataset`` into ``demo/<task>/sample_idx_<i>/`` via
  ``gen_demo_cfg``.
* ``tools/inference.py -c configs/<this> -s {float|calibration|hbir|hbm}``
  — runs one stage's model over the demo frames and visualizes
  det + occ + om.

The IR/HBM model is :class:`SparseMultitaskOEIrInfer` (alias of
``SparseMultitaskIrInfer``). On the first frame it only needs ``img`` +
``lidar2img`` (``gen_projection_mat`` reads ``metas["img"]`` shapes and
``metas["lidar2img"]``; ``export_reference_points`` reads the same two
keys). Camera-only inference does not feed lidar points to the model —
points are saved/loaded only for the BEV point-cloud subplot in the
visualization.
"""

import copy
import logging
import os
import shutil

import cv2
import numpy as np
import torch
from common import (
    bev_h_,
    bev_w_,
    map_point_cloud_range,
    march,
    resize_shape,
    task_name,
    use_lidar_gt,
    val_data_shape,
)
from data import single_infer_dataset
from hat_engine.registry import OBJECT_REGISTRY
from model import hbir_infer_model, hbm_infer_model
from model import model as train_model
from PIL import Image
from qconfig import (
    calibration_predict_model_convert_pipeline,
    float_merged_predict_model_convert_pipeline,
)
from torchvision.transforms.functional import pil_to_tensor

logger = logging.getLogger(__name__)

try:
    import horizon_plugin_pytorch as horizon
except ImportError:
    horizon = None

try:
    from torchvision.transforms.functional_tensor import resize
except ImportError:
    from torchvision.transforms._functional_tensor import resize


# ---------------------------------------------------------------------------
# Image / homography preprocessing (resize + center-crop, matching
# val_transforms: MultiViewsImgResize(resize_shape[1:]) ->
# MultiViewsImgCrop(val_data_shape[1:])).
# ---------------------------------------------------------------------------
def process_img(img_path, resize_size, crop_size):
    orig_img = cv2.imread(img_path)
    cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB, orig_img)
    orig_img = Image.fromarray(orig_img)
    orig_img = pil_to_tensor(orig_img)
    resize_hw = (int(resize_size[0]), int(resize_size[1]))
    orig_shape = (orig_img.shape[1], orig_img.shape[2])  # (H, W)
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


# ---------------------------------------------------------------------------
# gen_demo.py: save one merged sample
# (img/img_name/ego2img/lidar2img + points) into
# demo/<task>/sample_idx_<i>/ as img*.jpg + *.npy.
# ``single_infer_dataset`` is a MergeDataset with transforms=None, so it
# yields a flat dict per index.
# ---------------------------------------------------------------------------
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
    if points is not None:
        np.save(os.path.join(save_path, "lidar.npy"), points)


demo_root = os.path.join(".", "demo", task_name)
demo_sample_dir = os.path.join(demo_root, "sample_idx_0")

gen_demo_cfg = dict(
    dataset=single_infer_dataset,
    sample_idx=[0],
    save_path=demo_root,
    data_saver=gen_demo_data_saver,
)


# ---------------------------------------------------------------------------
# tools/inference.py: load a demo frame dir back into raw inputs.
# ---------------------------------------------------------------------------
def prepare_inputs(frame_path):
    file_list = list(os.listdir(frame_path))
    image_dir_list = sorted(
        name for name in file_list if name.endswith(".jpg")
    )
    if not image_dir_list:
        raise FileNotFoundError("No .jpg images under %s" % frame_path)
    lidar2img_path = os.path.join(frame_path, "lidar2img.npy")
    ego2img_path = os.path.join(frame_path, "ego2img.npy")
    lidar_path = os.path.join(frame_path, "lidar.npy")
    data = {
        "img_paths": [
            os.path.join(frame_path, name) for name in image_dir_list
        ],
        "lidar2img": np.load(lidar2img_path),
        "ego2img": (
            np.load(ego2img_path) if os.path.isfile(ego2img_path) else None
        ),
        "points": np.load(lidar_path) if os.path.isfile(lidar_path) else None,
    }
    return data


def process_inputs(data, transforms=None, stage=None):
    """Build ``model_input`` (img + lidar2img) + ``vis_inputs`` for one frame.

    Float/hbir/HBM all feed ``SparseMultitaskOEIrInfer`` the same camera-only
    metas (img + lidar2img); only the image color-space handling differs:
    float/hbir use YUV444 + normalize, HBM (use_nv12) uses raw RGB uint8.
    """
    resize_size = resize_shape[1:]
    input_size = val_data_shape[1:]
    orig_imgs = []
    for i, img_path in enumerate(data["img_paths"]):
        img, orig_shape = process_img(img_path, resize_size, input_size)
        orig_imgs.append({"name": i, "img": img})

    if stage == "hbm":
        # Raw RGB uint8 (NCHW); the BPU pyramid input_source does YUV/NV12
        # on-chip, so do NOT convert to YUV444 or normalize here.
        input_imgs = torch.cat([item["img"] for item in orig_imgs])
    else:
        if horizon is None:
            raise ImportError(
                "horizon_plugin_pytorch is required for float/hbir sparse "
                "multitask infer."
            )
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
    }
    if transforms is not None:
        model_input = transforms(model_input)

    vis_inputs = {
        "img": orig_imgs,
        "points": (
            [torch.as_tensor(data["points"])]
            if data.get("points") is not None
            else None
        ),
        "meta": {"lidar2img": copy.deepcopy(homo)},
    }
    return model_input, vis_inputs


def resolve_demo_sample_dir(data_path):
    """Resolve gen_demo dir (``sample_idx_*``) containing ``img*.jpg``."""
    if any(
        f.endswith(".jpg")
        for f in os.listdir(data_path)
        if os.path.isfile(os.path.join(data_path, f))
    ):
        return data_path
    default_dir = os.path.join(data_path, "sample_idx_0")
    if os.path.isdir(default_dir) and os.path.isfile(
        os.path.join(default_dir, "lidar2img.npy")
    ):
        return default_dir
    if os.path.isdir(data_path):
        for name in sorted(os.listdir(data_path)):
            sub_dir = os.path.join(data_path, name)
            if os.path.isdir(sub_dir) and os.path.isfile(
                os.path.join(sub_dir, "lidar2img.npy")
            ):
                return sub_dir
    return data_path


def data_loader(data_path):
    if not data_path or not str(data_path).strip():
        raise ValueError("`data_path` cannot be empty.")
    sample_dir = resolve_demo_sample_dir(data_path)
    if not os.path.isdir(sample_dir):
        raise FileNotFoundError(
            "Demo directory not found: %s. "
            "Run `python3 tools/gen_demo.py -c configs/%s` first."
            % (sample_dir, task_name)
        )
    if not os.path.isfile(os.path.join(sample_dir, "lidar2img.npy")):
        raise FileNotFoundError(
            "No lidar2img.npy under %s. Run tools/gen_demo.py first."
            % sample_dir
        )
    return [{"frame_path": sample_dir, "save_path": sample_dir}]


def pre_processor(data, transforms=None):
    raw = prepare_inputs(data["frame_path"])
    model_input, vis_inputs = process_inputs(
        raw, transforms=transforms, stage=data.get("stage")
    )
    model_input["vis_imgs"] = vis_inputs["img"]
    model_input["vis_points"] = vis_inputs["points"]
    model_input["vis_meta"] = vis_inputs["meta"]
    if "save_path" in data:
        model_input["save_path"] = data["save_path"]
    if "stage" in data:
        model_input["stage"] = data["stage"]
    return model_input


def hbm_pre_processor(data, transforms=None):
    # HBM (use_nv12) differs from float/hbir only in image color-space, which
    # process_inputs already selects on data["stage"] == "hbm"; keep this thin
    # wrapper as an explicit HBM hook (mirrors the BevFusion sibling config).
    return pre_processor(data, transforms=transforms)


def post_processor(pred, data):
    """Normalize model output to ``{"det", "occ", "om"}`` for viz.

    - IR/HBM (:class:`SparseMultitaskOEIrInfer`) already returns this
      dict, with ``occ`` a 1-element list wrapping the decoded
      ``(B, Dx, Dy, Dz)`` class-index tensor (same contract as float).
    - Float (:class:`SparseMultitaskOE`, eval) returns ``det``/``om`` as
      decoded prediction lists and ``occ`` as ``[occ_pre]``.
    Either way, pass the structured dict straight to the visualizer.
    """
    if isinstance(pred, dict) and {"det", "occ", "om"} <= set(pred.keys()):
        return pred, data
    raise ValueError(
        "Unexpected model output for sparse multitask infer: %r" % (pred,)
    )


def _occ_to_numpy(occ):
    """Extract the decoded occ voxel grid (HxWxD int) from either output shape.

    The occ head already argmaxes over classes; the returned tensor is a class
    index per voxel with a leading batch dim, so just drop it (and the list
    wrapper from the float path) and cast to uint8. ``OccViz.vis_occ`` reads
    ``semantics.shape = (H, W, D)`` and collapses the depth axis itself.
    """
    if isinstance(occ, (list, tuple)):
        occ = occ[0]
    if isinstance(occ, torch.Tensor):
        occ = occ.detach().cpu()
    arr = np.asarray(occ)
    return arr.squeeze().astype(np.uint8)


def _det_to_viz_array(det_results):
    """Reformat det decoder output for :class:`NuscenesMultitaskViz`.

    The det decoder returns ``[{bboxes, scores, labels}]`` with ``bboxes`` as
    ``(N, C)`` where the leading 9 cols are
    ``[cx, cy, cz, w, l, h, yaw, vx, vy]`` (the decoder catenates
    xyz + exp(wlh) + yaw + vel; observed C is 9 or 10 depending on the
    trailing extra fields). ``NuscenesViz`` consumes ``preds["bev_det"][0]``
    as an ``(N, 11)`` array whose layout ``bbox_to_corner`` reads is
    ``[cx, cy, cz, w, l, h, yaw, vx, vy, score, label]`` (``bbox[3:6]``=wlh,
    ``bbox[6]``=yaw, ``bbox[9]``=score, ``bbox[10]``=label; cols 7:9 are
    unused by the viz). Take the first 9 decoded cols + score + label and
    wrap in a one-element list so the viz's ``preds["bev_det"][0]`` indexing
    yields the (N, 11) array.
    """
    if not det_results:
        return []
    det = det_results[0]
    if isinstance(det, dict) and "bboxes" in det:
        bboxes = det["bboxes"]
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.numpy()
        scores = det["scores"]
        if isinstance(scores, torch.Tensor):
            scores = scores.numpy()
        labels = det["labels"]
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        bboxes = np.asarray(bboxes)
        if bboxes.ndim == 2:
            keep = min(bboxes.shape[1], 9)
            bboxes9 = bboxes[:, :keep]
            if keep < 9:  # pad vel cols with zeros if under-filled
                bboxes9 = np.concatenate(
                    [
                        bboxes9,
                        np.zeros(
                            (bboxes.shape[0], 9 - keep), dtype=bboxes.dtype
                        ),
                    ],
                    axis=1,
                )
            scores = np.asarray(scores).reshape(-1, 1)
            labels = np.asarray(labels).reshape(-1, 1)
            return [np.concatenate([bboxes9, scores, labels], axis=1)]
    # Fall back to the original list-of-dicts if the layout is unexpected.
    return det_results


@OBJECT_REGISTRY.register
class SparseMultitaskInferViz(object):
    """Adapter from :class:`Inference` to :class:`NuscenesMultitaskViz`.

    Builds the ``{bev_det, occ_det, om_det}`` preds dict that
    ``NuscenesMultitaskViz.__call__`` consumes from the model's
    ``{det, occ, om}`` output, and passes the saved lidar points + lidar2img
    meta for the BEV point / cam-projection subplots. ``om_viz`` (a
    :class:`NuscenesMapViz`) is required to draw the map lines.
    """

    def __init__(
        self,
        score_thresh=0.3,
        is_plot=True,
        save_path=None,
    ):
        from heal.visualize.nuscenes import NuscenesMultitaskViz
        from heal.visualize.nuscenes_map import NuscenesMapViz
        from heal.visualize.occ import OccViz

        self.viz = NuscenesMultitaskViz(
            occ_viz=OccViz(
                vcs_range=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
                vis_bev_2d=True,
            ),
            om_viz=NuscenesMapViz(
                pc_range=map_point_cloud_range,
                is_plot=is_plot,
                use_lidar=use_lidar_gt,
            ),
            bev_size=(bev_h_, bev_w_, 0.625),
            score_thresh=score_thresh,
            is_plot=is_plot,
            use_bce=False,
        )
        self.save_path = save_path

    def __call__(self, pred, data):
        def _to_cpu(x):
            return x.detach().cpu() if isinstance(x, torch.Tensor) else x

        preds = {
            "bev_det": _det_to_viz_array(pred["det"]),
            "occ_det": _occ_to_numpy(pred["occ"]),
            "om_det": pred["om"],
        }
        vis_imgs = [
            {"name": i["name"], "img": _to_cpu(i["img"])}
            for i in data["vis_imgs"]
        ]
        vis_points = (
            [_to_cpu(p) for p in data["vis_points"]]
            if data.get("vis_points")
            else None
        )
        vis_meta = _to_cpu(data["vis_meta"])
        base_dir = data.get("save_path") or self.save_path or demo_sample_dir
        self.viz(
            vis_imgs,
            points=vis_points,
            preds=preds,
            meta=vis_meta,
            save_path=base_dir,
            stage=data.get("stage"),
        )
        logger.info("Sparse multitask visualization saved under %s", base_dir)


# ---------------------------------------------------------------------------
# tools/inference.py stage configs ({stage}_infer). inference.py builds an
# Inference object and feeds each frame (from data_loader) through
# pre_processor -> model -> post_processor -> visualize_processor.
# ---------------------------------------------------------------------------
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
            type="SparseMultitaskInferViz",
            score_thresh=0.3,
            is_plot=True,
            save_path=demo_sample_dir,
        ),
    ),
)

# float inference: full float SparseMultitask model (det + om + occ)
# loaded from the merged checkpoint (stage1 det + stage2 om/occ). The
# float model's eval forward already decodes det/om and returns occ as
# [occ_pre]; post_processor normalizes the dict shape for the visualizer.
float_infer = copy.deepcopy(_infer_common)
float_infer["infer"]["model"] = train_model
float_infer["infer"][
    "model_convert_pipeline"
] = float_merged_predict_model_convert_pipeline

# calibration inference: float model converted to calibration quantized graph
# (RepModel2Deploy + Float2QAT) + calibration-checkpoint-last.
calibration_infer = copy.deepcopy(_infer_common)
calibration_infer["infer"]["model"] = train_model
calibration_infer["infer"][
    "model_convert_pipeline"
] = calibration_predict_model_convert_pipeline

# hbir inference: compiled IR model (quantized.bc). SparseMultitaskOEIrInfer
# wraps the HbirModule + deploy_model; no convert pipeline needed.
hbir_infer = copy.deepcopy(_infer_common)
hbir_infer["infer"]["model"] = hbir_infer_model
hbir_infer["infer"].pop("model_convert_pipeline", None)

# hbm inference: on-board BPU runtime (compile/model.hbm) after compile_perf.
# use_nv12 needs raw RGB uint8 img (no YUV444 / no normalize); the BPU pyramid
# input_source converts to NV12 on-chip. Override `host` with your board IP
# (see model.py hbm_infer_model).
hbm_infer = copy.deepcopy(_infer_common)
hbm_infer["infer"]["model"] = hbm_infer_model
hbm_infer["infer"].pop("model_convert_pipeline", None)
hbm_infer["infer"]["pre_processors"] = hbm_pre_processor


# Standalone infer_cfg kept for parity with sibling configs that expose a
# ``process_inputs`` / ``viz_func`` / ``prepare_inputs`` entry (used by some
# custom demo scripts); not consumed by tools/inference.py's stage path.
infer_cfg = dict(
    model=hbir_infer_model,
    input_path=demo_root,
    gen_inputs_cfg=gen_demo_cfg,
    prepare_inputs=prepare_inputs,
    process_inputs=process_inputs,
    viz_func=dict(
        type="SparseMultitaskInferViz",
        score_thresh=0.3,
        is_plot=True,
        save_path=demo_sample_dir,
    ),
)
