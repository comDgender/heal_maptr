"""Local inference configs."""

import copy
import logging
import os

import numpy as np
import torch
from common import ckpt_dir, convert_mode, march, qat_mode, task_name
from data import single_infer_dataset
from hat_engine.registry import OBJECT_REGISTRY
from model import deploy_inputs, example_inputs, hbir_model, hbm_model, model
from qconfig import calib_qconfig_setter, qat_qconfig_setter

from heal.visualize.lidar_det import lidar_det_visualize

logger = logging.getLogger(__name__)


def data_loader(data_path):
    input_paths = []
    for root, _, file_names in os.walk(data_path):
        for file_name in file_names:
            if file_name == "points.npy":
                input_paths.append(os.path.join(root, file_name))

    outputs = []
    for path in sorted(input_paths):
        points = np.load(path).reshape((-1, 5))
        outputs.append(
            {
                "lidar": {"points": points},
                "vis_inputs": points,
                "points_path": path,
                "save_path": os.path.dirname(path),
            }
        )
    return outputs


def pre_processor(data, transforms=None):
    points = data["lidar"]["points"]
    if not isinstance(points, torch.Tensor):
        points = torch.from_numpy(np.asarray(points))
    model_input = {"points": [points.float()]}
    if transforms is not None:
        model_input = transforms(model_input)
    model_input["vis_inputs"] = data["vis_inputs"]
    model_input["save_path"] = data.get("save_path")
    if "stage" in data:
        model_input["stage"] = data["stage"]
    if "points_path" in data:
        model_input["points_path"] = data["points_path"]
    return model_input


def post_processor(pred, data):
    if isinstance(pred, list) and len(pred) > 0:
        preds = pred[0]
    else:
        preds = pred
    return {"lidar_det": preds}, data


def inputs_save_func(data, save_path):
    if os.path.isdir(save_path):
        import shutil

        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, "points.npy"), data["lidar"]["points"])


demo_root = os.path.join(".", "demo", task_name)

gen_demo_cfg = dict(
    dataset=single_infer_dataset,
    sample_idx=[0],
    save_path=demo_root,
    data_saver=inputs_save_func,
)


@OBJECT_REGISTRY.register
class CenterPointInferViz(object):
    """Adapter from :class:`Inference` to :class:`lidar_det_visualize`."""

    def __init__(
        self,
        score_thresh=0.4,
        is_plot=True,
        reverse=True,
        save_path=None,
    ):
        self.score_thresh = score_thresh
        self.is_plot = is_plot
        self.reverse = reverse
        self.save_path = save_path

    def __call__(self, pred, data):
        preds = (
            pred
            if isinstance(pred, dict) and "lidar_det" in pred
            else {"lidar_det": pred}
        )
        det = preds["lidar_det"]
        if isinstance(det, list) and len(det) > 0:
            det = det[0]
        vis_inputs = data.get("vis_inputs")
        save_path = data.get("save_path") or self.save_path
        stage = data.get("stage")
        save_name = "lidar_pred_%s.png" % stage if stage else "lidar_pred.png"
        lidar_det_visualize(
            vis_inputs,
            det,
            reverse=self.reverse,
            score_thresh=self.score_thresh,
            is_plot=self.is_plot,
            save_path=save_path,
            save_name=save_name,
        )
        if save_path:
            out_png = os.path.join(save_path, save_name)
            logger.info("Visualization saved to %s", out_png)


_infer_common = dict(
    data_path=demo_root,
    data_loader=data_loader,
    transforms=None,
    infer=dict(
        type="Inference",
        pre_processors=pre_processor,
        post_processors=post_processor,
        visualize_processors=dict(
            type="CenterPointInferViz",
            score_thresh=0.4,
            is_plot=True,
            reverse=True,
            save_path=demo_root,
        ),
    ),
)

float_infer = copy.deepcopy(_infer_common)
float_infer["infer"]["model"] = model
float_infer["infer"]["march"] = march
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
        ),
    ],
)

calibration_infer = copy.deepcopy(_infer_common)
calibration_infer["infer"]["model"] = model
calibration_infer["infer"]["march"] = march
calibration_infer["infer"]["model_convert_pipeline"] = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(
            type="Float2Calibration",
            convert_mode=convert_mode,
            example_inputs=example_inputs,
            qconfig_setter=calib_qconfig_setter,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "calibration-checkpoint-last.pth.tar"
            ),
            allow_miss=True,
            ignore_extra=True,
            verbose=True,
        ),
    ],
)

qat_infer = copy.deepcopy(_infer_common)
qat_infer["infer"]["model"] = model
qat_infer["infer"]["march"] = march
qat_infer["infer"]["model_convert_pipeline"] = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        dict(
            type="Float2QAT",
            convert_mode=convert_mode,
            example_inputs=deploy_inputs,
            qconfig_setter=qat_qconfig_setter,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "qat-checkpoint-last.pth.tar"
            ),
            allow_miss=True,
            ignore_extra=True,
            verbose=True,
        ),
    ],
)

hbir_infer = copy.deepcopy(_infer_common)
hbir_infer["infer"]["model"] = hbir_model
hbir_infer["infer"]["march"] = march
hbir_infer["infer"].pop("model_convert_pipeline", None)

hbm_infer = copy.deepcopy(_infer_common)
hbm_infer["infer"]["model"] = hbm_model
hbm_infer["infer"]["march"] = march
hbm_infer["infer"].pop("model_convert_pipeline", None)
