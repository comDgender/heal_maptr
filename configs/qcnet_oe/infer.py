import copy
import os
import pickle

from common import (
    ckpt_dir,
    num_historical_steps,
    ori_future_sec,
    ori_historical_sec,
    ori_sample_fre,
    qat_mode,
    sample_fre,
    task_name,
)
from data import int_infer_data_loader
from hat_engine.registry import OBJECT_REGISTRY
from model import cali_model, deploy_inputs, hbir_infer_model
from qconfig import float2qat

from heal.data.collates.qc_collate import collate_qc_argoverse2
from heal.visualize.argoverse2 import Argoverse2Viz


class _RawDataHolder:
    """Carry the raw (un-collated) sample through ``Inference.forward``.

    ``Inference.forward`` moves the model input to cuda via ``to_cuda`` before
    the model runs. Wrapping the raw sample keeps it a pytree leaf so its
    tensors stay on cpu for matplotlib-based visualization.
    """

    def __init__(self, data):
        self.data = data


def data_loader(data_path):
    """Load every ``qcnet_data.pkl`` saved under ``data_path``."""
    outputs = []
    for root, _, file_names in os.walk(data_path):
        for file_name in file_names:
            if file_name == "qcnet_data.pkl":
                with open(os.path.join(root, file_name), "rb") as f:
                    outputs.append(pickle.load(f))
    return outputs


def pre_processor(data, transforms=None):
    model_input = collate_qc_argoverse2(
        [data],
        ori_historical_sec=ori_historical_sec,
        ori_future_sec=ori_future_sec,
        ori_sample_fre=ori_sample_fre,
        sample_fre=sample_fre,
        stage="val",
        agent_num=30,
        pl_N=80,
        pt_N=50,
        pt_N_downsample_nums=50,
    )
    if transforms is not None:
        model_input = transforms(model_input)
    model_input["vis_inputs"] = _RawDataHolder(data)
    return model_input


def post_processor(model_outs, data):
    holder = data.get("vis_inputs", None) if isinstance(data, dict) else None
    raw_data = holder.data if isinstance(holder, _RawDataHolder) else None
    return model_outs, raw_data


def data_saver(data, save_path):
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "qcnet_data.pkl"), "wb") as f:
        pickle.dump(data, f)


single_infer_dataset = copy.deepcopy(int_infer_data_loader["dataset"])
single_infer_dataset["transforms"] = None

gen_demo_cfg = dict(
    dataset=single_infer_dataset,
    sample_idx=[100],
    save_path=f"./demo/{task_name}",
    data_saver=data_saver,
)


@OBJECT_REGISTRY.register
class QCNetViz(object):
    """Inference visualizer adapting :class:`Argoverse2Viz`."""

    def __init__(self, num_historical_steps, save_path=None):
        self.num_historical_steps = num_historical_steps
        self.save_path = save_path
        self._viz = None

    def __call__(self, pred, data):
        if data is None:
            return
        if self._viz is None:
            self._viz = Argoverse2Viz(
                num_historical_steps=self.num_historical_steps
            )
        self._viz(data, pred, map_path=None, save_path=self.save_path)


float_infer = dict(
    data_path=f"./demo/{task_name}",
    data_loader=data_loader,
    transforms=None,
    infer=dict(
        type="Inference",
        model=cali_model,
        model_convert_pipeline=dict(
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
        ),
        pre_processors=pre_processor,
        post_processors=post_processor,
        visualize_processors=dict(
            type="QCNetViz",
            num_historical_steps=num_historical_steps,
            save_path=f"./tmp_viz_results/{task_name}_float/",
        ),
    ),
)

calibration_infer = copy.deepcopy(float_infer)
calibration_infer["infer"]["model_convert_pipeline"] = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    qconfig_params=dict(
        activation_calibration_observer="mse",
    ),
    converters=[
        float2qat,
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
calibration_infer["infer"]["visualize_processors"][
    "save_path"
] = f"./tmp_viz_results/{task_name}_calibration/"

qat_infer = copy.deepcopy(float_infer)
qat_infer["infer"]["model_convert_pipeline"] = dict(
    type="ModelConvertPipeline",
    qat_mode=qat_mode,
    converters=[
        float2qat,
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
qat_infer["infer"]["visualize_processors"][
    "save_path"
] = f"./tmp_viz_results/{task_name}_qat/"

hbir_infer = copy.deepcopy(float_infer)
hbir_infer["infer"]["model"] = hbir_infer_model
hbir_infer["infer"].pop("model_convert_pipeline", None)
hbir_infer["infer"]["visualize_processors"][
    "save_path"
] = f"./tmp_viz_results/{task_name}_hbir/"

# QCNetOEHbmInfer wraps an HbmModule (.hbm) and reuses the streaming-infer
# preprocess/postprocess of the hbir model. Requires board/hbm_infer runtime.
hbm_infer_model = copy.deepcopy(hbir_infer_model)
hbm_infer_model["type"] = "QCNetOEHbmInfer"
hbm_infer_model["ir_model"] = dict(
    type="HbmModule",
    model_path=os.path.join(ckpt_dir, "compile", "model.hbm"),
    host="10.35.72.2",
)
hbm_infer_model["example_data"] = deploy_inputs

hbm_infer = copy.deepcopy(float_infer)
hbm_infer["infer"]["model"] = hbm_infer_model
hbm_infer["infer"].pop("model_convert_pipeline", None)
hbm_infer["infer"]["visualize_processors"][
    "save_path"
] = f"./tmp_viz_results/{task_name}_hbm/"
