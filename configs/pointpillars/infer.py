import copy
import os
import pickle

import numpy as np
import torch
from common import base_data_dir, ckpt_dir, class_names, task_name
from data import single_infer_dataset
from hat_engine.registry import OBJECT_REGISTRY, build_from_registry
from model import deploy_infer_model, example_inputs, hbir_model, hbm_model
from qconfig import calib_qconfig_setter, qat_qconfig_setter
from tqdm import tqdm

from heal.visualize.lidar_det import lidar_det_visualize


def data_loader(data_path):
    input_paths = []
    for root, _, file_names in os.walk(data_path):
        for file_name in file_names:
            if file_name == "points.npy":
                input_paths.append(os.path.join(root, file_name))

    outputs = []
    for path in input_paths:
        points = np.load(path).reshape((-1, 4))
        data = {
            "lidar": {"points": points},
            "vis_inputs": points,  # raw numpy array for visualization
        }
        outputs.append(data)

    return outputs


def pre_processor(data, transforms=None):
    points = torch.from_numpy(data["lidar"]["points"])
    model_input = {
        "points": [points],
    }

    if transforms is not None:
        model_input = transforms(model_input)

    model_input["vis_inputs"] = data["vis_inputs"]
    return model_input


def post_processor(model_outs, data):
    vis_inputs = data.pop("vis_inputs")
    data["vis_inputs"] = vis_inputs
    preds = model_outs[0]
    data["preds"] = preds
    return model_outs, data


def data_saver(data, save_path):
    points_path = os.path.join(save_path, "points.npy")
    np.save(points_path, data["lidar"]["points"])


gen_demo_cfg = dict(
    dataset=single_infer_dataset,
    sample_idx=[0],
    save_path=f"./demo/{task_name}",
    data_saver=data_saver,
)


@OBJECT_REGISTRY.register
class PointPillarsViz(object):
    def __init__(self, save_path=None):
        self.save_path = save_path

    def __call__(self, model_outs, data):
        vis_inputs = data["vis_inputs"]
        preds = data["preds"]
        lidar_det_visualize(
            vis_inputs,
            preds,
            score_thresh=0.4,
            is_plot=True,
            save_path=self.save_path,
        )


def update_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[f"ir_model.{k}"] = v
    return new_state_dict


float_infer = dict(
    data_path=f"./demo/{task_name}",
    data_loader=data_loader,
    transforms=None,
    infer=dict(
        type="Inference",
        model=deploy_infer_model,
        model_convert_pipeline=dict(
            type="ModelConvertPipeline",
            converters=[
                dict(
                    type="LoadCheckpoint",
                    checkpoint_path=os.path.join(
                        ckpt_dir, "float-checkpoint-best.pth.tar"
                    ),
                    state_dict_update_func=update_state_dict,
                    ignore_extra=True,
                    allow_miss=True,
                ),
            ],
        ),
        pre_processors=pre_processor,
        post_processors=post_processor,
        visualize_processors=dict(
            type="PointPillarsViz",
            save_path="./tmp_viz_results/pointpillars_hbir/",
        ),
    ),
)

calibration_infer = copy.deepcopy(float_infer)
calibration_infer["infer"]["model_convert_pipeline"] = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(
            type="Float2Calibration",
            convert_mode="jit-strip",
            example_inputs=example_inputs,
            qconfig_setter=calib_qconfig_setter,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "calibration-checkpoint-best.pth.tar"
            ),
            state_dict_update_func=update_state_dict,
            verbose=True,
            ignore_extra=True,
            allow_miss=True,
        ),
    ],
)

qat_infer = copy.deepcopy(float_infer)
qat_infer["infer"]["model_convert_pipeline"] = dict(
    type="ModelConvertPipeline",
    qat_mode="fuse_bn",
    converters=[
        dict(
            type="Float2QAT",
            convert_mode="jit-strip",
            example_inputs=example_inputs,
            qconfig_setter=qat_qconfig_setter,
        ),
        dict(
            type="LoadCheckpoint",
            checkpoint_path=os.path.join(
                ckpt_dir, "qat-checkpoint-best.pth.tar"
            ),
            state_dict_update_func=update_state_dict,
            verbose=True,
            allow_miss=True,
            ignore_extra=True,
        ),
    ],
)

hbir_infer = copy.deepcopy(float_infer)
hbir_infer["infer"]["model"] = hbir_model
hbir_infer["infer"].pop("model_convert_pipeline", None)

hbm_infer = copy.deepcopy(float_infer)
hbm_infer["infer"]["model"] = hbm_model
hbm_infer["infer"].pop("model_convert_pipeline", None)

# --- bpu_val_preprocess ---
height = 1
width = 150000

bpu_val_dataset = dict(
    dict(
        type="Kitti3D",
        data_path=os.path.join(base_data_dir, "val_lmdb"),
        transforms=[
            dict(type="LidarReformat"),
        ],
    ),
)


def bpu_preprocess_data(val_dataset, save_path):
    velodyne_save_path = os.path.join(save_path, "reduced_velodyne")
    gt_root = os.path.join(save_path, "gt")
    lst_root = os.path.join(save_path, "lst")
    os.makedirs(gt_root, exist_ok=True)
    os.makedirs(lst_root, exist_ok=True)
    os.makedirs(velodyne_save_path, exist_ok=True)

    norm_save_path = os.path.normpath(save_path)
    save_path_prefixes = norm_save_path.split(os.sep)
    lst_path = os.path.join(lst_root, "kitti3d.lst")
    gt_path = os.path.join(gt_root, "val_gt_infos.pkl")
    padding_np = (
        (np.ones(width * 4) * -100).astype(np.float32).reshape(1, 1, -1, 4)
    )
    keystr = "_" + str(height) + "_" + str(width)
    gt_info = {}
    with open(lst_path, "w") as f:
        for i, sample in enumerate(val_dataset):
            if (i + 1) % 1000 == 0:
                print("{}th finished!".format(i + 1))
            idx = sample["metadata"]["image_idx"]
            reduced_velodyne = sample["points"].reshape(1, 1, -1, 4)
            padding_bin = np.concatenate(
                (reduced_velodyne, padding_np), axis=2
            )
            target_bin = padding_bin[:, :, :width, :]
            bin_name = "{:06d}{}{}.bin".format(idx, keystr, keystr)
            save_file = os.path.join(velodyne_save_path, bin_name)
            lst_bin_path = os.path.join(
                f"../../../data/{save_path_prefixes[-2]}/{task_name}/reduced_velodyne",  # noqa
                bin_name,
            )
            f.write(lst_bin_path + "\n")
            with open(save_file, "w") as fb:
                target_bin.tofile(fb)
            gt_info[str(idx)] = {
                "metadata": sample["metadata"],
                "calib": sample["calib"],
            }
    with open(gt_path, "wb") as handle:
        pickle.dump(gt_info, handle, protocol=pickle.HIGHEST_PROTOCOL)


# --- bpu_eval_metric ---
def bpu_process_result_data(pred_path):
    if isinstance(pred_path, list):
        pred_path = pred_path[0]
    lines = open(pred_path).read().splitlines()
    all_preds_data = {}
    for line in lines:
        data = line.split(": ")
        image_idx = data[0].split("_")[0]
        preds = data[1].split("; ")
        box3d_lidar_list = []
        scores_list = []
        label_label_list = []
        for i in range(len(preds) - 1):
            pred = preds[i].split(" ")
            box3d_lidar_list.append(pred[:7])
            scores_list.append(pred[7])
            label_label_list.append(pred[8])
        preds_data = {
            "image_idx": int(image_idx),
            "box3d_lidar": np.array(
                box3d_lidar_list, dtype=np.float32
            ).reshape(-1, 7),
            "label_preds": np.array(label_label_list, dtype=np.int8),
            "scores": np.array(scores_list, dtype=np.float32),
        }
        dict_key = str(image_idx)
        all_preds_data[dict_key] = preds_data
    return all_preds_data


bpu_eval_metric = dict(
    type="Kitti3DMetricDet",
    compute_aos=True,
    current_classes=class_names,
    difficultys=[0, 1, 2],
)


def bpu_process_metric(result_data, eval_metric, gt_path):  # noqa
    if isinstance(eval_metric, dict):
        eval_metric = build_from_registry(eval_metric)
    with open(gt_path, "rb") as f:
        gt_info = pickle.load(f)

    for idx, pred_info in tqdm(result_data.items()):
        # import pdb;pdb.set_trace()
        pred = [
            (
                pred_info["box3d_lidar"],
                pred_info["label_preds"],
                pred_info["scores"],
            )
        ]
        label = gt_info[str(int(idx))]
        label["metadata"] = [label["metadata"]]
        for k in label["calib"]:
            label["calib"][k] = [label["calib"][k]]
        eval_metric.update(pred, label)
    d3_names, d3_values = eval_metric.get()
    for k, v in zip(d3_names, d3_values):
        print("{}[{:.2f}]".format(str(k), v))
