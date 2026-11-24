# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import copy
import json
import logging
import os
import pickle
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union

import msgpack_numpy
import numpy as np
import torch
from hat_engine.metrics.metric import EvalMetric
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.distributed import (
    all_gather_object,
    get_dist_info,
    rank_zero_only,
)
from hat_engine.utils.package_helper import require_packages

from heal.core.nms.box3d_nms import nms_bev_multiclass
from heal.utils.perception_to_planner import (
    perception_dyn_to_planner_agents,
    perception_maptr_to_planner_lanes,
    perception_static_to_planner_static,
)

try:
    import pyquaternion
    from nuscenes.eval.common.data_classes import EvalBox, EvalBoxes
    from nuscenes.eval.common.utils import Quaternion
    from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
    from nuscenes.eval.detection.constants import TP_METRICS
    from nuscenes.eval.detection.data_classes import (
        DetectionBox,
        DetectionConfig,
        DetectionMetricDataList,
        DetectionMetrics,
    )
    from nuscenes.utils.data_classes import Box as NuScenesBox
except ImportError as _det_exc:
    pyquaternion = None
    Quaternion = None
    NuScenesBox = None
    DetectionConfig = None
    DetectionMetricDataList = None
    TP_METRICS = None
    DetectionBox = None
    DetectionMetrics = None
    EvalBoxes = None
    EvalBox = None
    accumulate = None
    calc_ap = None
    calc_tp = None
    _DET_IMPORT_ERROR = _det_exc
else:
    _DET_IMPORT_ERROR = None

try:
    from nuscenes.eval.tracking.algo import TrackingEvaluation
    from nuscenes.eval.tracking.constants import AVG_METRIC_MAP, MOT_METRIC_MAP
    from nuscenes.eval.tracking.data_classes import (
        TrackingBox,
        TrackingConfig,
        TrackingMetricData,
        TrackingMetricDataList,
        TrackingMetrics,
    )
except ImportError as _trk_exc:
    TrackingBox = None
    TrackingConfig = None
    TrackingMetricDataList = None
    TrackingMetricData = None
    TrackingMetrics = None
    TrackingEvaluation = None
    AVG_METRIC_MAP = None
    MOT_METRIC_MAP = None
    _TRK_IMPORT_ERROR = _trk_exc
else:
    _TRK_IMPORT_ERROR = None

msgpack_numpy.patch()

logger = logging.getLogger(__name__)
if _DET_IMPORT_ERROR is not None:
    logger.warning(
        "nuscenes detection eval import failed: %s", _DET_IMPORT_ERROR
    )
if _TRK_IMPORT_ERROR is not None:
    logger.warning(
        "nuscenes tracking eval import failed (optional): %s",
        _TRK_IMPORT_ERROR,
    )

__all__ = [
    "NavsimPerceptionMetric",
    "NavsimPerceptionTrackMetric",
]


CLASSES = (
    "vehicle",
    "pedestrian",
    "bicycle",
    "czone_sign",
    "barrier",
    "traffic_cone",
    "generic_object",
)

DefaultAttribute = {
    "car": "vehicle.parked",
    "pedestrian": "pedestrian.moving",
    "trailer": "vehicle.parked",
    "truck": "vehicle.parked",
    "bus": "vehicle.moving",
    "motorcycle": "cycle.without_rider",
    "construction_vehicle": "vehicle.parked",
    "bicycle": "cycle.without_rider",
    "vehicle": "vehicle.parked",
    "barrier": "",
    "traffic_cone": "",
    "czone_sign": "",
    "generic_object": "",
}


def _to_tensor(arr, dtype=torch.float32):
    """Convert input data to a tensor of the target dtype.

    Args:
        arr: Input tensor, numpy array, or list.
        dtype: Target torch dtype.

    Returns:
        Tensor of the requested dtype.
    """
    if dtype == torch.long:
        numpy_type = np.int64
    else:
        numpy_type = np.float32

    if isinstance(arr, torch.Tensor):
        if arr.dtype != dtype:
            if dtype == torch.long:
                arr = arr.long()
            else:
                arr = arr.float()
        return arr

    if isinstance(arr, np.ndarray):
        if arr.dtype != numpy_type:
            arr = arr.astype(numpy_type, copy=False)
        return torch.from_numpy(arr)

    return torch.tensor(arr, dtype=torch.float32)


if DetectionConfig is None:
    raise ImportError(
        "nuscenes detection eval is required for NavsimPerceptionMetric; "
        f"import failed: {_DET_IMPORT_ERROR}"
    )


class CustomDetectionConfig(DetectionConfig):
    """Data class that specifies the detection evaluation settings."""

    def __init__(
        self,
        class_range: Dict[str, int],
        dist_fcn: str,
        dist_ths: List[float],
        dist_th_tp: float,
        min_recall: float,
        min_precision: float,
        max_boxes_per_sample: int,
        mean_ap_weight: int,
    ):
        """Initialize detection evaluation configuration.

        Args:
            class_range: Mapping from class name to detection range.
            dist_fcn: Name of the distance function for matching.
            dist_ths: Distance thresholds for AP computation.
            dist_th_tp: Distance threshold for TP metrics.
            min_recall: Minimum recall for AP integration.
            min_precision: Minimum precision for AP integration.
            max_boxes_per_sample: Maximum number of boxes per sample.
            mean_ap_weight: Weight of mAP in the nuScenes detection score.
        """
        self.class_range = class_range
        self.dist_fcn = dist_fcn
        self.dist_ths = dist_ths
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.max_boxes_per_sample = max_boxes_per_sample
        self.mean_ap_weight = mean_ap_weight

        self.class_names = self.class_range.keys()


class CustomDetectionBox(DetectionBox):
    # Data class used during detection evaluation.
    # Can be a prediction or ground truth.

    def __init__(
        self,
        sample_token: str = "",
        translation: Tuple[float, float, float] = (0, 0, 0),
        size: Tuple[float, float, float] = (0, 0, 0),
        rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
        velocity: Tuple[float, float] = (0, 0),
        ego_translation: [float, float, float] = (0, 0, 0),
        num_pts: int = -1,
        detection_name: str = "car",
        detection_score: float = -1.0,
        attribute_name: str = "",
    ):
        """Initialize a detection evaluation box.

        Args:
            sample_token: Token of the sample this box belongs to.
            translation: Box center translation.
            size: Box size as (width, length, height).
            rotation: Box orientation as quaternion.
            velocity: Box velocity in meters per second.
            ego_translation: Translation to ego vehicle in meters.
            num_pts: Number of lidar or radar points inside the box.
            detection_name: Class name used in the detection challenge.
            detection_score: Detection confidence score.
            attribute_name: Box attribute name.
        """
        super().__init__(
            sample_token,
            translation,
            size,
            rotation,
            velocity,
            ego_translation,
            num_pts,
        )

        assert (
            type(detection_score) == float
        ), "Error: detection_score must be a float!"
        assert not np.any(
            np.isnan(detection_score)
        ), "Error: detection_score may not be NaN!"

        # Assign.
        self.detection_name = detection_name
        self.detection_score = detection_score
        self.attribute_name = attribute_name


@OBJECT_REGISTRY.register
class NavsimPerceptionMetric(EvalMetric):
    """Evaluation Navsim Detection.

    Args:
        name: Name of this metric instance for display.
        save_prefix: Path to save result.
        verbose: Wether output verbose log.
        use_lidar: Whether use lidar bbox.
        classes: List of class name.
        eval_config: The config of eval.
        use_ddp: Whether use ddp to eval metric.
    """

    @require_packages(
        "nuscenes",
        "pyquaternion",
        raise_msg="Please `pip3 install nuscenes-devkit pyquaternion`",
    )
    def __init__(
        self,
        name: str = "NavsimDetectionMetric",
        save_prefix: str = "./WORKSPACE/results",
        verbose: bool = True,
        use_lidar: bool = False,
        classes: Sequence[str] = None,
        use_ddp: bool = True,
        trans_lidar_dim: bool = False,
        trans_lidar_rot: bool = True,
        eval_config: Optional[dict] = None,
        lidar_key="lidar2ego",
    ):
        """Initialize the Navsim perception detection metric.

        Args:
            name: Name of this metric instance for display.
            save_prefix: Path to save result.
            verbose: Whether output verbose log.
            use_lidar: Whether use lidar bbox.
            classes: List of class name.
            use_ddp: Whether use ddp to eval metric.
            trans_lidar_dim: Whether transpose lidar dims when converting boxes.
            trans_lidar_rot: Whether transpose lidar rotation when converting boxes.
            eval_config: The config of eval.
            lidar_key: Key prefix for lidar-to-ego calibration in metadata.
        """
        super(NavsimPerceptionMetric, self).__init__(name)
        self.verbose = verbose
        self.save_prefix = save_prefix
        self.use_ddp = use_ddp
        self.lidar_key = lidar_key

        self.trans_lidar_dim = trans_lidar_dim
        self.trans_lidar_rot = trans_lidar_rot

        if self.use_ddp:
            self.res_path = os.path.join(
                self.save_prefix, f"results_{name}.json"
            )
        else:
            global_rank, _ = get_dist_info()
            self.res_path = os.path.join(
                self.save_prefix, f"results_{name}_{global_rank}.json"
            )
        self.ret = ["NDS", 0.0]
        self.nusc_annos = {}
        self.nusc_gts = {}
        self.use_lidar = use_lidar
        if classes is None:
            self.CLASSES = CLASSES
        else:
            self.CLASSES = classes
        if eval_config is None:
            eval_config = {
                "class_range": {
                    "vehicle": [-51.2, 102.4],
                    "pedestrian": [-40.96, 81.92],
                    "bicycle": [-40.96, 81.92],
                    "czone_sign": [-30.72, 61.44],
                    "barrier": [-30.72, 61.44],
                    "traffic_cone": [-30.72, 61.44],
                    "generic_object": [-30.72, 61.44],
                },
                # nuscenes
                # "class_range":{
                #     "car": 50,
                #     "truck": 50,
                #     "bus": 50,
                #     "trailer": 50,
                #     "construction_vehicle": 50,
                #     "pedestrian": 40,
                #     "motorcycle": 40,
                #     "bicycle": 40,
                #     "traffic_cone": 30,
                #     "barrier": 30
                # },
                "dist_fcn": "center_distance",
                "dist_ths": [0.5, 1.0, 2.0, 4.0],
                "dist_th_tp": 2.0,
                "min_recall": 0.1,
                "min_precision": 0.1,
                "max_boxes_per_sample": 500,
                "mean_ap_weight": 5,
            }
        self.eval_detection_configs = CustomDetectionConfig.deserialize(
            eval_config
        )

    def _output_to_nusc_box(self, bboxes):
        """Convert output bboxes to NuScenesBox list.

        Args:
            bboxes: Output bbox tensor of shape (N, 11)
                [x, y, z, w, l, h, yaw, vx, vy, score, label].

        Returns:
            List of NuScenesBox instances.
        """
        box_list = []
        for bbox in bboxes:
            bbox = bbox.cpu().numpy()
            center = bbox[:3]
            dims = bbox[3:6]
            yaw = bbox[6]
            quat = Quaternion(axis=[0, 0, 1], radians=yaw)
            velocity = (*bbox[7:9], 0.0)
            cat_id = bbox[10]
            score = bbox[9]
            box = NuScenesBox(
                center,
                dims,
                quat,
                label=cat_id,
                score=score,
                velocity=velocity,
            )
            box_list.append(box)
        return box_list

    def _ego_to_global(self, bbox, meta):
        """Transform a box from ego to global coordinate.

        Args:
            bbox: Box to transform.
            meta: Metadata containing ego2global calibration.

        Returns:
            The transformed box.
        """
        ego2global_translation = meta["ego2global_translation"]
        ego2global_rotation = meta["ego2global_rotation"]
        bbox.rotate(Quaternion(ego2global_rotation))
        bbox.translate(ego2global_translation)
        return bbox

    def compute(self):
        """Compute evaluation results (no-op for this metric)."""
        pass

    def _output_to_nusc_box_lidar(self, detection, with_velocity=True):
        """Convert the output to the box class in the nuScenes.

        Args:
            detection: Detection results.

                - boxes_3d: Detection bbox.
                - scores_3d: Detection scores.
                - labels_3d: Predicted box labels.

        Returns:
            list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
        """
        if isinstance(detection, dict):
            box3d = detection["bboxes"].detach().cpu().numpy()
            scores = detection["scores"].detach().cpu().numpy()
            labels = detection["labels"].detach().cpu().numpy()

        else:
            if len(detection) == 0:
                return []
            box3d = detection.cpu().numpy()
        if self.trans_lidar_rot is True:
            box3d[:, 6] = -box3d[:, 6] - np.pi / 2

        box_list = []
        for i in range(len(box3d)):
            quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box3d[i, 6])
            if with_velocity:
                velocity = (*box3d[i, 7:9], 0.0)
            else:
                velocity = (0, 0, 0)
            if isinstance(detection, dict):
                label = labels[i]
                score = scores[i]
            else:
                label = box3d[i, 10]
                score = box3d[i, 9]

            box = NuScenesBox(
                box3d[i, :3],
                (
                    box3d[i, 3:6]
                    if self.trans_lidar_dim is False
                    else box3d[i, [4, 3, 5]]
                ),
                quat,
                label=label,
                score=score,
                velocity=velocity,
            )
            box_list.append(box)
        return box_list

    def _lidar_nusc_box_to_global(
        self,
        info,
        box,
        classes,
        eval_configs,
    ):
        """Convert the box from ego to global coordinate.

        Args:
            info (dict): Info for a specific sample data, including the
                calibration information.
            boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
            classes (list[str]): Mapped classes in the evaluation.
            eval_configs (object): Evaluation configuration object.

        Returns:
            list: List of standard NuScenesBoxes in the global
                coordinate.
        """

        # Move box to ego vehicle coord system
        lidar2ego_rotation = self.lidar_key + "_rotation"
        lidar2ego_translation = self.lidar_key + "_translation"
        box.rotate(pyquaternion.Quaternion(info[lidar2ego_rotation]))
        box.translate(np.array(info[lidar2ego_translation]))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2])
        rear_range, front_range = cls_range_map[classes[box.label]]
        if (box.center[0] < 0 and radius > abs(rear_range)) or (
            box.center[0] >= 0 and radius > front_range
        ):
            return None
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info["ego2global_rotation"]))
        box.translate(np.array(info["ego2global_translation"]))

        return box

    def update(self, meta, pred_bboxes, gt_bboxes):
        """Update detection predictions and ground truths.

        Args:
            meta: List of metadata dicts for each sample.
            pred_bboxes: List of predicted bbox tensors.
            gt_bboxes: List of ground truth bbox tensors.
        """
        for m, bboxes in zip(meta, pred_bboxes):
            annos = []
            if self.use_lidar:
                bboxes_list = self._output_to_nusc_box_lidar(bboxes)
            else:
                bboxes_list = self._output_to_nusc_box(bboxes)
            sample_token = m["sample_token"]
            for box in bboxes_list:
                if self.use_lidar:
                    box = self._lidar_nusc_box_to_global(
                        m, box, self.CLASSES, self.eval_detection_configs
                    )
                else:
                    # ego2global_translation
                    # ego2global_rotation
                    box = self._ego_to_global(box, m)
                if box is None:
                    continue
                name = self.CLASSES[box.label]
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                        "vehicle",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = DefaultAttribute[name]
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = DefaultAttribute[name]

                nusc_anno = {
                    "sample_token": sample_token,
                    "translation": box.center.tolist(),
                    "size": box.wlh.tolist(),
                    "rotation": box.orientation.elements.tolist(),
                    "velocity": box.velocity[:2].tolist(),
                    "detection_name": name,
                    "detection_score": box.score,
                    "attribute_name": attr,
                }
                annos.append(nusc_anno)
            self.nusc_annos[sample_token] = annos
        for m, bboxes in zip(meta, gt_bboxes):
            gts = []
            if self.use_lidar:
                bboxes_list = self._output_to_nusc_box_lidar(bboxes)
            else:
                bboxes_list = self._output_to_nusc_box(bboxes)
            sample_token = m["sample_token"]
            for box in bboxes_list:
                if self.use_lidar:
                    box = self._lidar_nusc_box_to_global(
                        m, box, self.CLASSES, self.eval_detection_configs
                    )
                else:
                    box = self._ego_to_global(box, m)
                if box is None:
                    continue
                name = self.CLASSES[box.label]
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                        "vehicle",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = DefaultAttribute[name]
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = DefaultAttribute[name]

                nusc_gt = {
                    "sample_token": sample_token,
                    "translation": box.center.tolist(),
                    "size": box.wlh.tolist(),
                    "rotation": box.orientation.elements.tolist(),
                    "velocity": box.velocity[:2].tolist(),
                    "detection_name": name,
                    "detection_score": box.score,
                    "attribute_name": attr,
                }
                gts.append(nusc_gt)
            self.nusc_gts[sample_token] = gts

    def _gather(self):
        global_rank, global_world_size = get_dist_info()
        global_output = [None for _ in range(global_world_size)]
        all_gather_object(global_output, self.nusc_annos)
        return global_output

    def _gather_gt(self):
        _, global_world_size = get_dist_info()
        global_output = [None for _ in range(global_world_size)]
        all_gather_object(global_output, self.nusc_gts)
        return global_output

    def reset(self):
        self.nusc_annos = {}
        self.nusc_gts = {}
        self.ret = ["NDS", 0.0]

    def get(self):
        logger.info(
            f"The length of self.nusc_annos is: {len(self.nusc_annos)}"
        )
        if len(self.nusc_annos) != 0:
            if self.use_ddp:
                nusc_annos = self._gather()
                self.nusc_annos = nusc_annos[0]
                for nusc_annos in nusc_annos[1:]:
                    self.nusc_annos.update(nusc_annos)
                nusc_gts = self._gather_gt()
                self.nusc_gts = nusc_gts[0]
                for nusc_gts in nusc_gts[1:]:
                    self.nusc_gts.update(nusc_gts)
                logger.info(
                    "After DDP gather: nusc_annos=%d, nusc_gts=%d",
                    len(self.nusc_annos),
                    len(self.nusc_gts),
                )
            self._get()
        return self.ret[0], self.ret[1]

    def _dump(self):
        modality = {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }
        nusc_sub = {
            "meta": modality,
            "results": self.nusc_annos,
        }
        logger.info(f"Results writes to {self.res_path}")
        if not os.path.exists(self.save_prefix):
            os.makedirs(self.save_prefix)

        with open(self.res_path, "w") as fs:
            json.dump(nusc_sub, fs)

    @rank_zero_only
    def _evaluate(self, pred, gt):
        start_time = time.time()
        # -----------------------------------
        # Step 1: Accumulate metric data for \
        #   all classes and distance thresholds.
        # -----------------------------------
        if self.verbose:
            print("Accumulating metric data...")
        metric_data_list = DetectionMetricDataList()

        for class_name in self.cfg.class_names:
            for dist_th in self.cfg.dist_ths:
                md = accumulate(
                    gt, pred, class_name, self.cfg.dist_fcn_callable, dist_th
                )
                metric_data_list.set(class_name, dist_th, md)

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        if self.verbose:
            print("Calculating metrics...")
        metrics = DetectionMetrics(self.cfg)
        for class_name in self.cfg.class_names:
            # Compute APs.
            for dist_th in self.cfg.dist_ths:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(
                    metric_data, self.cfg.min_recall, self.cfg.min_precision
                )
                metrics.add_label_ap(class_name, dist_th, ap)

            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[
                    (class_name, self.cfg.dist_th_tp)
                ]
                if class_name in [
                    "traffic_cone",
                    "generic_object",
                    "czone_sign",
                ] and metric_name in ["attr_err", "vel_err", "orient_err"]:
                    tp = np.nan
                elif class_name in ["barrier"] and metric_name in [
                    "attr_err",
                    "vel_err",
                ]:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, self.cfg.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)

        # Compute evaluation time.
        metrics.add_runtime(time.time() - start_time)

        return metrics, metric_data_list

    def evaluate(self, pred, gt):
        metrics, metric_data_list = self._evaluate(pred, gt)
        if self.verbose:
            print("Saving metrics to: %s" % self.save_prefix)
        metrics_summary = metrics.serialize()
        # metrics_summary['meta'] = self.meta.copy()
        with open(
            os.path.join(self.save_prefix, "metrics_summary.json"), "w"
        ) as f:
            json.dump(metrics_summary, f, indent=2)
        with open(
            os.path.join(self.save_prefix, "metrics_details.json"), "w"
        ) as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        # Print high-level metrics.
        print("mAP: %.4f" % (metrics_summary["mean_ap"]))
        err_name_mapping = {
            "trans_err": "mATE",
            "scale_err": "mASE",
            "orient_err": "mAOE",
            "vel_err": "mAVE",
            "attr_err": "mAAE",
        }
        for tp_name, tp_val in metrics_summary["tp_errors"].items():
            print("%s: %.4f" % (err_name_mapping[tp_name], tp_val))
        print("NDS: %.4f" % (metrics_summary["nd_score"]))
        print("Eval time: %.1fs" % metrics_summary["eval_time"])

        # Print per-class metrics.
        print()
        print("Per-class results:")
        print(
            "%-20s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s\t%-6s"
            % ("Object Class", "AP", "ATE", "ASE", "AOE", "AVE", "AAE")
        )
        class_aps = metrics_summary["mean_dist_aps"]
        class_tps = metrics_summary["label_tp_errors"]
        for class_name in class_aps.keys():
            print(
                "%-20s\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f\t%-6.3f"
                % (
                    class_name,
                    class_aps[class_name],
                    class_tps[class_name]["trans_err"],
                    class_tps[class_name]["scale_err"],
                    class_tps[class_name]["orient_err"],
                    class_tps[class_name]["vel_err"],
                    class_tps[class_name]["attr_err"],
                )
            )

        return metrics_summary

    @rank_zero_only
    def _get(self):
        self._dump()
        pred_boxes_anno = EvalBoxes.deserialize(
            self.nusc_annos, CustomDetectionBox
        )
        gt_boxes_anno = EvalBoxes.deserialize(
            self.nusc_gts, CustomDetectionBox
        )
        self.cfg = self.eval_detection_configs
        self.cfg.class_names = self.CLASSES

        self.evaluate(pred_boxes_anno, gt_boxes_anno)

        # record metrics
        metric_file = os.path.join(self.save_prefix, "metrics_summary.json")
        with open(metric_file) as fs:
            metrics = json.load(fs)

        log_info = ""
        log_info += "NDS: {:.4f}, mAP:{:.4f}\n".format(
            metrics["nd_score"], metrics["mean_ap"]
        )
        for name in self.CLASSES:
            log_info += "{}_AP:".format(name)
            for k, v in metrics["label_aps"][name].items():
                log_info += " [{}]:{:.4f} ".format(k, v)
            log_info += "\n"
        logger.info(log_info)
        self.ret[1] = metrics["nd_score"]


if TrackingConfig is None:
    CustomTrackingConfig = None  # type: ignore[misc,assignment]
    CustomTrackingBox = None  # type: ignore[misc,assignment]
else:

    class CustomTrackingConfig(TrackingConfig):
        """Data class that specifies the tracking evaluation settings."""

        def __init__(
            self,
            tracking_names: List[str],
            pretty_tracking_names: Dict[str, str],
            tracking_colors: Dict[str, str],
            class_range: Dict[str, int],
            dist_fcn: str,
            dist_th_tp: float,
            min_recall: float,
            max_boxes_per_sample: int,
            metric_worst: Dict[str, float],
            num_thresholds: int,
        ):
            self.tracking_names = tracking_names
            self.pretty_tracking_names = pretty_tracking_names
            self.tracking_colors = tracking_colors
            self.class_range = class_range
            self.dist_fcn = dist_fcn
            self.dist_th_tp = dist_th_tp
            self.min_recall = min_recall
            self.max_boxes_per_sample = max_boxes_per_sample
            self.metric_worst = metric_worst
            self.num_thresholds = num_thresholds

            TrackingMetricData.set_nelem(num_thresholds)

            self.class_names = sorted(self.class_range.keys())

    class CustomTrackingBox(EvalBox):
        """Data class used during tracking evaluation for navsim."""

        def __init__(
            self,
            sample_token: str = "",
            translation: Tuple[float, float, float] = (0, 0, 0),
            size: Tuple[float, float, float] = (0, 0, 0),
            rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
            velocity: Tuple[float, float] = (0, 0),
            ego_translation: Tuple[float, float, float] = (0, 0, 0),
            num_pts: int = -1,
            tracking_id: str = "",
            tracking_name: str = "",
            tracking_score: float = -1.0,
        ):
            super().__init__(
                sample_token,
                translation,
                size,
                rotation,
                velocity,
                ego_translation,
                num_pts,
            )

            assert (
                type(tracking_score) == float
            ), "Error: tracking_score must be a float!"
            assert not np.any(
                np.isnan(tracking_score)
            ), "Error: tracking_score may not be NaN!"

            # Assign.
            self.tracking_id = tracking_id
            self.tracking_name = tracking_name
            self.tracking_score = tracking_score

        def __eq__(self, other):
            return (
                self.sample_token == other.sample_token
                and self.translation == other.translation
                and self.size == other.size
                and self.rotation == other.rotation
                and self.velocity == other.velocity
                and self.ego_translation == other.ego_translation
                and self.num_pts == other.num_pts
                and self.tracking_id == other.tracking_id
                and self.tracking_name == other.tracking_name
                and self.tracking_score == other.tracking_score
            )

        def serialize(self) -> dict:
            """Serialize instance into json-friendly format."""
            return {
                "sample_token": self.sample_token,
                "translation": self.translation,
                "size": self.size,
                "rotation": self.rotation,
                "velocity": self.velocity,
                "ego_translation": self.ego_translation,
                "num_pts": self.num_pts,
                "tracking_id": self.tracking_id,
                "tracking_name": self.tracking_name,
                "tracking_score": self.tracking_score,
            }

        @classmethod
        def deserialize(cls, content: dict):
            """Initialize from serialized content."""
            return cls(
                sample_token=content["sample_token"],
                translation=tuple(content["translation"]),
                size=tuple(content["size"]),
                rotation=tuple(content["rotation"]),
                velocity=tuple(content["velocity"]),
                ego_translation=(
                    (0.0, 0.0, 0.0)
                    if "ego_translation" not in content
                    else tuple(content["ego_translation"])
                ),
                num_pts=(
                    -1 if "num_pts" not in content else int(content["num_pts"])
                ),
                tracking_id=content["tracking_id"],
                tracking_name=content["tracking_name"],
                tracking_score=(
                    -1.0
                    if "tracking_score" not in content
                    else float(content["tracking_score"])
                ),
            )


@OBJECT_REGISTRY.register
class NavsimPerceptionTrackMetric(EvalMetric):
    """Evaluation Navsim Tracking.

    Args:
        name: Name of this metric instance for display.
        save_prefix: Path to save result.
        verbose: Whether output verbose log.
        use_lidar: Whether use lidar bbox.
        classes: List of class name.
        use_ddp: Whether use ddp to eval metric.
        tracking_threshold: Score threshold for tracking predictions.
            If None, all predictions are used.
        num_thresholds: Number of recall thresholds for evaluation.
    """

    _patch_applied = False
    _NAVSIM_TRACKING_NAMES = [
        "vehicle",
        "pedestrian",
        "bicycle",
        "czone_sign",
        "barrier",
        "traffic_cone",
        "generic_object",
    ]

    @staticmethod
    def _patch_nuscenes_tracking_compat(tracking_names=None):
        """
        Patch pandas 2.x / motmetrics 1.4.0 incompatibilities \
            in nuscenes tracking evaluation code.

        Args:
            tracking_names: List of tracking class names to patch
                into TRACKING_NAMES. If None, uses
                _NAVSIM_TRACKING_NAMES as default.
        """
        if NavsimPerceptionTrackMetric._patch_applied:
            return
        NavsimPerceptionTrackMetric._patch_applied = True

        import pandas as pd
        from nuscenes.eval.tracking import data_classes as nusc_track_dc
        from nuscenes.eval.tracking import metrics as nusc_track_metrics
        from nuscenes.eval.tracking import mot as nusc_mot
        from nuscenes.eval.tracking import utils as nusc_track_utils

        # Patch 0: Set TRACKING_NAMES for custom classes.
        # TrackingBox.__init__ validates tracking_name against this list,
        # but it's empty by default (populated by nuscenes TrackingEval).
        if len(nusc_track_dc.TRACKING_NAMES) == 0:
            if tracking_names is None:
                tracking_names = list(
                    NavsimPerceptionTrackMetric._NAVSIM_TRACKING_NAMES
                )
            nusc_track_dc.TRACKING_NAMES = list(tracking_names)

        # Patch 1: MOTAccumulatorCustom.new_event_dataframe_with_data
        # In pandas 2.x, MultiIndex.from_tuples requires list of tuples,
        # not a dict.
        _orig_new_event_df = (
            nusc_mot.MOTAccumulatorCustom.new_event_dataframe_with_data
        )

        @staticmethod
        def _patched_new_event_dataframe_with_data(event_indices, events=None):
            if isinstance(event_indices, dict):
                event_indices = list(
                    zip(event_indices["FrameId"], event_indices["Event"])
                )
            return _orig_new_event_df(event_indices, events=events)

        nusc_mot.MOTAccumulatorCustom.new_event_dataframe_with_data = (
            _patched_new_event_dataframe_with_data
        )

        # Patch 2: MOTAccumulatorCustom.merge_event_dataframes
        # DataFrame.append was removed in pandas 2.0; use pd.concat.
        @staticmethod
        def _patched_merge_event_dataframes(
            dfs,
            update_frame_indices=True,
            update_oids=True,
            update_hids=True,
            return_mappings=False,
        ):
            from itertools import count

            mapping_infos = []
            new_oid = count()
            new_hid = count()

            r = nusc_mot.MOTAccumulatorCustom.new_event_dataframe()

            for df in dfs:
                if isinstance(df, nusc_mot.MOTAccumulatorCustom):
                    df = df.events

                copy = df.copy()
                infos = {}

                if update_frame_indices:
                    if isinstance(r.index, pd.MultiIndex) and len(r.index) > 0:
                        max_frame = r.index.get_level_values(0).max()
                    elif len(r.index) > 0:
                        max_frame = max(x[0] for x in r.index)
                    else:
                        max_frame = -1
                    next_frame_id = int(max_frame) + 1
                    if isinstance(r.index, pd.MultiIndex) and len(r.index) > 0:
                        n_unique = (
                            r.index.get_level_values(0).unique().shape[0]
                        )
                    elif len(r.index) > 0:
                        n_unique = len({x[0] for x in r.index})
                    else:
                        n_unique = 0
                    next_frame_id = max(next_frame_id, n_unique)
                    if np.isnan(next_frame_id):
                        next_frame_id = 0
                    copy.index = pd.MultiIndex.from_tuples(
                        [(x[0] + next_frame_id, x[1]) for x in copy.index],
                        names=["FrameId", "Event"],
                    )
                    infos["frame_offset"] = next_frame_id

                if update_oids:
                    oid_map = dict(
                        [oid, str(next(new_oid))]
                        for oid in copy["OId"].dropna().unique()
                    )
                    copy["OId"] = copy["OId"].map(
                        lambda x: oid_map[x], na_action="ignore"
                    )
                    infos["oid_map"] = oid_map

                if update_hids:
                    hid_map = dict(
                        [hid, str(next(new_hid))]
                        for hid in copy["HId"].dropna().unique()
                    )
                    copy["HId"] = copy["HId"].map(
                        lambda x: hid_map[x], na_action="ignore"
                    )
                    infos["hid_map"] = hid_map

                r = pd.concat([r, copy])
                mapping_infos.append(infos)

            if return_mappings:
                return r, mapping_infos
            else:
                return r

        nusc_mot.MOTAccumulatorCustom.merge_event_dataframes = (
            _patched_merge_event_dataframes
        )

        # Patch 3: Custom metric functions need **kwargs to accept
        # options={'ana': None} from MetricsHost.compute() in
        # motmetrics 1.4.0.
        def _make_kwargs_wrapper(orig_fn):
            def wrapper(*args, **kwargs):
                return orig_fn(*args)

            return wrapper

        _patch_fns = [
            "motar",
            "mota_custom",
            "motp_custom",
            "faf",
            "num_fragmentations_custom",
            "track_initialization_duration",
            "longest_gap_duration",
        ]
        for fn_name in _patch_fns:
            if hasattr(nusc_track_metrics, fn_name):
                orig_fn = getattr(nusc_track_metrics, fn_name)
                patched_fn = _make_kwargs_wrapper(orig_fn)
                setattr(nusc_track_metrics, fn_name, patched_fn)
                if hasattr(nusc_track_utils, fn_name):
                    setattr(nusc_track_utils, fn_name, patched_fn)

        # Patch 4: create_motmetrics must register pred_frequencies
        # since num_predictions depends on it.
        import motmetrics as mm
        from nuscenes.eval.tracking import algo as nusc_track_algo

        _orig_create_motmetrics = nusc_track_utils.create_motmetrics

        def _patched_create_motmetrics():
            mh = _orig_create_motmetrics()
            try:
                mh.register(
                    mm.metrics.pred_frequencies,
                    name="pred_frequencies",
                )
            except Exception:
                pass
            return mh

        nusc_track_utils.create_motmetrics = _patched_create_motmetrics
        nusc_track_algo.create_motmetrics = _patched_create_motmetrics

        # Patch 5: Fix duplicate_thresholds calculation in
        # TrackingEvaluation.accumulate().
        # numpy 1.24+ collapses all NaN values into a single element
        # in np.unique, so the original formula
        #   duplicate_thresholds = len(thresholds) - len(np.unique(thresholds))
        # inflates duplicate_thresholds by (N_nan - 1) when there
        # are NaN thresholds, causing the assertion
        #   unachieved + duplicate + len(thresh_metrics) == num_thresholds
        # to fail. We replace accumulate() with a copy that fixes
        # only this one line.
        def _patched_accumulate(self_ev):
            import pandas
            from nuscenes.eval.tracking.constants import MOT_METRIC_MAP
            from nuscenes.eval.tracking.data_classes import TrackingMetricData
            from nuscenes.eval.tracking.utils import (
                create_motmetrics,
                print_threshold_metrics,
            )

            if self_ev.verbose:
                print(
                    "Computing metrics for class %s...\n" % self_ev.class_name
                )
            accumulators = []
            thresh_metrics = []
            md = TrackingMetricData()

            gt_box_count = 0
            gt_track_ids = set()
            for scene_tracks_gt in self_ev.tracks_gt.values():
                for frame_gt in scene_tracks_gt.values():
                    for box in frame_gt:
                        if box.tracking_name == self_ev.class_name:
                            gt_box_count += 1
                            gt_track_ids.add(box.tracking_id)
            if gt_box_count == 0:
                return md

            mh = create_motmetrics()

            thresholds, recalls = self_ev.compute_thresholds(gt_box_count)
            md.confidence = thresholds
            md.recall_hypo = recalls
            if self_ev.verbose:
                print("Computed thresholds\n")

            for t, threshold in enumerate(thresholds):
                if np.isnan(threshold):
                    continue
                if threshold in thresholds[:t]:
                    continue
                acc, _ = self_ev.accumulate_threshold(threshold)
                accumulators.append(acc)
                thresh_name = self_ev.name_gen(threshold)
                thresh_summary = mh.compute(
                    acc,
                    metrics=MOT_METRIC_MAP.keys(),
                    name=thresh_name,
                )
                thresh_metrics.append(thresh_summary)
                if self_ev.verbose:
                    print_threshold_metrics(thresh_summary.to_dict())

            if len(thresh_metrics) == 0:
                summary = []
            else:
                summary = pandas.concat(thresh_metrics)

            # Sanity checks.
            unachieved_thresholds = np.sum(np.isnan(thresholds))
            # FIX: only count duplicates among non-NaN values,
            # because np.unique collapses all NaNs into one.
            non_nan_thresholds = np.array(
                [t for t in thresholds if not np.isnan(t)]
            )
            duplicate_thresholds = len(non_nan_thresholds) - len(
                np.unique(non_nan_thresholds)
            )
            assert (
                unachieved_thresholds
                + duplicate_thresholds
                + len(thresh_metrics)
                == self_ev.num_thresholds
            )

            valid_thresholds = [t for t in thresholds if not np.isnan(t)]
            assert valid_thresholds == sorted(valid_thresholds)
            rep_counts = [
                np.sum(np.array(thresholds) == t)
                for t in np.unique(valid_thresholds)
            ]

            for mot_name, metric_name in MOT_METRIC_MAP.items():
                if metric_name == "":
                    continue
                if len(thresh_metrics) == 0:
                    worst = self_ev.metric_worst[metric_name]
                    if worst == -1:
                        if metric_name == "ml":
                            worst = len(gt_track_ids)
                        elif metric_name in ["gt", "fn"]:
                            worst = gt_box_count
                        elif metric_name in [
                            "fp",
                            "ids",
                            "frag",
                        ]:
                            worst = np.nan
                        else:
                            raise NotImplementedError
                    all_values = [worst] * TrackingMetricData.nelem
                else:
                    values = summary.get(mot_name).values
                    assert np.all(
                        values[np.logical_not(np.isnan(values))] >= 0
                    )
                    assert len(rep_counts) == len(values)
                    values = np.concatenate(
                        [([v] * r) for v, r in zip(values, rep_counts)]
                    )
                    all_values = [np.nan] * int(unachieved_thresholds)
                    all_values.extend(values)

                assert len(all_values) == TrackingMetricData.nelem
                md.set_metric(metric_name, all_values)

            return md

        nusc_track_algo.TrackingEvaluation.accumulate = _patched_accumulate

    @require_packages(
        "nuscenes",
        "pyquaternion",
        "motmetrics",
        raise_msg="Please `pip3 install nuscenes-devkit pyquaternion motmetrics`",  # noqa
    )
    def __init__(
        self,
        name: str = "NavsimTrackingMetric",
        save_prefix: str = "./WORKSPACE/results",
        verbose: bool = True,
        use_lidar: bool = False,
        classes: Sequence[str] = None,
        use_ddp: bool = True,
        trans_lidar_dim: bool = False,
        trans_lidar_rot: bool = True,
        tracking_threshold: float = None,
        eval_config: Optional[dict] = None,
        lidar_key="lidar2ego",
    ):
        super(NavsimPerceptionTrackMetric, self).__init__(name)
        self.verbose = verbose
        self.save_prefix = save_prefix
        self.use_ddp = use_ddp
        self.lidar_key = lidar_key
        self.tracking_threshold = tracking_threshold

        self.trans_lidar_dim = trans_lidar_dim
        self.trans_lidar_rot = trans_lidar_rot

        if self.use_ddp:
            self.res_path = os.path.join(
                self.save_prefix, f"results_{name}.json"
            )
        else:
            global_rank, _ = get_dist_info()
            self.res_path = os.path.join(
                self.save_prefix, f"results_{name}_{global_rank}.json"
            )

        self.use_lidar = use_lidar
        if classes is None:
            self.CLASSES = CLASSES
        else:
            self.CLASSES = classes

        if eval_config is None:
            eval_config = {
                "tracking_names": [
                    "vehicle",
                    "bicycle",
                    "pedestrian",
                ],
                "pretty_tracking_names": {
                    "vehicle": "Vehicle",
                    "bicycle": "Bicycle",
                    "pedestrian": "Pedestrian",
                },
                "tracking_colors": {
                    "bicycle": "C9",
                    "pedestrian": "C5",
                    "vehicle": "C1",
                },
                "class_range": {
                    "vehicle": [-51.2, 102.4],
                    "pedestrian": [-40.96, 81.92],
                    "bicycle": [-40.96, 81.92],
                },
                "dist_fcn": "center_distance",
                "dist_th_tp": 2.0,
                "min_recall": 0.1,
                "max_boxes_per_sample": 500,
                "metric_worst": {
                    "amota": 0.0,
                    "amotp": 2.0,
                    "recall": 0.0,
                    "motar": 0.0,
                    "mota": 0.0,
                    "motp": 2.0,
                    "mt": 0.0,
                    "ml": -1.0,
                    "faf": 500,
                    "gt": -1,
                    "tp": 0.0,
                    "fp": -1.0,
                    "fn": -1.0,
                    "ids": -1.0,
                    "frag": -1.0,
                    "tid": 20,
                    "lgd": 20,
                },
                "num_thresholds": 40,
            }
        self.eval_config = eval_config

        self.eval_track_configs = CustomTrackingConfig.deserialize(eval_config)
        self.reset()

    def _output_to_nusc_box(self, bboxes, instance_ids=None):
        box_list = []
        for i, bbox in enumerate(bboxes):
            bbox = bbox.cpu().numpy()
            center = bbox[:3]
            dims = bbox[3:6]
            yaw = bbox[6]
            quat = Quaternion(axis=[0, 0, 1], radians=yaw)
            velocity = (*bbox[7:9], 0.0)
            cat_id = bbox[10]
            score = bbox[9]
            box = NuScenesBox(
                center,
                dims,
                quat,
                label=cat_id,
                score=score,
                velocity=velocity,
            )
            if instance_ids is not None:
                box.token = instance_ids[i]
            box_list.append(box)
        return box_list

    def _ego_to_global(self, bbox, meta):
        ego2global_translation = meta["ego2global_translation"]
        ego2global_rotation = meta["ego2global_rotation"]
        bbox.rotate(Quaternion(ego2global_rotation))
        bbox.translate(ego2global_translation)
        return bbox

    def compute(self):
        pass

    def _output_to_nusc_box_lidar(
        self, detection, with_velocity=True, threshold=None
    ):
        """Convert the output to the box class in the nuScenes.

        Args:
            detection: Detection results.

                - bboxes: Detection bbox (N, 11).
                - scores: Detection scores (N,).
                - labels: Predicted box labels (N,).
                - instance_ids: Tracking instance IDs (N,).

        Returns:
            list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
        """
        if isinstance(detection, dict):
            box3d = detection["bboxes"].detach().cpu().numpy()
            scores = detection["scores"].detach().cpu().numpy()
            labels = detection["labels"].detach().cpu().numpy()
            ids = detection.get("instance_ids", None).detach().cpu().numpy()
        else:
            if len(detection) == 0:
                return []
            box3d = detection.cpu().numpy()
            ids = None

        if self.trans_lidar_rot is True:
            box3d[:, 6] = -box3d[:, 6] - np.pi / 2

        if threshold is not None:
            if isinstance(detection, dict):
                if "cls_scores" in detection:
                    mask = detection["cls_scores"].numpy() >= threshold
                else:
                    mask = scores >= threshold
            else:
                mask = box3d[:, 9] >= threshold
            box3d = box3d[mask]
            if isinstance(detection, dict):
                scores = scores[mask]
                labels = labels[mask]
            if ids is not None:
                ids = ids[mask]

        box_list = []
        for i in range(len(box3d)):
            quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box3d[i, 6])
            if with_velocity:
                velocity = (*box3d[i, 7:9], 0.0)
            else:
                velocity = (0, 0, 0)
            if isinstance(detection, dict):
                label = labels[i]
                score = scores[i]
            else:
                label = box3d[i, 10]
                score = box3d[i, 9]

            box = NuScenesBox(
                box3d[i, :3],
                (
                    box3d[i, 3:6]
                    if self.trans_lidar_dim is False
                    else box3d[i, [4, 3, 5]]
                ),
                quat,
                label=label,
                score=score,
                velocity=velocity,
            )
            if ids is not None:
                box.token = ids[i]
            box_list.append(box)
        return box_list

    def _lidar_nusc_box_to_global(self, info, box, classes, eval_configs):
        """Convert the box from ego to global coordinate."""
        lidar2ego_rotation = self.lidar_key + "_rotation"
        lidar2ego_translation = self.lidar_key + "_translation"
        box.rotate(pyquaternion.Quaternion(info[lidar2ego_rotation]))
        box.translate(np.array(info[lidar2ego_translation]))

        cls_range_map = eval_configs.class_range
        if isinstance(list(cls_range_map.values())[0], (list, tuple)):
            radius = np.linalg.norm(box.center[:2])
            rear_range, front_range = cls_range_map[classes[box.label]]
            if (box.center[0] < 0 and radius > abs(rear_range)) or (
                box.center[0] >= 0 and radius > front_range
            ):
                return None
        else:
            radius = np.linalg.norm(box.center[:2], 2)
            det_range = cls_range_map[classes[box.label]]
            if radius > det_range:
                return None

        box.rotate(pyquaternion.Quaternion(info["ego2global_rotation"]))
        box.translate(np.array(info["ego2global_translation"]))
        return box

    def update(
        self,
        meta,
        pred_bboxes,
        gt_bboxes,
        gt_instance_ids=None,
        pred_instance_ids=None,
    ):
        """Update tracking predictions and ground truths.

        Args:
            meta: List of metadata dicts for each sample.
            pred_bboxes: List of predicted bbox tensors.
            gt_bboxes: List of ground truth bbox tensors.
            gt_instance_ids: List of gt instance id lists per sample.
            pred_instance_ids: List of pred instance id lists per sample.
        """
        for idx, (m, bboxes) in enumerate(zip(meta, pred_bboxes)):
            annos = []
            p_ids = None
            if pred_instance_ids is not None:
                p_ids = pred_instance_ids[idx]
            if self.use_lidar:
                if isinstance(bboxes, dict):
                    det = dict(bboxes)
                    if p_ids is not None:
                        det["instance_ids"] = p_ids
                    bboxes_list = self._output_to_nusc_box_lidar(
                        det, threshold=self.tracking_threshold
                    )
                else:
                    bboxes_list = self._output_to_nusc_box(
                        bboxes, instance_ids=p_ids
                    )
            else:
                bboxes_list = self._output_to_nusc_box(
                    bboxes, instance_ids=p_ids
                )
            sample_token = m["sample_token"]
            self.sample_meta[sample_token] = {
                "scene": m.get("scene", sample_token),
                "timestamp": float(m.get("timestamp", 0).cpu()),
            }
            for box in bboxes_list:
                if self.use_lidar:
                    box = self._lidar_nusc_box_to_global(
                        m, box, self.CLASSES, self.eval_track_configs
                    )
                else:
                    box = self._ego_to_global(box, m)
                if box is None:
                    continue
                name = self.CLASSES[box.label]
                tracking_id = (
                    str(box.token)
                    if hasattr(box, "token") and box.token is not None
                    else ""
                )
                nusc_anno = {
                    "sample_token": sample_token,
                    "translation": box.center.tolist(),
                    "size": box.wlh.tolist(),
                    "rotation": box.orientation.elements.tolist(),
                    "velocity": box.velocity[:2].tolist(),
                    "tracking_name": name,
                    "tracking_score": box.score,
                    "tracking_id": tracking_id,
                }
                annos.append(nusc_anno)
            self.nusc_annos[sample_token] = annos

        for idx, (m, bboxes) in enumerate(zip(meta, gt_bboxes)):
            gts = []
            g_ids = None
            if gt_instance_ids is not None:
                g_ids = gt_instance_ids[idx]
            if self.use_lidar:
                if isinstance(bboxes, dict):
                    det = dict(bboxes)
                    if g_ids is not None:
                        det["instance_ids"] = g_ids
                    bboxes_list = self._output_to_nusc_box_lidar(det)
                else:
                    bboxes_list = self._output_to_nusc_box(
                        bboxes, instance_ids=g_ids
                    )
            else:
                bboxes_list = self._output_to_nusc_box(
                    bboxes, instance_ids=g_ids
                )
            sample_token = m["sample_token"]
            for box in bboxes_list:
                if self.use_lidar:
                    box = self._lidar_nusc_box_to_global(
                        m, box, self.CLASSES, self.eval_track_configs
                    )
                else:
                    box = self._ego_to_global(box, m)
                if box is None:
                    continue
                name = self.CLASSES[box.label]
                tracking_id = (
                    str(box.token)
                    if hasattr(box, "token") and box.token is not None
                    else ""
                )
                nusc_gt = {
                    "sample_token": sample_token,
                    "translation": box.center.tolist(),
                    "size": box.wlh.tolist(),
                    "rotation": box.orientation.elements.tolist(),
                    "velocity": box.velocity[:2].tolist(),
                    "tracking_name": name,
                    "tracking_score": box.score,
                    "tracking_id": tracking_id,
                }
                gts.append(nusc_gt)
            self.nusc_gts[sample_token] = gts

    def _gather(self):
        global_rank, global_world_size = get_dist_info()
        global_output = [None for _ in range(global_world_size)]
        all_gather_object(global_output, self.nusc_annos)
        return global_output

    def _gather_gt(self):
        global_rank, global_world_size = get_dist_info()
        global_output = [None for _ in range(global_world_size)]
        all_gather_object(global_output, self.nusc_gts)
        return global_output

    def _gather_meta(self):
        global_rank, global_world_size = get_dist_info()
        global_output = [None for _ in range(global_world_size)]
        all_gather_object(global_output, self.sample_meta)
        return global_output

    def reset(self):
        self.nusc_annos = {}
        self.nusc_gts = {}
        self.sample_meta = {}
        self.ret = {
            "amota": 0.0,
            "amotp": 0.0,
            "ids": 0,
        }

    def get(self):
        logger.info(
            f"The length of self.nusc_annos is: {len(self.nusc_annos)}"
        )
        if len(self.nusc_annos) != 0:
            if self.use_ddp:
                nusc_annos = self._gather()
                self.nusc_annos = nusc_annos[0]
                for nusc_annos in nusc_annos[1:]:
                    self.nusc_annos.update(nusc_annos)
                nusc_gts = self._gather_gt()
                self.nusc_gts = nusc_gts[0]
                for nusc_gts in nusc_gts[1:]:
                    self.nusc_gts.update(nusc_gts)
                sample_metas = self._gather_meta()
                self.sample_meta = sample_metas[0]
                for sample_metas in sample_metas[1:]:
                    self.sample_meta.update(sample_metas)
            self._get()
        res_names = []
        res_values = []
        for name, value in self.ret.items():
            res_names.append(name)
            res_values.append(value)
        return res_names, res_values

    def _dump(self):
        modality = {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }
        nusc_sub = {
            "meta": modality,
            "results": self.nusc_annos,
        }
        logger.info(f"Results writes to {self.res_path}")
        if not os.path.exists(self.save_prefix):
            os.makedirs(self.save_prefix)

        with open(self.res_path, "w") as fs:
            json.dump(nusc_sub, fs)

    @rank_zero_only
    def _get(self):
        self._dump()
        pred_boxes_anno = EvalBoxes.deserialize(
            self.nusc_annos, CustomTrackingBox
        )
        gt_boxes_anno = EvalBoxes.deserialize(self.nusc_gts, CustomTrackingBox)

        self._evaluate(pred_boxes_anno, gt_boxes_anno)

        metric_file = os.path.join(self.save_prefix, "metrics_summary.json")
        with open(metric_file) as fs:
            metrics = json.load(fs)

        keys = [
            "amota",
            "amotp",
            "recall",
            "motar",
            "gt",
            "mota",
            "motp",
            "mt",
            "ml",
            "faf",
            "tp",
            "fp",
            "fn",
            "ids",
            "frag",
            "tid",
            "lgd",
        ]

        log_info = ""
        for key in keys:
            if key in metrics:
                log_info += "{}: {:.4f}\n".format(key, metrics[key])
        for class_name in self.eval_track_configs.tracking_names:
            if class_name in metrics.get("label_metrics", {}):
                log_info += "{}: ".format(class_name)
                for metric_name, val in metrics["label_metrics"][
                    class_name
                ].items():
                    if metric_name in keys:
                        log_info += " {}={:.4f}".format(metric_name, val)
                log_info += "\n"
        logger.info(log_info)

        for key in keys:
            if key in self.ret.keys():
                self.ret[key] = metrics[key]

    @rank_zero_only
    def _evaluate(self, pred, gt):
        """Run tracking evaluation using nuscenes tracking algo."""
        self._patch_nuscenes_tracking_compat(
            tracking_names=self.eval_config.get("tracking_names", None)
        )
        start_time = time.time()

        TrackingMetricData.set_nelem(self.eval_config["num_thresholds"])

        # Build scene/timestamp structure from collected meta info
        sample2scene = {}
        sample2timestamp = {}
        for sample_token in gt.sample_tokens:
            if sample_token in self.sample_meta:
                meta = self.sample_meta[sample_token]
                scene = meta.get("scene", sample_token)
                ts = meta.get("timestamp", 0)
            else:
                scene = sample_token
                ts = 0
            sample2scene[sample_token] = scene
            sample2timestamp[sample_token] = ts

        # Create tracks from boxes
        tracks_gt = self._create_tracks(gt, sample2scene, sample2timestamp)
        tracks_pred = self._create_tracks(pred, sample2scene, sample2timestamp)

        # Replace prediction scores with average track scores
        for _, scene_tracks in tracks_pred.items():
            track_id_scores = {}
            for _, boxes in scene_tracks.items():
                for box in boxes:
                    if box.tracking_id not in track_id_scores:
                        track_id_scores[box.tracking_id] = []
                    track_id_scores[box.tracking_id].append(box.tracking_score)
            track_id_avg_scores = {}
            for tracking_id, scores in track_id_scores.items():
                track_id_avg_scores[tracking_id] = np.mean(scores)
            for _, boxes in scene_tracks.items():
                for box in boxes:
                    box.tracking_score = float(
                        track_id_avg_scores[box.tracking_id]
                    )

        # Interpolate GT and prediction tracks to fill holes.
        # This follows nuscenes official create_tracks logic where
        # GT boxes with 0 lidar points are interpolated across gaps.
        for scene_token in tracks_gt:
            tracks_gt[scene_token] = self._interpolate_tracks(
                tracks_gt[scene_token]
            )
        for scene_token in tracks_pred:
            tracks_pred[scene_token] = self._interpolate_tracks(
                tracks_pred[scene_token]
            )

        metrics_obj = TrackingMetrics(self.eval_track_configs)

        if self.verbose:
            print("Accumulating tracking metric data...")
        metric_data_list = TrackingMetricDataList()

        for class_name in self.eval_track_configs.tracking_names:
            curr_ev = TrackingEvaluation(
                tracks_gt,
                tracks_pred,
                class_name,
                self.eval_track_configs.dist_fcn_callable,
                self.eval_track_configs.dist_th_tp,
                self.eval_track_configs.min_recall,
                num_thresholds=self.eval_config["num_thresholds"],
                metric_worst=self.eval_track_configs.metric_worst,
                verbose=self.verbose,
            )
            md = curr_ev.accumulate()
            metric_data_list.set(class_name, md)

        if self.verbose:
            print("Calculating tracking metrics...")
        for class_name in self.eval_track_configs.tracking_names:
            md = metric_data_list[class_name]
            if np.all(np.isnan(md.mota)):
                best_thresh_idx = None
            else:
                best_thresh_idx = np.nanargmax(md.mota)

            if best_thresh_idx is not None:
                for metric_name in MOT_METRIC_MAP.values():
                    if metric_name == "":
                        continue
                    value = md.get_metric(metric_name)[best_thresh_idx]
                    metrics_obj.add_label_metric(
                        metric_name, class_name, value
                    )

            for metric_name in AVG_METRIC_MAP.keys():
                values = np.array(md.get_metric(AVG_METRIC_MAP[metric_name]))
                if np.all(np.isnan(values)):
                    value = np.nan
                else:
                    values[np.isnan(values)] = (
                        self.eval_track_configs.metric_worst[metric_name]
                    )
                    value = float(np.nanmean(values))
                metrics_obj.add_label_metric(metric_name, class_name, value)

        metrics_obj.add_runtime(time.time() - start_time)

        # Save metrics
        if self.verbose:
            print("Saving tracking metrics to: %s" % self.save_prefix)
        metrics_summary = metrics_obj.serialize()
        with open(
            os.path.join(self.save_prefix, "metrics_summary.json"), "w"
        ) as f:
            json.dump(metrics_summary, f, indent=2)
        with open(
            os.path.join(self.save_prefix, "metrics_details.json"), "w"
        ) as f:
            json.dump(metric_data_list.serialize(), f, indent=2)

        # Print high-level metrics
        print("AMOTA: %.4f" % (metrics_summary.get("amota", 0)))
        print("AMOTP: %.4f" % (metrics_summary.get("amotp", 0)))
        print("MOTA: %.4f" % (metrics_summary.get("mota", 0)))
        print("MOTP: %.4f" % (metrics_summary.get("motp", 0)))
        print("IDS: %d" % (metrics_summary.get("ids", 0)))
        print("Eval time: %.1fs" % metrics_summary.get("eval_time", 0))

        # Print per-class metrics
        if "label_metrics" in metrics_summary:
            print()
            print("Per-class tracking results:")
            for class_name in self.eval_track_configs.tracking_names:
                if class_name in metrics_summary["label_metrics"]:
                    cls_metrics = metrics_summary["label_metrics"][class_name]
                    print(
                        "%-20s\tAMOTA:%.4f\tAMOTP:%.4f\tIDS:%d"
                        % (
                            class_name,
                            cls_metrics.get("amota", 0),
                            cls_metrics.get("amotp", 0),
                            cls_metrics.get("ids", 0),
                        )
                    )

        return metrics_summary

    def _create_tracks(self, all_boxes, sample2scene, sample2timestamp):
        """Group boxes into tracks by scene and timestamp.

        Follows nuscenes create_tracks logic:
        1. Initialize all timestamps for every scene (even empty ones).
        2. Fill in boxes from all_boxes.
        3. Sort each scene's timestamps chronologically.
        """
        from collections import defaultdict

        tracks = defaultdict(lambda: defaultdict(list))

        # Step 1: Initialize all timestamps for every scene to
        # guarantee time-axis completeness (matches nuscenes behavior).
        for sample_token, scene_token in sample2scene.items():
            timestamp = sample2timestamp.get(sample_token, 0)
            tracks[scene_token][timestamp]  # noqa: touch key

        # Step 2: Fill in boxes from all_boxes.
        for sample_token in all_boxes.sample_tokens:
            scene_token = sample2scene.get(sample_token, sample_token)
            timestamp = sample2timestamp.get(sample_token, 0)
            tracks[scene_token][timestamp] = list(all_boxes[sample_token])

        # Step 3: Sort timestamps chronologically within each scene.
        sorted_tracks = {}
        for scene_token, scene_data in tracks.items():
            sorted_tracks[scene_token] = defaultdict(
                list, sorted(scene_data.items(), key=lambda kv: kv[0])
            )

        return dict(sorted_tracks)

    @staticmethod
    def _interpolate_tracks(tracks_by_timestamp):
        """Interpolate tracks to fill holes in GT or predictions.

        Ported from nuscenes.eval.tracking.loaders.interpolate_tracks.
        Linearly interpolates missing timestamps within each track's
        time span to handle gaps (e.g. GT boxes removed for 0 points).
        """
        from bisect import bisect
        from collections import defaultdict

        # Group tracks by id.
        tracks_by_id = defaultdict(list)
        track_timestamps_by_id = defaultdict(list)
        for timestamp, tracking_boxes in tracks_by_timestamp.items():
            for tracking_box in tracking_boxes:
                tracks_by_id[tracking_box.tracking_id].append(tracking_box)
                track_timestamps_by_id[tracking_box.tracking_id].append(
                    timestamp
                )

        timestamps = list(tracks_by_timestamp.keys())
        for timestamp in timestamps:
            for tracking_id, track in tracks_by_id.items():
                ts_list = track_timestamps_by_id[tracking_id]
                if (
                    ts_list[0] <= timestamp <= ts_list[-1]
                    and timestamp not in ts_list
                ):
                    # Ensure the timestamp key exists in the dict.
                    if timestamp not in tracks_by_timestamp:
                        tracks_by_timestamp[timestamp] = []

                    # Find closest boxes before and after.
                    right_ind = bisect(ts_list, timestamp)
                    left_ind = right_ind - 1
                    left_ts = ts_list[left_ind]
                    right_ts = ts_list[right_ind]
                    left_box = track[left_ind]
                    right_box = track[right_ind]

                    # Linear interpolation.
                    ratio = float(timestamp - left_ts) / (right_ts - left_ts)
                    interp_translation = tuple(
                        (1.0 - ratio) * np.array(left_box.translation)
                        + ratio * np.array(right_box.translation)
                    )
                    interp_size = tuple(
                        (1.0 - ratio) * np.array(left_box.size)
                        + ratio * np.array(right_box.size)
                    )
                    interp_rotation = pyquaternion.Quaternion.slerp(
                        q0=pyquaternion.Quaternion(left_box.rotation),
                        q1=pyquaternion.Quaternion(right_box.rotation),
                        amount=ratio,
                    ).elements
                    interp_velocity = tuple(
                        (1.0 - ratio) * np.array(left_box.velocity)
                        + ratio * np.array(right_box.velocity)
                    )
                    interp_score = (
                        (1.0 - ratio) * left_box.tracking_score
                        + ratio * right_box.tracking_score
                    )

                    interp_box = TrackingBox(
                        sample_token=right_box.sample_token,
                        translation=interp_translation,
                        size=interp_size,
                        rotation=interp_rotation,
                        velocity=interp_velocity,
                        tracking_id=right_box.tracking_id,
                        tracking_name=right_box.tracking_name,
                        tracking_score=interp_score,
                    )
                    tracks_by_timestamp[timestamp].append(interp_box)

        return tracks_by_timestamp


@OBJECT_REGISTRY.register
class PerceptionInferCacheProcess(EvalMetric):
    def __init__(
        self,
        name="perception_infer_save",
        save_dir=None,
        flush_interval=15000,
        postprocess_cfg=None,
        viz_cfg=None,
    ):
        self.rank, self.world_size = get_dist_info()
        self.save_dir = save_dir
        self.viz_cfg = viz_cfg
        self.postprocess_cfg = postprocess_cfg
        self.flush_interval = flush_interval

        if self.viz_cfg is not None:
            self._init_viz()

        super(PerceptionInferCacheProcess, self).__init__(name)

    def _init_states(self):
        self.token_to_idx = {}
        self._chunk_id = 0
        self._total_count = 0
        self._current_chunk_count = 0
        self._current_fp = None

    def _get_chunk_path(self, chunk_id):
        return os.path.join(
            self.save_dir, f"split_{self.rank}_chunk_{chunk_id}.pkl"
        )

    def _open_chunk(self):
        path = self._get_chunk_path(self._chunk_id)
        os.makedirs(self.save_dir, exist_ok=True)
        self._current_fp = open(path, "wb")
        self._current_chunk_count = 0

    def _close_chunk(self):
        if self._current_fp is not None:
            chunk_path = self._get_chunk_path(self._chunk_id)
            self._current_fp.close()
            self._current_fp = None
            print(f"Saved {chunk_path}")
            self._chunk_id += 1

    def _append_to_chunk(self, entry):
        if self._current_fp is None:
            self._open_chunk()
        offset = self._current_fp.tell()
        pickle.dump(entry, self._current_fp)
        self._current_fp.flush()
        self._current_chunk_count += 1
        self._total_count += 1
        return offset

    def _init_viz(self):
        try:
            from heal.visualize.nuscenes_map import NuscenesMapViz
            from heal.visualize.perception_multitask import MultitaskViz
        except ImportError:
            MultitaskViz = None
            NuscenesMapViz = None

        pc_range = self.viz_cfg.get("pc_range")
        bev_resolution = self.viz_cfg.get("bev_resolution")
        map_pc_range = self.viz_cfg.get("map_pc_range")
        bev_size = self.viz_cfg.get("bev_size")
        car_img_path = self.viz_cfg.get("car_img_path")
        score_thresh = self.viz_cfg.get("score_thresh", 0.0)
        save_path = self.viz_cfg.get("save_path", "viz/perception/cache")
        viz_num = self.viz_cfg.get("viz_num", 10)

        self._viz = None
        if MultitaskViz is not None:
            self._viz = MultitaskViz(
                om_viz=NuscenesMapViz(
                    pc_range=map_pc_range,
                    car_img_path=car_img_path,
                ),
                score_thresh=score_thresh,
                is_plot=True,
                bev_size=bev_size,
                use_bce=False,
                num_pics=11,
                pc_range=pc_range,
                bev_resolution=bev_resolution,
            )
        self._viz_save_path = save_path
        self._viz_num = viz_num
        self._viz_count = 0

    def _viz_frame(self, data, cur_batch_info):
        if self._viz_count >= self._viz_num:
            return
        if getattr(self, "_viz", None) is None:
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        imgs = data.get("imgs", data.get("img"))
        cam_names = data.get("cam_name")
        lidar2img = data.get("lidar2img")

        if imgs is None or lidar2img is None:
            return

        # Build imgs list for viz: list of {"name": str, "img": CHW tensor}
        from PIL import Image
        from torchvision.transforms import functional as F

        viz_imgs = []
        img_list = imgs
        name_list = cam_names[0]
        for img, name in zip(img_list, name_list):
            if isinstance(img, Image.Image):
                img = F.pil_to_tensor(img)
            elif isinstance(img, np.ndarray):
                if img.ndim == 3 and img.shape[2] in (3, 4):
                    img = torch.from_numpy(img).permute(2, 0, 1)
                else:
                    img = torch.from_numpy(img)
            if isinstance(img, torch.Tensor) and img.dim() == 3:
                # Reverse Normalize(mean=128, std=128): yuv = norm * 128 + 128
                img = img * 128.0 + 128.0
                # Reverse BgrToYuv444(rgb_input=True, swing="full"):
                # YUV -> RGB using the inverse of the BT.601 full-swing matrix
                # Forward: [Y, U-128, V-128] = M @ [R, G, B] / 256
                # where M = [[77,150,29],[-43,-84,127],[127,-106,-21]]
                M = torch.tensor(
                    [[77, 150, 29], [-43, -84, 127], [127, -106, -21]],
                    dtype=torch.float32,
                    device=img.device,
                )
                M_inv = torch.inverse(M / 256.0)
                yuv_centered = torch.stack(
                    [img[0], img[1] - 128, img[2] - 128], dim=0
                )
                c, h, w = yuv_centered.shape
                rgb = (M_inv @ yuv_centered.view(c, -1)).view(c, h, w)
                img = rgb.clamp(0, 255).to(torch.uint8)
                if img.is_cuda:
                    viz_imgs.append({"name": name, "img": img.cpu()})
                else:
                    viz_imgs.append({"name": name, "img": img})

        if len(viz_imgs) == 0:
            return

        if lidar2img.is_cuda:
            meta = {"lidar2img": lidar2img.cpu()}
        else:
            meta = {"lidar2img": lidar2img}

        # Build preds for viz from cur_batch_info
        preds_viz = {}

        # dynamic_objects_det: (N, 11) [x,y,z,w,l,h,yaw,vx,vy,score,label]
        dyn = cur_batch_info.get("dynamic_objects_det", {})
        if "bboxes" in dyn and dyn["bboxes"].shape[0] > 0:
            bboxes = dyn["bboxes"]  # (N, 10) [x,y,z,w,l,h,yaw,vx,vy,score]
            N = bboxes.shape[0]
            labels = dyn["labels"]
            if isinstance(labels, np.ndarray) and labels.ndim > 1:
                labels = labels.flatten()
            labels = np.asarray(labels, dtype=np.float32)
            if labels.shape[0] < N:
                labels = np.concatenate([labels, np.ones(N - labels.shape[0])])
            dyn_viz = np.zeros((N, 11), dtype=np.float32)
            dyn_viz[:, :10] = bboxes[:, :10]
            # swap w,l to match viz convention (l,w)
            dyn_viz[:, 3] = bboxes[:, 4]
            dyn_viz[:, 4] = bboxes[:, 3]
            # dyn_viz[:, 10] = labels[:N]
            dyn_viz[:, -1] = 1
            dyn_viz[:, 9] = dyn.get("scores", 1.0)  # score
            preds_viz["dynamic_objects_det"] = [dyn_viz]

        # static_objects_det: same format
        static = cur_batch_info.get("static_objects_det", {})
        if "bboxes" in static and static["bboxes"].shape[0] > 0:
            bboxes = static["bboxes"]
            N = bboxes.shape[0]
            labels = static.get("labels", np.ones(N, dtype=np.float32))
            if isinstance(labels, np.ndarray) and labels.ndim > 1:
                labels = labels.flatten()
            labels = np.asarray(labels, dtype=np.float32)
            if labels.shape[0] < N:
                labels = np.concatenate([labels, np.ones(N - labels.shape[0])])
            static_viz = np.zeros((N, 11), dtype=np.float32)
            static_viz[:, :10] = bboxes[:, :10]
            static_viz[:, 3] = bboxes[:, 4]
            static_viz[:, 4] = bboxes[:, 3]
            # static_viz[:, 10] = labels[:N]
            static_viz[:, -1] = 1
            static_viz[:, 9] = static.get("scores", 1.0)
            preds_viz["static_objects_det"] = [static_viz]

        # maptr -> om_det for viz
        maptr = cur_batch_info.get("maptr", {})
        if isinstance(maptr, dict) and "pts_3d" in maptr:
            pts = maptr["pts_3d"]
            maptr_labels = maptr["labels_3d"]
            maptr_scores = maptr["scores_3d"]

            score_thresh = self.viz_cfg.get("score_thresh")
            if score_thresh is not None:
                thresh_mask = maptr_scores > score_thresh
                tmp_score = score_thresh
                while thresh_mask.sum() == 0:
                    tmp_score *= 0.9
                    if tmp_score < 0.01:
                        thresh_mask = maptr_scores > -1
                        break
                    thresh_mask = maptr_scores >= tmp_score
            pts = pts[thresh_mask]
            maptr_labels = maptr_labels[thresh_mask]
            maptr_scores = maptr_scores[thresh_mask]

            pts_3d_list = []
            labels_3d_list = []
            for i in range(pts.shape[0]):
                xy = pts[i, :, :2]
                z_col = np.zeros((xy.shape[0], 1), dtype=np.float32)
                pts_3d = np.concatenate([xy, z_col], axis=-1)
                pts_3d_list.append(torch.from_numpy(pts_3d))
                if maptr_labels is not None:
                    labels_3d_list.append(int(maptr_labels[i]))
                else:
                    labels_3d_list.append(0)
            maptr_pred = {
                "labels_3d": torch.tensor(labels_3d_list, dtype=torch.long),
                "pts_3d": pts_3d_list,
            }
            preds_viz["om_det"] = [maptr_pred]
        else:
            lanes_data = maptr  # (70, 20, 12)
            valid = lanes_data[:, :, 0:2].abs().sum(dim=(1, 2)) > 0
            lanes_valid = lanes_data[valid]
            pts_3d_list = []
            labels_3d_list = []
            for i in range(lanes_valid.shape[0]):
                xy = lanes_valid[i, :, 0:2]
                z_col = torch.zeros(xy.shape[0], 1)
                pts = torch.cat([xy, z_col], dim=-1)  # (20, 3)
                pts_3d_list.append(pts)
                labels_3d_list.append(0)
            maptr_pred = {
                "labels_3d": torch.tensor(labels_3d_list, dtype=torch.long),
                "pts_3d": pts_3d_list,
            }
            preds_viz["om_det"] = [maptr_pred]

        if len(preds_viz) > 0:
            self._viz(
                viz_imgs,
                None,
                preds_viz,
                meta,
                save_path=self._viz_save_path,
            )
            plt.close("all")
            self._viz_count += 1

    def update(self, data, pred):
        # The model (NavsimPerceptionCacheWrapper) returns flattened
        # per-frame results.
        #   pred["sample_tokens"]       – list[str], one per frame
        #   pred["scenes"]              – list[str]
        #   pred["dynamic_objects_det"] – list[dict]
        #   pred["static_objects_det"]  – list[dict]
        #   pred["maptr"]               – list[dict]
        #   pred["bev_embeds"]          – list[np.ndarray | None]
        sample_tokens = pred["sample_tokens"]
        scenes = pred["scenes"]
        dyn_infos = pred["dynamic_objects_det"]
        static_infos = pred["static_objects_det"]
        maptr_infos = pred["maptr"]
        occ_infos = pred["occ"]
        bev_embeds = pred.get("bev_embeds", [None] * len(sample_tokens))

        for i in range(len(sample_tokens)):
            sample_token = sample_tokens[i]
            scene = scenes[i]
            dyn_info = dyn_infos[i]
            static_info = static_infos[i]
            maptr_info = maptr_infos[i]
            occ_info = occ_infos[i]
            bev_embed = bev_embeds[i]

            idx = self._current_chunk_count
            cur_batch_info = {
                "sample_token": sample_token,
                "scene": scene,
                "dynamic_objects_det": {},
                "static_objects_det": {},
                "maptr": {},
            }

            if bev_embed is not None:
                if isinstance(bev_embed, np.ndarray):
                    cur_batch_info["bev_embed"] = bev_embed.copy()
            if "dynamic_bev_embed" in dyn_info:
                dynamic_bev_embed = dyn_info["dynamic_bev_embed"]
                if isinstance(dynamic_bev_embed, torch.Tensor):
                    cur_batch_info["dynamic_bev_embed"] = (
                        dynamic_bev_embed.detach().cpu().numpy().copy()
                    )
                else:
                    cur_batch_info["dynamic_bev_embed"] = dynamic_bev_embed
            if "maptr_bev_embed" in maptr_info:
                maptr_bev_embed = maptr_info["maptr_bev_embed"]
                if isinstance(maptr_bev_embed, torch.Tensor):
                    cur_batch_info["maptr_bev_embed"] = (
                        maptr_bev_embed.detach().cpu().numpy().copy()
                    )
                else:
                    cur_batch_info["maptr_bev_embed"] = maptr_bev_embed
            if "occ_bev_embed" in occ_info:
                occ_bev_embed = occ_info["occ_bev_embed"]
                if isinstance(occ_bev_embed, torch.Tensor):
                    cur_batch_info["occ_bev_embed"] = (
                        occ_bev_embed.detach().cpu().numpy().copy()
                    )
                else:
                    cur_batch_info["occ_bev_embed"] = occ_bev_embed

            # dynamic_objects_det
            if isinstance(dyn_info, dict):
                for k, v in dyn_info.items():
                    if k != "dynamic_bev_embed":
                        if isinstance(v, torch.Tensor):
                            cur_batch_info["dynamic_objects_det"][k] = (
                                v.detach().cpu().numpy().copy()
                            )
                        elif isinstance(v, np.ndarray):
                            cur_batch_info["dynamic_objects_det"][k] = v.copy()
                        else:
                            cur_batch_info["dynamic_objects_det"][k] = v

            # static_objects_det
            if isinstance(static_info, dict):
                for k, v in static_info.items():
                    if isinstance(v, torch.Tensor):
                        cur_batch_info["static_objects_det"][k] = (
                            v.detach().cpu().numpy().copy()
                        )
                    elif isinstance(v, np.ndarray):
                        cur_batch_info["static_objects_det"][k] = v.copy()
                    else:
                        cur_batch_info["static_objects_det"][k] = v

            # maptr
            if isinstance(maptr_info, dict) and len(maptr_info) > 0:
                for key, v in maptr_info.items():
                    if k != "maptr_bev_embed":
                        if isinstance(v, torch.Tensor):
                            cur_batch_info["maptr"][key] = (
                                v.detach().cpu().numpy().copy()
                            )
                        elif isinstance(v, np.ndarray):
                            cur_batch_info["maptr"][key] = v.copy()
                        else:
                            cur_batch_info["maptr"][key] = v

            if self.postprocess_cfg is not None:
                if self.viz_cfg is not None:
                    return_perception_predict = True
                else:
                    return_perception_predict = False
                (
                    perception_postprocess,
                    perception_out,
                ) = self.perception_postprocess(
                    copy.deepcopy(cur_batch_info["dynamic_objects_det"]),
                    copy.deepcopy(cur_batch_info["static_objects_det"]),
                    copy.deepcopy(cur_batch_info["maptr"]),
                    return_perception_predict=return_perception_predict,
                )
                cur_batch_info["perception_postprocess"] = (
                    perception_postprocess
                )

            if self.viz_cfg is not None:
                if self.postprocess_cfg is not None:
                    self._viz_frame(data, perception_out)
                else:
                    self._viz_frame(data, cur_batch_info)

            offset = self._append_to_chunk(cur_batch_info)
            self.token_to_idx[sample_token] = (
                self.rank,
                self._chunk_id,
                idx,
                offset,
            )

            if self._current_chunk_count >= self.flush_interval:
                self._close_chunk()

    def _flush_chunk(self):
        if self._current_fp is not None:
            self._close_chunk()

    def _save_cache_metas(self):
        """Gather token_to_idx from all ranks and save cache_metas.pkl."""
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            all_token_to_idx = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(
                all_token_to_idx,
                self.token_to_idx,
            )
            if self.rank == 0:
                merged = {}
                for t2i in all_token_to_idx:
                    merged.update(t2i)
                meta_file = os.path.join(self.save_dir, "cache_metas.pkl")
                with open(meta_file, "wb") as f:
                    pickle.dump(merged, f)
                print(f"Saved cache_metas.pkl with {len(merged)} entries")
        else:
            meta_file = os.path.join(self.save_dir, "cache_metas.pkl")
            with open(meta_file, "wb") as f:
                pickle.dump(self.token_to_idx, f)
            print(
                f"Saved cache_metas.pkl with "
                f"{len(self.token_to_idx)} entries"
            )

    def perception_postprocess(
        self, dyn_dict, static_dict, maptr_dict, return_perception_predict
    ):
        dynamic_objects_score_threshold = self.postprocess_cfg.get(
            "dynamic_objects_score_threshold"
        )
        static_objects_score_threshold = self.postprocess_cfg.get(
            "static_objects_score_threshold"
        )
        maptr_score_threshold = self.postprocess_cfg.get(
            "maptr_score_threshold"
        )
        agent_num = self.postprocess_cfg.get("agent_num", 32)
        static_objects_num = self.postprocess_cfg.get("static_objects_num", 5)
        model_time_len = self.postprocess_cfg.get("model_time_len", 4)
        max_ped_bike = self.postprocess_cfg.get("max_ped_bike", 10)
        lane_num = self.postprocess_cfg.get("lane_num", 70)
        lane_len = self.postprocess_cfg.get("lane_len", 20)
        dyn_dict["scores"] = _to_tensor(
            dyn_dict["scores"], dtype=torch.float32
        )
        dyn_dict["bboxes"] = _to_tensor(
            dyn_dict["bboxes"], dtype=torch.float32
        )
        dyn_dict["labels"] = _to_tensor(dyn_dict["labels"], dtype=torch.long)
        static_dict["scores"] = _to_tensor(
            static_dict["scores"], dtype=torch.float32
        )
        static_dict["bboxes"] = _to_tensor(
            static_dict["bboxes"], dtype=torch.float32
        )
        static_dict["labels"] = _to_tensor(
            static_dict["labels"], dtype=torch.long
        )
        maptr_dict["scores_3d"] = _to_tensor(
            maptr_dict["scores_3d"], dtype=torch.float32
        )
        maptr_dict["labels_3d"] = _to_tensor(
            maptr_dict["labels_3d"], dtype=torch.long
        )
        maptr_dict["pts_3d"] = _to_tensor(
            maptr_dict["pts_3d"], dtype=torch.float32
        )
        if dynamic_objects_score_threshold is not None:
            mask = dyn_dict["scores"] >= dynamic_objects_score_threshold
            dyn_dict["scores"] = dyn_dict["scores"][mask]
            dyn_dict["bboxes"] = dyn_dict["bboxes"][mask]
            dyn_dict["labels"] = dyn_dict["labels"][mask]
        if static_objects_score_threshold is not None:
            mask = static_dict["scores"] >= static_objects_score_threshold
            static_dict["scores"] = static_dict["scores"][mask]
            static_dict["bboxes"] = static_dict["bboxes"][mask]
            static_dict["labels"] = static_dict["labels"][mask]
        do_nms = self.postprocess_cfg.get("do_nms", False)
        if do_nms:
            nms_thresh = self.postprocess_cfg.get("nms_thresh", 0.5)
            nms_num_max = self.postprocess_cfg.get("nms_num_max", 500)
            # nms_bev_multiclass requires CUDA tensors, move to GPU first
            nms_device = torch.device("cuda")
            # dynamic objects: bboxes [N,10] -> [N,5] for NMS
            # [x, y, z, w, l, h, yaw, vx, vy, score]
            # -> [x, y, l, w, yaw]
            if dyn_dict["bboxes"].shape[0] > 0:
                dyn_bboxes_gpu = dyn_dict["bboxes"].to(nms_device)
                dyn_scores_gpu = dyn_dict["scores"].to(nms_device)
                dyn_labels_gpu = dyn_dict["labels"].to(nms_device)
                dyn_bev_boxes = torch.stack(
                    [
                        dyn_bboxes_gpu[:, 0],  # x
                        dyn_bboxes_gpu[:, 1],  # y
                        dyn_bboxes_gpu[:, 4],  # w
                        dyn_bboxes_gpu[:, 3],  # l
                        dyn_bboxes_gpu[:, 6],  # yaw
                    ],
                    dim=-1,
                )
                dyn_num_classes = 3  # vehicle, pedestrian, bicycle
                dyn_keep = nms_bev_multiclass(
                    dyn_bev_boxes,
                    dyn_scores_gpu,
                    dyn_labels_gpu,
                    num_classes=dyn_num_classes,
                    thresh=nms_thresh,
                    num_max=nms_num_max,
                )
                dyn_dict["scores"] = dyn_scores_gpu[dyn_keep].cpu()
                dyn_dict["bboxes"] = dyn_bboxes_gpu[dyn_keep].cpu()
                dyn_dict["labels"] = dyn_labels_gpu[dyn_keep].cpu()
            # static objects: bboxes [N,10] -> [N,5] for NMS
            if static_dict["bboxes"].shape[0] > 0:
                static_bboxes_gpu = static_dict["bboxes"].to(nms_device)
                static_scores_gpu = static_dict["scores"].to(nms_device)
                static_labels_gpu = static_dict["labels"].to(nms_device)
                static_bev_boxes = torch.stack(
                    [
                        static_bboxes_gpu[:, 0],  # x
                        static_bboxes_gpu[:, 1],  # y
                        static_bboxes_gpu[:, 4],  # w
                        static_bboxes_gpu[:, 3],  # l
                        static_bboxes_gpu[:, 6],  # yaw
                    ],
                    dim=-1,
                )
                static_num_classes = (
                    4  # czone_sign, barrier, traffic_cone, generic_object
                )
                static_keep = nms_bev_multiclass(
                    static_bev_boxes,
                    static_scores_gpu,
                    static_labels_gpu,
                    num_classes=static_num_classes,
                    thresh=nms_thresh,
                    num_max=nms_num_max,
                )
                static_dict["scores"] = static_scores_gpu[static_keep].cpu()
                static_dict["bboxes"] = static_bboxes_gpu[static_keep].cpu()
                static_dict["labels"] = static_labels_gpu[static_keep].cpu()
        if maptr_score_threshold is not None:
            thresh_mask = maptr_dict["scores_3d"] > maptr_score_threshold
            tmp_score = maptr_score_threshold
            while thresh_mask.sum() == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = maptr_dict["scores_3d"] > -1
                    break
                thresh_mask = maptr_dict["scores_3d"] >= tmp_score
            maptr_dict["scores_3d"] = maptr_dict["scores_3d"][thresh_mask]
            maptr_dict["labels_3d"] = maptr_dict["labels_3d"][thresh_mask]
            maptr_dict["pts_3d"] = maptr_dict["pts_3d"][thresh_mask]
        if return_perception_predict:
            (
                neighbor_agents_past,
                dynamic_objects_perception,
            ) = perception_dyn_to_planner_agents(
                [dyn_dict],
                agent_num=agent_num,
                model_time_len=model_time_len,
                max_ped_bike=max_ped_bike,
                return_perception_predict=True,
            )
            neighbor_agents_past = neighbor_agents_past[0]
            for key in dynamic_objects_perception.keys():
                dynamic_objects_perception[key] = dynamic_objects_perception[
                    key
                ][0]

            (
                static_objects,
                static_objects_perception,
            ) = perception_static_to_planner_static(
                [static_dict],
                static_objects_num=static_objects_num,
                return_perception_predict=True,
            )
            static_objects = static_objects[0]
            for key in static_objects_perception.keys():
                static_objects_perception[key] = static_objects_perception[
                    key
                ][0]
        else:
            neighbor_agents_past = perception_dyn_to_planner_agents(
                [dyn_dict],
                agent_num=agent_num,
                model_time_len=model_time_len,
                max_ped_bike=max_ped_bike,
            )[
                0
            ]  # (agent_num, model_time_len, 11)

            static_objects = perception_static_to_planner_static(
                [static_dict],
                static_objects_num=static_objects_num,
            )[
                0
            ]  # (static_objects_num, 10)

        # Convert maptr -> lanes, lanes_speed_limit, lanes_has_speed_limit
        lanes, lanes_sl, lanes_hsl = perception_maptr_to_planner_lanes(
            [maptr_dict],
            lane_num=lane_num,
            lane_len=lane_len,
        )
        lanes = lanes[0]  # (lane_num, lane_len, 12)
        lanes_sl = lanes_sl[0]  # (lane_num, 1)
        lanes_hsl = lanes_hsl[0].float()  # (lane_num, 1)
        result = {
            "neighbor_agents_past": neighbor_agents_past,
            "static_objects": static_objects,
            "lanes": lanes,
            "lanes_speed_limit": lanes_sl,
            "lanes_has_speed_limit": lanes_hsl,
        }
        if return_perception_predict:
            perception_out = {
                "dynamic_objects_det": dynamic_objects_perception,
                "static_objects_det": static_objects_perception,
                "maptr": lanes,
            }
            return result, perception_out
        else:
            return result, None

    def compute(self):
        # Flush any remaining data that hasn't been written yet
        self._flush_chunk()

        if self.save_dir is not None:
            print(
                f"Rank {self.rank} saved results to "
                f"{self.save_dir}, total {self._total_count} samples"
            )
            self._save_cache_metas()

        # Only gather counts, not the full data, to avoid OOM
        local_count = self._total_count
        all_counts = [None for _ in range(self.world_size)]
        all_gather_object(all_counts, local_count)
        all_data = sum(all_counts)
        if self.rank == 0:
            print(f"All gather number: {all_data} samples")

        return int(all_data)

    def get(self) -> Tuple[Union[str, List[str]], Union[float, List[float]]]:
        """Get current evaluation result.

        To skip the synchronization among devices, please override this method
        and calculate results without calling `self.compute()`.

        Returns:
           names: Name of the metrics.
           values: Value of the evaluations.
        """

        values = self.compute()
        return self.name, values
