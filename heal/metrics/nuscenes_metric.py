# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import json
import logging
import os
from typing import Optional, Sequence

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
from tqdm import tqdm

from heal.core.nms.box3d_nms import nms_bev_multiclass

try:
    import pyquaternion
    from nuscenes.eval.common.config import config_factory as track_configs
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.common.loaders import (
        add_center_dist,
        filter_eval_boxes,
        load_prediction,
    )
    from nuscenes.eval.common.utils import Quaternion
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.data_classes import DetectionBox
    from nuscenes.eval.detection.evaluate import DetectionEval
    from nuscenes.eval.tracking.data_classes import TrackingBox
    from nuscenes.eval.tracking.evaluate import TrackingEval
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import Box as NuScenesBox
    from nuscenes.utils.splits import create_splits_scenes
except ImportError:
    Quaternion = None
    config_factory = None
    DetectionEval = None
    NuScenes = None
    NuScenesBox = None
    pyquaternion = None
    TrackingEval = None
    track_configs = None
    load_prediction = None
    DetectionBox = None
    TrackingBox = None
    EvalBoxes = None
    add_center_dist = None
    filter_eval_boxes = None
    create_splits_scenes = None

logger = logging.getLogger(__name__)

__all__ = ["NuscenesMetric", "NuscenesMonoMetric"]


CLASSES = (
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
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
    "barrier": "",
    "traffic_cone": "",
}

_VEHICLE_MOVING_NAMES = frozenset(
    {"car", "construction_vehicle", "bus", "truck", "trailer"}
)
_CYCLE_WITH_RIDER_NAMES = frozenset({"bicycle", "motorcycle"})


def _nusc_detection_dict(sample_token, box, classes):
    name = classes[box.label]
    return {
        "sample_token": sample_token,
        "translation": box.center.tolist(),
        "size": box.wlh.tolist(),
        "rotation": box.orientation.elements.tolist(),
        "velocity": box.velocity[:2].tolist(),
        "detection_name": name,
        "detection_score": box.score,
        "attribute_name": _attribute_from_velocity(name, box.velocity),
    }


def _attribute_from_velocity(name: str, velocity) -> str:
    speed = np.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
    if speed > 0.2:
        if name in _VEHICLE_MOVING_NAMES:
            return "vehicle.moving"
        if name in _CYCLE_WITH_RIDER_NAMES:
            return "cycle.with_rider"
        return DefaultAttribute[name]
    if name == "pedestrian":
        return "pedestrian.standing"
    if name == "bus":
        return "vehicle.stopped"
    return DefaultAttribute[name]


class NuScenesEval(DetectionEval):
    def __init__(
        self,
        nusc,
        config,
        result_path,
        eval_set,
        output_dir,
        verbose,
        detection_mapping=None,
    ):

        if detection_mapping is not None:
            self.detection_mapping = detection_mapping
        else:
            self.detection_mapping = {
                "movable_object.barrier": "barrier",
                "vehicle.bicycle": "bicycle",
                "vehicle.bus.bendy": "bus",
                "vehicle.bus.rigid": "bus",
                "vehicle.car": "car",
                "vehicle.construction": "construction_vehicle",
                "vehicle.motorcycle": "motorcycle",
                "human.pedestrian.adult": "pedestrian",
                "human.pedestrian.child": "pedestrian",
                "human.pedestrian.construction_worker": "pedestrian",
                "human.pedestrian.police_officer": "pedestrian",
                "movable_object.trafficcone": "traffic_cone",
                "vehicle.trailer": "trailer",
                "vehicle.truck": "truck",
            }

        self.nusc = nusc
        self.result_path = result_path
        self.eval_set = eval_set
        self.output_dir = output_dir
        self.verbose = verbose
        self.cfg = config

        # Check result file exists.
        assert os.path.exists(
            result_path
        ), "Error: The result file does not exist!"

        # Make dirs.
        self.plot_dir = os.path.join(self.output_dir, "plots")
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.isdir(self.plot_dir):
            os.makedirs(self.plot_dir)

        # Load data.
        if verbose:
            print("Initializing nuScenes detection evaluation")
        self.pred_boxes, self.meta = load_prediction(
            self.result_path,
            self.cfg.max_boxes_per_sample,
            DetectionBox,
            verbose=verbose,
        )
        all_sample_tokens = self.pred_boxes.sample_tokens
        self.gt_boxes = self.load_gt(
            self.nusc,
            self.eval_set,
            DetectionBox,
            verbose=verbose,
            sample_tokens=all_sample_tokens,
        )

        assert set(self.pred_boxes.sample_tokens) == set(
            self.gt_boxes.sample_tokens
        ), "Samples in split doesn't match samples in predictions."

        # Add center distances.
        self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
        self.gt_boxes = add_center_dist(nusc, self.gt_boxes)

        # Filter boxes (distance, points per box, etc.).
        if verbose:
            print("Filtering predictions")
        self.pred_boxes = filter_eval_boxes(
            nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose
        )
        if verbose:
            print("Filtering ground truth annotations")
        self.gt_boxes = filter_eval_boxes(
            nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose
        )

        self.sample_tokens = self.gt_boxes.sample_tokens

    def load_gt(self, nusc, eval_split, box_cls, verbose, sample_tokens=None):
        # Init.
        if box_cls == DetectionBox:
            attribute_map = {a["token"]: a["name"] for a in nusc.attribute}

        if verbose:
            print(
                "Loading annotations for {} split from nuScenes version: {}".format(  # noqa
                    eval_split, nusc.version
                )
            )
        # Read out all sample_tokens in DB.
        sample_tokens_all = [s["token"] for s in nusc.sample]
        assert len(sample_tokens_all) > 0, "Error: Database has no samples!"

        # Only keep samples from this split.
        splits = create_splits_scenes()

        # Check compatibility of split with nusc_version.
        version = nusc.version
        if eval_split in {"train", "val", "train_detect", "train_track"}:
            assert version.endswith(
                "trainval"
            ), "Error: Requested split {} which is not compatible with NuScenes version {}".format(  # noqa
                eval_split, version
            )
        elif eval_split in {"mini_train", "mini_val"}:
            assert version.endswith(
                "mini"
            ), "Error: Requested split {} which is not compatible with NuScenes version {}".format(  # noqa
                eval_split, version
            )
        elif eval_split == "test":
            assert version.endswith(
                "test"
            ), "Error: Requested split {} which is not compatible with NuScenes version {}".format(  # noqa
                eval_split, version
            )
        else:
            raise ValueError(
                "Error: Requested split {} which this function cannot map to the correct NuScenes version.".format(  # noqa
                    eval_split
                )
            )

        if eval_split == "test":
            # Check that you aren't trying to cheat :).
            assert (
                len(nusc.sample_annotation) > 0
            ), "Error: You are trying to evaluate on the test set but you do not have the annotations!"  # noqa

        if sample_tokens is None:
            sample_tokens = []
            for sample_token in sample_tokens_all:
                scene_token = nusc.get("sample", sample_token)["scene_token"]
                scene_record = nusc.get("scene", scene_token)
                if scene_record["name"] in splits[eval_split]:
                    sample_tokens.append(sample_token)

        all_annotations = EvalBoxes()

        # Load annotations and filter predictions and annotations.
        tracking_id_set = set()
        for sample_token in tqdm(sample_tokens, leave=verbose):

            sample = nusc.get("sample", sample_token)
            sample_annotation_tokens = sample["anns"]

            sample_boxes = []
            for sample_annotation_token in sample_annotation_tokens:

                sample_annotation = nusc.get(
                    "sample_annotation", sample_annotation_token
                )
                if box_cls == DetectionBox:
                    # Get label name in detection task and filter unused labels.  # noqa
                    detection_name = self.category_to_detection_name(
                        sample_annotation["category_name"]
                    )
                    if detection_name is None:
                        continue

                    # Get attribute_name.
                    attr_tokens = sample_annotation["attribute_tokens"]
                    attr_count = len(attr_tokens)
                    if attr_count == 0:
                        attribute_name = ""
                    elif attr_count == 1:
                        attribute_name = attribute_map[attr_tokens[0]]
                    else:
                        raise Exception(
                            "Error: GT annotations must not have more than one attribute!"  # noqa
                        )

                    sample_boxes.append(
                        box_cls(
                            sample_token=sample_token,
                            translation=sample_annotation["translation"],
                            size=sample_annotation["size"],
                            rotation=sample_annotation["rotation"],
                            velocity=nusc.box_velocity(
                                sample_annotation["token"]
                            )[:2],
                            num_pts=sample_annotation["num_lidar_pts"]
                            + sample_annotation["num_radar_pts"],
                            detection_name=detection_name,
                            detection_score=-1.0,
                            attribute_name=attribute_name,
                        )
                    )
                elif box_cls == TrackingBox:
                    # Use nuScenes token as tracking id.
                    tracking_id = sample_annotation["instance_token"]
                    tracking_id_set.add(tracking_id)

                    # Get label name in detection task and filter unused labels.  # noqa
                    # Import locally to avoid errors when motmetrics package is not installed.  # noqa
                    from nuscenes.eval.tracking.utils import (
                        category_to_tracking_name,
                    )

                    tracking_name = category_to_tracking_name(
                        sample_annotation["category_name"]
                    )
                    if tracking_name is None:
                        continue

                    sample_boxes.append(
                        box_cls(
                            sample_token=sample_token,
                            translation=sample_annotation["translation"],
                            size=sample_annotation["size"],
                            rotation=sample_annotation["rotation"],
                            velocity=nusc.box_velocity(
                                sample_annotation["token"]
                            )[:2],
                            num_pts=sample_annotation["num_lidar_pts"]
                            + sample_annotation["num_radar_pts"],
                            tracking_id=tracking_id,
                            tracking_name=tracking_name,
                            tracking_score=-1.0,
                        )
                    )
                else:
                    raise NotImplementedError(
                        "Error: Invalid box_cls %s!" % box_cls
                    )

            all_annotations.add_boxes(sample_token, sample_boxes)

        if verbose:
            print(
                "Loaded ground truth annotations for {} samples.".format(
                    len(all_annotations.sample_tokens)
                )
            )

        return all_annotations

    def category_to_detection_name(self, category_name):
        if category_name in self.detection_mapping:
            return self.detection_mapping[category_name]
        else:
            return None


@OBJECT_REGISTRY.register
class NuscenesMetric(EvalMetric):
    """Evaluation Nuscenes Detection.

    Args:
        name: Name of this metric instance for display.
        data_root: Data path of nuscenes data.
        version: Version of nuscenes data.
                 Choosen from ['v1.0-mini', 'v1.0-tranval'].
        save_prefix: Path to save result.
        verbose: Wether output verbose log.
        eval_version: Eval version.
        use_lidar: Whether use lidar bbox.
        classes: List of class name.
        use_ddp: Whether use ddp to eval metric.
    """

    @require_packages(
        "nuscenes",
        "pyquaternion",
        raise_msg="Please `pip3 install nuscenes-devkit pyquaternion`",
    )
    def __init__(
        self,
        name: str = "NuscenesMetric",
        data_root: str = "",
        version: str = "v1.0-mini",
        save_prefix: str = "./WORKSPACE/results",
        verbose: bool = True,
        eval_version: str = "detection_cvpr_2019",
        use_lidar: bool = False,
        classes: Sequence[str] = None,
        use_ddp: bool = True,
        trans_lidar_dim: bool = False,
        trans_lidar_rot: bool = True,
        meta_key="meta",
        lidar_key="lidar2ego",
        detection_mapping: Optional[dict] = None,
    ):
        super(NuscenesMetric, self).__init__(name)
        self.verbose = verbose
        self.save_prefix = save_prefix
        self.use_ddp = use_ddp
        self.meta_key = meta_key
        self.lidar_key = lidar_key
        self.detection_mapping = detection_mapping

        self.trans_lidar_dim = trans_lidar_dim
        self.trans_lidar_rot = trans_lidar_rot

        if self.use_ddp:
            self.res_path = os.path.join(self.save_prefix, "results_nusc.json")
        else:
            global_rank, _ = get_dist_info()
            self.res_path = os.path.join(
                self.save_prefix, "results_nusc_{}.json".format(global_rank)
            )
        self.version = version
        self.data_root = data_root
        self.eval_detection_configs = config_factory(eval_version)
        if classes is None:
            self.CLASSES = CLASSES
        else:
            self.CLASSES = classes
            custom_class_range = {}
            for class_name in classes:
                custom_class_range[class_name] = (
                    self.eval_detection_configs.class_range[class_name]
                )
            self.eval_detection_configs.class_range = custom_class_range
            self.eval_detection_configs.class_names = custom_class_range.keys()

        self.ret = ["NDS", 0.0]
        self.nusc_annos = {}
        self.use_lidar = use_lidar

    def _output_to_nusc_box(self, bboxes):
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
        ego2global_translation = meta["ego2global_translation"]
        ego2global_rotation = meta["ego2global_rotation"]
        bbox.rotate(Quaternion(ego2global_rotation))
        bbox.translate(ego2global_translation)
        return bbox

    def compute(self):
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
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            return None
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info["ego2global_rotation"]))
        box.translate(np.array(info["ego2global_translation"]))

        return box

    def update(
        self,
        meta,
        pred_bboxes,
    ):
        meta = meta[self.meta_key]
        for m, bboxes in zip(meta, pred_bboxes):
            if self.use_lidar:
                boxes_list = self._output_to_nusc_box_lidar(bboxes)
            else:
                boxes_list = self._output_to_nusc_box(bboxes)
            sample_token = m["sample_token"]
            annos = []
            for box in boxes_list:
                if self.use_lidar:
                    box = self._lidar_nusc_box_to_global(
                        m, box, self.CLASSES, self.eval_detection_configs
                    )
                else:
                    box = self._ego_to_global(box, m)
                if box is None:
                    continue
                annos.append(
                    _nusc_detection_dict(sample_token, box, self.CLASSES)
                )
            self.nusc_annos[sample_token] = annos

    def _gather(self):
        global_rank, global_world_size = get_dist_info()
        global_output = [None for _ in range(global_world_size)]
        all_gather_object(global_output, self.nusc_annos)
        return global_output

    def reset(self):
        self.nusc_annos = {}
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
            self._get()
        return self.ret[0], self.ret[1]

    def fast_get(self):
        """Return cached NDS without re-running official nuScenes evaluation."""  # noqa: E501
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
    def _get(self):
        self._dump()
        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
        }
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=self.verbose
        )

        nusc_eval = NuScenesEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=self.res_path,
            eval_set=eval_set_map[self.version],
            output_dir=self.save_prefix,
            verbose=self.verbose,
            detection_mapping=self.detection_mapping,
        )
        nusc_eval.main(render_curves=False)

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


@OBJECT_REGISTRY.register
class NuscenesMonoMetric(NuscenesMetric):
    """Evaluation Nuscenes Detection for mono.

    Args:
        nms_threshold: NMS threshold for detection under same sample.
    """

    @require_packages(
        "nuscenes",
        "pyquaternion",
        raise_msg="Please `pip3 install nuscenes-devkit pyquaternion`",
    )
    def __init__(
        self, nms_threshold: float = 0.05, use_cpu: bool = False, **kwargs
    ):
        super(NuscenesMonoMetric, self).__init__(**kwargs)
        self.nms_threshold = nms_threshold
        self.use_cpu = use_cpu

    def _cam_to_ego(self, bbox, meta):
        cam2ego_translation = meta["sensor2ego_translation"]
        cam2ego_rotation = meta["sensor2ego_rotation"]
        bbox.rotate(Quaternion(cam2ego_rotation))
        bbox.translate(cam2ego_translation)
        return bbox

    def _get_modality(self):
        modality = {
            "use_camera": True,
            "use_lidar": False,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }
        return modality

    def get_attr_name(self, attr_idx, label_name):
        """Get attribute from predicted index.

        This is a workaround to predict attribute when the predicted velocity
        is not reliable. We map the predicted attribute index to the one
        in the attribute set. If it is consistent with the category, we will
        keep it. Otherwise, we will use the default attribute.

        Args:
            attr_idx (int): Attribute index.
            label_name (str): Predicted category name.

        Returns:
            str: Predicted attribute name.
        """
        # TODO: Simplify the variable name
        AttrMapping_rev2 = [
            "cycle.with_rider",
            "cycle.without_rider",
            "pedestrian.moving",
            "pedestrian.standing",
            "pedestrian.sitting_lying_down",
            "vehicle.moving",
            "vehicle.parked",
            "vehicle.stopped",
            "None",
        ]
        if (
            label_name == "car"
            or label_name == "bus"
            or label_name == "truck"
            or label_name == "trailer"
            or label_name == "construction_vehicle"
        ):
            if (
                AttrMapping_rev2[attr_idx] == "vehicle.moving"
                or AttrMapping_rev2[attr_idx] == "vehicle.parked"
                or AttrMapping_rev2[attr_idx] == "vehicle.stopped"
            ):
                return AttrMapping_rev2[attr_idx]
            else:
                return DefaultAttribute[label_name]
        elif label_name == "pedestrian":
            if (
                AttrMapping_rev2[attr_idx] == "pedestrian.moving"
                or AttrMapping_rev2[attr_idx] == "pedestrian.standing"
                or AttrMapping_rev2[attr_idx]
                == "pedestrian.sitting_lying_down"
            ):
                return AttrMapping_rev2[attr_idx]
            else:
                return DefaultAttribute[label_name]
        elif label_name == "bicycle" or label_name == "motorcycle":
            if (
                AttrMapping_rev2[attr_idx] == "cycle.with_rider"
                or AttrMapping_rev2[attr_idx] == "cycle.without_rider"
            ):
                return AttrMapping_rev2[attr_idx]
            else:
                return DefaultAttribute[label_name]
        else:
            return DefaultAttribute[label_name]

    def _output_to_nusc_box(self, bboxes):
        box_list = []
        attrs = []
        for bbox in bboxes:
            bbox = bbox.cpu().numpy()
            center = bbox[:3]
            dims = bbox[3:6]
            yaw = bbox[6]
            q1 = Quaternion(axis=[0, 0, 1], radians=yaw)
            q2 = Quaternion(axis=[1, 0, 0], radians=np.pi / 2)
            quat = q2 * q1
            velocity = (bbox[7], 0.0, bbox[8])
            cat_id = bbox[10]
            score = bbox[9]
            if len(bbox) == 12:
                attrs.append(bbox[11])
            box = NuScenesBox(
                center,
                dims,
                quat,
                label=cat_id,
                score=score,
                velocity=velocity,
            )
            box_list.append(box)
        return box_list, attrs

    def update(
        self,
        metas,
        pred_bboxes,
    ):
        for i in range(len(pred_bboxes)):
            bboxes = pred_bboxes[i]
            m = {}
            m["ego2global_translation"] = metas["ego2global_translation"][i]
            m["ego2global_rotation"] = metas["ego2global_rotation"][i]
            m["sensor2ego_translation"] = metas["sensor2ego_translation"][i]
            m["sensor2ego_rotation"] = metas["sensor2ego_rotation"][i]
            m["token"] = metas["token"][i]
            bboxes = bboxes["ret"]
            annos = []
            bboxes_list, attrs = self._output_to_nusc_box(bboxes)
            for i, box in enumerate(bboxes_list):
                box = self._cam_to_ego(box, m)
                cls_range_map = self.eval_detection_configs.class_range
                radius = np.linalg.norm(box.center[:2], 2)
                det_range = cls_range_map[CLASSES[box.label]]
                if radius > det_range:
                    continue
                box = self._ego_to_global(box, m)
                nusc_anno = {
                    "sample_token": m["token"],
                    "bbox": box,
                    "attribute": attrs[i],
                }
                annos.append(nusc_anno)
            if m["token"] in self.nusc_annos:
                self.nusc_annos[m["token"]].extend(annos)
            else:
                self.nusc_annos[m["token"]] = annos

    def get(self):
        if len(self.nusc_annos) != 0:
            nusc_annos = self._gather()
            self.nusc_annos = {}
            for annos in nusc_annos:
                for k, v in annos.items():
                    if k in self.nusc_annos:
                        self.nusc_annos[k].extend(v)
                    else:
                        self.nusc_annos[k] = v
            self._get()
        return self.ret[0], self.ret[1]

    def fast_get(self):
        return self.ret[0], self.ret[1]

    def _nms(self):
        for token, annos in self.nusc_annos.items():
            bboxes = []
            scores = []
            labels = []
            if len(annos) == 0:
                continue
            for anno in annos:
                bbox = anno["bbox"]
                center = bbox.center.tolist()
                dims = bbox.wlh.tolist()
                rot = Quaternion(
                    bbox.orientation.elements.tolist()
                ).yaw_pitch_roll[0]
                nms_box = torch.tensor(
                    [center[0], center[1], dims[0], dims[1], rot],
                    dtype=torch.float32,
                )
                if not self.use_cpu:
                    nms_box = nms_box.cuda()
                bboxes.append(nms_box)
                scores.append(bbox.score)
                labels.append(bbox.label)
            bboxes = torch.stack(bboxes)
            scores = torch.tensor(scores)
            labels = torch.tensor(labels)
            if not self.use_cpu:
                scores = scores.cuda()
                labels = labels.cuda()
            keep = nms_bev_multiclass(
                bboxes, scores, labels, len(CLASSES), self.nms_threshold
            )
            new_annos = []
            for idx in keep:
                anno = annos[idx]
                box = anno["bbox"]
                name = CLASSES[box.label]
                attr = self.get_attr_name(int(anno["attribute"]), name)
                nusc_anno = {
                    "sample_token": anno["sample_token"],
                    "translation": box.center.tolist(),
                    "size": box.wlh.tolist(),
                    "rotation": box.orientation.elements.tolist(),
                    "velocity": box.velocity[:2].tolist(),
                    "detection_name": name,
                    "detection_score": box.score,
                    "attribute_name": attr,
                }

                new_annos.append(nusc_anno)
            self.nusc_annos[token] = new_annos

    def _dump(self):
        self._nms()
        super(NuscenesMonoMetric, self)._dump()
