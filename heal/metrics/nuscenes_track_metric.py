# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import json
import logging
import os
from typing import Sequence

import numpy as np
from hat_engine.metrics.metric import EvalMetric
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.distributed import (
    all_gather_object,
    get_dist_info,
    rank_zero_only,
)
from hat_engine.utils.package_helper import require_packages

try:
    import pyquaternion
    from nuscenes.eval.common.config import config_factory as track_configs
    from nuscenes.eval.common.utils import Quaternion
    from nuscenes.eval.tracking.evaluate import TrackingEval
    from nuscenes.utils.data_classes import Box as NuScenesBox
except ImportError:
    pyquaternion = None
    track_configs = None
    Quaternion = None
    TrackingEval = None
    NuScenesBox = None

logger = logging.getLogger(__name__)

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

__all__ = ["NuscenesTrackMetric"]


@OBJECT_REGISTRY.register
class NuscenesTrackMetric(EvalMetric):
    """Evaluation Nuscenes Tracking.

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
        "motmetrics",
        raise_msg="Please `pip3 install nuscenes-devkit pyquaternion motmetrics`",  # noqa
    )
    def __init__(
        self,
        name: str = "NuscenesMetric",
        data_root: str = "",
        version: str = "v1.0-mini",
        save_prefix: str = "./WORKSPACE/results",
        verbose: bool = True,
        eval_version: str = "tracking_nips_2019",
        use_lidar: bool = False,
        classes: Sequence[str] = None,
        use_ddp: bool = True,
        trans_lidar_dim: bool = False,
        trans_lidar_rot: bool = True,
        tracking_threshold: float = None,
        meta_key="meta",
        lidar_key="lidar2ego",
    ):
        """Initialize nuScenes tracking metric.

        Args:
            name: Name of this metric instance for display.
            data_root: Data path of nuscenes data.
            version: Version of nuscenes data.
            save_prefix: Path to save result.
            verbose: Whether output verbose log.
            eval_version: Eval version.
            use_lidar: Whether use lidar bbox.
            classes: List of class name.
            use_ddp: Whether use ddp to eval metric.
            trans_lidar_dim: Whether to transpose lidar box dims.
            trans_lidar_rot: Whether to transform lidar box rotation.
            tracking_threshold: Score threshold to filter tracking boxes.
            meta_key: Key to fetch meta info from batch data.
            lidar_key: Prefix key for lidar-to-ego calibration in meta.
        """
        super(NuscenesTrackMetric, self).__init__(name)
        self.tracking_threshold = tracking_threshold
        self.verbose = verbose
        self.save_prefix = save_prefix
        self.use_ddp = use_ddp
        self.meta_key = meta_key
        self.lidar_key = lidar_key

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
        self.eval_track_configs = track_configs(eval_version)
        self.use_lidar = use_lidar
        self.reset()
        if classes is None:
            self.CLASSES = CLASSES
        else:
            self.CLASSES = classes

    def _output_to_nusc_box(self, bboxes):
        """Convert predicted bbox tensors to nuScenes boxes.

        Args:
            bboxes: Iterable of bbox tensors in ego coordinate.

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
        """Transform box from ego to global coordinate.

        Args:
            bbox: NuScenesBox in ego coordinate.
            meta: Meta info with ego2global calibration.

        Returns:
            The NuScenesBox in global coordinate.
        """
        ego2global_translation = meta["ego2global_translation"]
        ego2global_rotation = meta["ego2global_rotation"]
        bbox.rotate(Quaternion(ego2global_rotation))
        bbox.translate(ego2global_translation)
        return bbox

    def compute(self):
        """Compute metric results (unused, kept for interface)."""
        pass

    def _output_to_nusc_box_lidar(
        self,
        detection,
        with_velocity=True,
        threshold=None,
    ):
        """Convert the output to the box class in the nuScenes.

        Args:
            detection: Detection results.

                - boxes_3d: Detection bbox.
                - scores_3d: Detection scores.
                - labels_3d: Predicted box labels.

        Returns:
            list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
        """
        box3d = detection["boxes_3d"].numpy()
        scores = detection["scores_3d"].numpy()
        labels = detection["labels_3d"].numpy()

        if self.trans_lidar_rot is True:
            box3d[:, 6] = -box3d[:, 6] - np.pi / 2

        if "instance_ids" in detection:
            ids = detection["instance_ids"]

        if threshold is not None:
            if "cls_scores" in detection:
                mask = detection["cls_scores"].numpy() >= threshold
            else:
                mask = scores >= threshold
            box3d = box3d[mask]
            scores = scores[mask]
            labels = labels[mask]
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
            if "instance_ids" in detection:
                box.token = ids[i]
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
        """Accumulate tracking predictions of one batch.

        Args:
            meta: Batch meta info containing calibration and sample tokens.
            pred_bboxes: Predicted tracking bboxes for each sample.
        """
        meta = meta[self.meta_key]
        for m, bboxes in zip(meta, pred_bboxes):
            annos = []
            if self.use_lidar:
                bboxes_list = self._output_to_nusc_box_lidar(
                    bboxes,
                    with_velocity=True,
                    threshold=self.tracking_threshold,
                )
            else:
                raise NotImplementedError("Not support now: `ues_lidar=False`")
                # bboxes_list = self._output_to_nusc_box(bboxes)
            sample_token = m["sample_token"]
            for box in bboxes_list:
                name = self.CLASSES[box.label]
                if name in [
                    "barrier",
                    "traffic_cone",
                    "construction_vehicle",
                ]:
                    continue
                if self.use_lidar:
                    box = self._lidar_nusc_box_to_global(
                        m, box, self.CLASSES, self.eval_track_configs
                    )
                else:
                    box = self._ego_to_global(box, m)
                if box is None:
                    continue
                nusc_anno = {
                    "sample_token": sample_token,
                    "translation": box.center.tolist(),
                    "size": box.wlh.tolist(),
                    "rotation": box.orientation.elements.tolist(),
                    "velocity": box.velocity[:2].tolist(),
                    "tracking_name": name,
                    "tracking_score": box.score,
                    "tracking_id": str(box.token),
                }
                annos.append(nusc_anno)
            self.nusc_annos[sample_token] = annos

    def _gather(self):
        """Gather accumulated annotations from all ranks.

        Returns:
            List of per-rank annotation dicts.
        """
        global_rank, global_world_size = get_dist_info()
        global_output = [None for _ in range(global_world_size)]
        all_gather_object(global_output, self.nusc_annos)
        return global_output

    def reset(self):
        """Reset accumulated annotations and metric results."""
        self.nusc_annos = {}
        self.ret = {
            "amota": 0.0,
            "amotp": 0.0,
            "ids": 0,
        }

    def get(self):
        """Evaluate accumulated results and return metric names and values.

        Returns:
            Tuple of metric name list and metric value list.
        """
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
        res_names = []
        res_values = []
        for name, value in self.ret.items():
            res_names.append(name)
            res_values.append(value)
        return res_names, res_values

    def _dump(self):
        """Dump accumulated tracking results to a nuScenes-format json file."""
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
        """Run nuScenes tracking evaluation and record summary metrics."""
        self._dump()
        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
        }

        nusc_eval = TrackingEval(
            config=self.eval_track_configs,
            result_path=self.res_path,
            eval_set=eval_set_map[self.version],
            output_dir=self.save_prefix,
            verbose=self.verbose,
            nusc_version=self.version,
            nusc_dataroot=self.data_root,
        )
        metrics = nusc_eval.main()

        # record metrics
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
        for key in keys:
            if key in self.ret.keys():
                self.ret[key] = metrics[key]
