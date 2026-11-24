# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
"""Packed NuScenes dataset for bev_vtv2 (multi-view + lidar 3D detection)."""

import copy
import logging
import math
import os
from typing import Callable, Dict, Optional, Tuple

import cv2
import msgpack
import numpy as np
import torch
from hat_engine.data.datasets.data_packer import Packer
from hat_engine.data.utils import decode_img
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.pack_type import PackTypeMapper
from hat_engine.utils.pack_type.utils import get_packtype_from_path
from hat_engine.utils.package_helper import require_packages
from PIL import Image
from torch.utils.data import Dataset

from heal.core.cam_box3d import CameraInstance3DBoxes

try:
    from nuscenes.eval.common.utils import Quaternion
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits
except ImportError:
    Quaternion = None
    NuScenes = None
    splits = None

logger = logging.getLogger(__name__)

__all__ = [
    "NuscenesBevDataset",
    "NuscenesBevSequenceDataset",
    "NuscenesMonoDataset",
    "NuscenesPacker",
]

CAM_NAMES = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
)

_BUS_SUBTYPES = ("vehicle.bus.bendy", "vehicle.bus.rigid")
_PEDESTRIAN_SUBTYPES = (
    "human.pedestrian.adult",
    "human.pedestrian.child",
    "human.pedestrian.construction_worker",
    "human.pedestrian.police_officer",
)

NameMapping = {
    "movable_object.barrier": "barrier",
    "vehicle.bicycle": "bicycle",
    **{src: "bus" for src in _BUS_SUBTYPES},
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    **{src: "pedestrian" for src in _PEDESTRIAN_SUBTYPES},
    "movable_object.trafficcone": "traffic_cone",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}

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

Attributes = (
    "cycle.with_rider",
    "cycle.without_rider",
    "pedestrian.moving",
    "pedestrian.standing",
    "pedestrian.sitting_lying_down",
    "vehicle.moving",
    "vehicle.parked",
    "vehicle.stopped",
    "None",
)

PACK_CAMERA_TYPES = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


class Bbox(object):
    """Single 3D box in lidar frame from a packed NuScenes sample."""

    def __init__(self, sample: dict, bbox: dict):
        self.sample = sample
        box = copy.deepcopy(bbox["bbox"])
        self.center = box[:3]
        self.wlh = box[3:6]
        self.cat = bbox["cat"]
        self.rot = Quaternion(bbox["rot"])
        self.attr_id = bbox.get("attr_id", 0)
        self.velocity = np.array([*bbox["velocity"], 0.0])
        if np.isnan(self.velocity).any():
            self.velocity = np.array([0.0, 0.0, 0.0])

    def _limit_period(self, val, offset=0.5, period=np.pi * 2):
        return val - np.floor(val / period + offset) * period

    def _translate(self, x: np.ndarray):
        self.center += x

    def _rotate(self, rot: Quaternion):
        self.center = np.dot(rot.rotation_matrix, self.center)
        self.rot = rot * self.rot
        self.velocity = np.dot(rot.rotation_matrix, self.velocity)

    def get_lidar_bboxes(self) -> np.ndarray:
        """Convert the box to lidar-frame label format used by CenterPoint.

        Returns:
            np.ndarray: Box tensor
                ``(center_xyz, lwh, yaw, vel_xy, class_id)``.
        """
        sensor2ego_translation = self.sample["sensor2ego_translation"]
        sensor2ego_rotation = self.sample["sensor2ego_rotation"]
        ego2global_translation = self.sample["ego2global_translation"]
        ego2global_rotation = self.sample["ego2global_rotation"]

        self._translate(-np.array(ego2global_translation))
        self._rotate(Quaternion(ego2global_rotation).inverse)
        self._translate(-np.array(sensor2ego_translation))
        self._rotate(Quaternion(sensor2ego_rotation).inverse)

        yaw = Quaternion(self.rot).yaw_pitch_roll[0]
        yaw = np.array([self._limit_period(yaw)])
        lwh = np.array([self.wlh[1], self.wlh[0], self.wlh[2]])
        if self.cat in NameMapping:
            cat_id = CLASSES.index(NameMapping[self.cat])
        else:
            cat_id = -1
        cat_id = np.array([cat_id])
        return np.concatenate(
            [self.center, lwh, yaw, self.velocity[:2], cat_id], axis=0
        )

    def get_bev_bboxes(self) -> np.ndarray:
        """Convert the box to ego-frame label format used by DETR3D/PETR.

        Returns:
            np.ndarray: Box tensor
                ``(center_xyz, wlh, yaw, vel_xy, class_id)``.
        """
        ego2global_translation = self.sample["ego2global_translation"]
        ego2global_rotation = self.sample["ego2global_rotation"]
        self._translate(-np.array(ego2global_translation))
        self._rotate(Quaternion(ego2global_rotation).inverse)

        yaw = np.array(
            [self._limit_period(Quaternion(self.rot).yaw_pitch_roll[0])]
        )
        wlh = np.array(self.wlh)
        if self.cat in NameMapping:
            cat_id = CLASSES.index(NameMapping[self.cat])
        else:
            cat_id = -1
        cat_id = np.array([cat_id])
        return np.concatenate(
            [self.center, wlh, yaw, self.velocity[:2], cat_id], axis=0
        )

    def get_attr_label(self):
        """Return attribute class id (e.g. moving/stopped). \
        FCOS3D uses this."""
        return self.attr_id

    def corners(self):
        """Compute 3D corner coordinates (3x8)."""
        w, l, h = self.wlh
        x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))
        corners = np.dot(self.rot.rotation_matrix, corners)
        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z
        return corners

    def get_cam_bboxes(self, cam):
        """Transform bbox to camera coordinate system. \
        FCOS3D mono head uses this."""
        ego2global_translation = cam["ego2global_translation"]
        ego2global_rotation = cam["ego2global_rotation"]
        sensor2ego_translation = cam["sensor2ego_translation"]
        sensor2ego_rotation = cam["sensor2ego_rotation"]

        self._translate(-np.array(ego2global_translation))
        self._rotate(Quaternion(ego2global_rotation).inverse)

        self._translate(-np.array(sensor2ego_translation))
        self._rotate(Quaternion(sensor2ego_rotation).inverse)

        yaw = Quaternion(self.rot).yaw_pitch_roll[0]
        yaw = np.array([self._limit_period(yaw)])
        wlh = np.array(self.wlh)
        lhw = wlh[..., [1, 2, 0]]
        if self.cat in NameMapping:
            cat_id = CLASSES.index(NameMapping[self.cat])
        else:
            cat_id = -1
        velocity = self.velocity[0::2]
        cat_id = np.array([cat_id])
        bboxes = np.concatenate(
            [self.center, lhw, -yaw, velocity, cat_id], axis=0
        )
        return bboxes


class NuscenesSample(object):
    """One decoded sample from packed NuScenes LMDB."""

    def __init__(self, sample: dict):
        self.sample = sample

    def get_lidar_bboxes(
        self, max_dist: float = 55.0, return_instance_ids: bool = False
    ):
        """Filter and convert GT boxes to lidar-frame tensors.

        Args:
            max_dist: Max radial distance (meters) from ego in lidar frame.
            return_instance_ids: If True, also return per-box instance ids.

        Returns:
            np.ndarray or tuple: Box array ``(N, 10)``, optionally with
            instance id list.
        """
        bboxes = []
        instance_ids = []
        for bbox in self.sample["gt_bboxes"]:
            if bbox["num_lidar_pts"] == 0 and bbox["num_radar_pts"] == 0:
                continue
            box = Bbox(self.sample, bbox)
            lidar_bbox = box.get_lidar_bboxes()
            if np.sqrt(np.sum(lidar_bbox[:2] ** 2)) > max_dist:
                continue
            bboxes.append(lidar_bbox)
            if return_instance_ids and "instance_id" in bbox:
                instance_ids.append(bbox["instance_id"])
        if not bboxes:
            empty = np.zeros((0, 10), dtype=np.float32)
            if return_instance_ids:
                return empty, []
            return empty
        stacked = np.stack(bboxes)
        if return_instance_ids:
            return stacked, instance_ids
        return stacked

    def get_timestamp(self):
        return self.sample["timestamp"] / 1e6

    def _get_cam2ego(self, cam):
        cam2ego = np.eye(4)
        rotation = Quaternion(cam["sensor2ego_rotation"]).rotation_matrix
        cam2ego[:3, :3] = rotation
        cam2ego[:3, 3] = cam["sensor2ego_translation"]
        return cam2ego

    def get_ego_bboxes(self, bev_range: Tuple[float, ...]) -> np.ndarray:
        """Filter and convert GT boxes to ego-frame tensors.

        Args:
            bev_range: Ego-frame cuboid filter
                ``(x_min, y_min, z_min, x_max, y_max, z_max)``.

        Returns:
            np.ndarray: Stacked boxes with shape ``(N, 10)``, or ``(0, 10)``.
        """
        if bev_range is None:
            raise ValueError("bev_range is needed to generate ego bboxes")
        bboxes = []
        for bbox in self.sample["gt_bboxes"]:
            if bbox["num_lidar_pts"] == 0 and bbox["num_radar_pts"] == 0:
                continue
            box = Bbox(self.sample, bbox)
            ego_bbox = box.get_bev_bboxes()
            valid_mask = (
                (ego_bbox[0] > bev_range[0])
                & (ego_bbox[0] < bev_range[3])
                & (ego_bbox[1] > bev_range[1])
                & (ego_bbox[1] < bev_range[4])
                & (ego_bbox[2] > bev_range[2])
                & (ego_bbox[2] < bev_range[5])
                & (ego_bbox[9] >= 0)
            )
            if not valid_mask:
                continue
            bboxes.append(ego_bbox)
        if not bboxes:
            return np.zeros((0, 10), dtype=np.float32)
        return np.stack(bboxes)

    def get_scene(self):
        return self.sample["scene"]

    def get_cam_by_name(self, name: str) -> Dict:
        """Load one camera view and its projection matrices.

        Args:
            name: NuScenes camera name (e.g. ``CAM_FRONT``).

        Returns:
            dict: Camera dict with ``img``, intrinsics, and ``global2img``.
        """
        cam = None
        for c in self.sample["cam"]:
            if name == c["name"]:
                cam = copy.deepcopy(c)
                break
        if cam is None:
            raise ValueError(f"Cannot find cam {name}")
        cam["img"] = decode_img(cam["img"], iscolor=cv2.IMREAD_COLOR)
        cam["img"] = Image.fromarray(cam["img"])
        ego2global = self._cam_ego2global(cam)
        ego2img = self._homography_by_cam(cam)
        cam["ego2img"] = ego2img
        cam["global2img"] = ego2img @ np.linalg.inv(ego2global)
        cam["cam2ego"] = self._get_cam2ego(cam)
        return cam

    def _homography_by_cam(self, cam):
        rotation = Quaternion(cam["sensor2ego_rotation"]).rotation_matrix
        ego2sensor_r = np.linalg.inv(rotation)
        ego2sensor_t = cam["sensor2ego_translation"] @ ego2sensor_r.T
        ego2sensor = np.eye(4)
        ego2sensor[:3, :3] = ego2sensor_r.T
        ego2sensor[3, :3] = -np.array(ego2sensor_t)
        intrinsic = np.array(cam["camera_intrinsic"])
        viewpad = np.eye(4)
        viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
        return viewpad @ ego2sensor.T

    def _cam_ego2global(self, cam):
        cam_ego2global = np.eye(4)
        rotation = Quaternion(cam["ego2global_rotation"]).rotation_matrix
        cam_ego2global[:3, :3] = rotation
        cam_ego2global[:3, 3] = cam["ego2global_translation"]
        return cam_ego2global

    def get_lidar2ego(self):
        l2e_t = np.array(self.sample["sensor2ego_translation"])
        l2e_r = np.array(self.sample["sensor2ego_rotation"])
        l2e_m = np.zeros((4, 4), dtype=np.float64)
        l2e_m[:3, :3] = Quaternion(l2e_r).rotation_matrix
        l2e_m[:3, 3] = l2e_t
        l2e_m[3, 3] = 1.0
        return l2e_m

    def get_ego2gloabl(self):
        e2g_t = np.array(self.sample["ego2global_translation"])
        e2g_r = np.array(self.sample["ego2global_rotation"])
        e2g_m = np.zeros((4, 4), dtype=np.float64)
        e2g_m[:3, :3] = Quaternion(e2g_r).rotation_matrix
        e2g_m[:3, 3] = e2g_t
        e2g_m[3, 3] = 1.0
        return e2g_m

    def get_lidar2global(self, lidar2ego, ego2global):
        return ego2global @ lidar2ego

    def get_lidar2img(self, cam_global2img, lidar2global):
        return [g2i @ lidar2global for g2i in cam_global2img]

    def get_center2d(self, bbox, cam):
        """Project 3D center to 2D and return (center2d, depth). \
        FCOS3D uses this."""
        center3d = bbox[:3]
        camera_intrinsic = np.array(cam["camera_intrinsic"])
        center2d = camera_intrinsic @ center3d
        center2d[:2] = center2d[:2] / center2d[2]
        return center2d[:2], center2d[2]

    def get_corner2d(self, corners3d, cam, im_size):
        """Project 3D corners to 2D bounding box [min_x, min_y, max_x, max_y]. FCOS3D uses this."""  # noqa: E501
        in_front = np.argwhere(corners3d[2, :] > 0).flatten()
        corners3d = corners3d[:, in_front]
        camera_intrinsic = np.array(cam["camera_intrinsic"])
        corner2d = camera_intrinsic @ corners3d
        corner2d = np.transpose(corner2d, (1, 0))
        corner2d[..., 0] = corner2d[..., 0] / corner2d[..., 2]
        corner2d[..., 1] = corner2d[..., 1] / corner2d[..., 2]

        if len(corner2d) == 0:
            return np.array([0, 0, 0, 0])

        min_x = max(min(corner2d[..., 0]), 0)
        min_y = max(min(corner2d[..., 1]), 0)
        max_x = min(max(corner2d[..., 0]), im_size[0])
        max_y = min(max(corner2d[..., 1]), im_size[1])

        return np.array([min_x, min_y, max_x, max_y])

    def get_mono_by_index(self, index):
        """Get mono-camera sample by camera index. FCOS3D uses this."""
        cam = self.sample["cam"][index]
        return self.get_mono_by_name(cam["name"])

    def get_mono_by_name(self, name):
        """Get mono-camera sample dict for one view. FCOS3D uses this."""
        mono = {}
        cam = self.get_cam_by_name(name)
        im_size = cam["img"].size
        gt_bboxes = self.sample["gt_bboxes"]
        bboxes = []
        center2ds = []
        depths = []
        corner2ds = []
        attr_labels = []
        for bbox in gt_bboxes:
            box = Bbox(self.sample, bbox)
            cam_bbox = box.get_cam_bboxes(cam)
            corners3d = box.corners()
            center2d, depth = self.get_center2d(cam_bbox, cam)
            corner2d = self.get_corner2d(corners3d, cam, im_size)
            w = corner2d[2] - corner2d[0]
            h = corner2d[3] - corner2d[1]
            valid_mask = (
                (corner2d[0] < im_size[0])
                & (corner2d[2] > 0)
                & (corner2d[1] < im_size[1])
                & (corner2d[3] > 0)
                & (cam_bbox[9] >= 0)
                & (depth > 0)
                & (h > 1)
                & (w > 1)
            )
            if not valid_mask:
                continue
            center2ds.append(center2d)
            depths.append(depth)
            corner2ds.append(corner2d)
            bboxes.append(cam_bbox)
            attr_labels.append(box.get_attr_label())
        if len(center2ds) > 0:
            center2ds = np.stack(center2ds)
            depths = np.stack(depths)
            corner2ds = np.stack(corner2ds)
            bboxes = np.stack(bboxes)
        mono["img"] = cam["img"]
        mono["file_name"] = cam["img_path"]
        mono["gt_bboxes_3d"] = bboxes
        mono["center2d"] = center2ds
        mono["depth"] = depths
        mono["gt_bboxes"] = corner2ds
        mono["ego2global_translation"] = cam["ego2global_translation"]
        mono["ego2global_rotation"] = cam["ego2global_rotation"]
        mono["sensor2ego_translation"] = cam["sensor2ego_translation"]
        mono["sensor2ego_rotation"] = cam["sensor2ego_rotation"]
        mono["camera_intrinsic"] = cam["camera_intrinsic"]
        mono["attr_labels"] = attr_labels
        mono["token"] = self.get_token()
        return mono

    def num_cams(self):
        """Return number of camera views. FCOS3D uses this."""
        return len(self.sample["cam"])

    def get_token(self):
        """Return sample token. FCOS3D uses this."""
        return self.sample.get("sample_token", "")

    def get_meta(self):
        keys = (
            "sample_token",
            "sensor2ego_rotation",
            "sensor2ego_translation",
            "ego2global_rotation",
            "ego2global_translation",
            "location",
            "scene",
        )
        meta = {k: self.sample[k] for k in keys if k in self.sample}
        meta["lidar2ego_translation"] = self.sample["sensor2ego_translation"]
        meta["lidar2ego_rotation"] = self.sample["sensor2ego_rotation"]
        return meta


class NuscenesDataset(Dataset):
    """Base loader for packed NuScenes LMDB.

    Args:
        data_path: Path to packed LMDB directory.
        transforms: Optional transform pipeline applied in ``__getitem__``.
        pack_type: Pack backend name when path type cannot be inferred.
        pack_kwargs: Extra kwargs passed to the pack file backend.
    """

    @require_packages(
        "nuscenes", raise_msg="Please `pip3 install nuscenes-devkit`"
    )
    def __init__(
        self,
        data_path: str,
        transforms: Optional[Callable] = None,
        pack_type: Optional[str] = None,
        pack_kwargs: Optional[dict] = None,
        num_split: int = 1,
    ):
        self.root = data_path
        self.transforms = transforms
        self.scenes_aug = {}
        self.kwargs = {} if pack_kwargs is None else pack_kwargs
        try:
            self.pack_type = get_packtype_from_path(data_path)
        except NotImplementedError:
            assert pack_type is not None
            self.pack_type = PackTypeMapper(pack_type.lower())

        self.pack_file = self.pack_type(
            self.root, writable=False, **self.kwargs
        )
        self.pack_file.open()
        self.samples = self.pack_file.get_keys()
        self.meta = self._decode_meta(self.pack_file)
        if self.meta and num_split > 1:
            org_len = len(self.meta["scene_info"])
            self._split_scenes(num_split)
            logger.info(
                "Split scenes from %d to %d",
                org_len,
                len(self.meta["scene_info"]),
            )

    def get_meta(self):
        return self.meta

    def get_aug(self):
        aug_cfg = {}
        if self.transforms is not None and hasattr(
            self.transforms, "transforms"
        ):
            for trans in self.transforms.transforms:
                if hasattr(trans, "get_aug"):
                    name, cfg = trans.get_aug()
                    aug_cfg[name] = cfg
        return aug_cfg

    def _split_scenes(self, num_split):
        new_scene_info = {}
        for k, v in self.meta["scene_info"].items():
            sequence_length = np.array(
                list(range(0, v, math.ceil(v / num_split))) + [v]
            )
            for idx, sub_seq_len in enumerate(
                sequence_length[1:] - sequence_length[:-1]
            ):
                new_scene_info[f"{k}_{idx}"] = sub_seq_len
        self.meta["scene_info"] = new_scene_info

    def _decode_meta(self, pack_file):
        return self._decode(pack_file, "__meta__")

    def _decode(self, pack_file, sample):
        def _decode_hook(obj):
            new_obj = {}
            for k, v in obj.items():
                k = k.decode("utf-8") if isinstance(k, bytes) else k
                if k in ("lidar_points", "sweeps"):
                    continue
                if k == "img":
                    new_obj[k] = v
                elif isinstance(v, bytes):
                    new_obj[k] = v.decode("utf-8")
                else:
                    new_obj[k] = v
            return new_obj

        raw = pack_file.read(sample)
        if raw is None:
            return None
        return msgpack.unpackb(raw, object_hook=_decode_hook, raw=True)

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.pack_file is not None:
            self.pack_file.close()
        state["pack_file"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.pack_file = self.pack_type(
            self.root, writable=False, **self.kwargs
        )
        self.pack_file.open()
        self.samples = self.pack_file.get_keys()


@OBJECT_REGISTRY.register
class NuscenesBevDataset(NuscenesDataset):
    """Packed NuScenes dataset for bev_vtv2 (6 cameras + lidar 3D boxes).

    Args:
        max_dist: Max GT box distance in lidar frame (meters).
            Only applied when ``with_lidar_bboxes=True``.
        with_lidar_bboxes: When True, populate ``lidar_bboxes_labels``.
            Default True preserves migrated bev_vtv2 behavior.
        with_ego_bboxes: When True, populate ``ego_bboxes_labels`` in ego frame
            (for DETR3D/PETR). Default False; field is omitted when False.
        bev_range: Ego-frame cuboid filter, required when
            ``with_ego_bboxes=True``.
        **kwargs: Forwarded to :class:`NuscenesDataset` (e.g. ``data_path``).
    """

    def __init__(
        self,
        max_dist: Optional[float] = 55.0,
        with_lidar_bboxes: bool = True,
        with_ego_bboxes: bool = False,
        bev_range: Optional[Tuple[float, ...]] = None,
        map_path: Optional[str] = None,
        map_size: Optional[tuple] = None,
        with_bev_bboxes: bool = False,
        with_bev_mask: bool = False,
        test_mode: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.test_mode = test_mode
        self.sampler = NuscenesBevSampler(
            max_dist=max_dist,
            with_lidar_bboxes=with_lidar_bboxes,
            with_ego_bboxes=with_ego_bboxes,
            bev_range=bev_range,
            with_bev_bboxes=with_bev_bboxes,
        )
        self.map_path = map_path
        self.map_size = map_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        aug_cfg = None
        if isinstance(item, dict):
            idx = item["idx"]
            aug_cfg = item.get("aug")
        else:
            idx = item
        sample = NuscenesSample(
            self._decode(self.pack_file, self.samples[idx])
        )
        data = self.sampler(sample, test_mode=self.test_mode)
        if aug_cfg:
            data["scenes_aug"] = aug_cfg
        if self.transforms is not None:
            data = self.transforms(data)
        return data


@OBJECT_REGISTRY.register
class NuscenesBevSequenceDataset(NuscenesBevDataset):
    """Packed NuScenes sequence dataset for temporal BEV models (e.g. BevFormer)."""  # noqa: E501

    def __init__(self, num_seq: int, **kwargs):
        super().__init__(**kwargs)
        self.num_seq = num_seq

    def __getitem__(self, item):
        start = item
        end = item - self.num_seq
        seq_data = []
        for i in range(start, end, -1):
            idx = max(i, 0)
            seq_data.append(super().__getitem__(idx))
        return seq_data


class NuscenesBevSampler(object):
    """Build one training/eval sample dict for BEV detection."""

    def __init__(
        self,
        max_dist: Optional[float] = 55.0,
        with_lidar_bboxes: bool = True,
        with_ego_bboxes: bool = False,
        bev_range: Optional[Tuple[float, ...]] = None,
        with_bev_bboxes: bool = False,
    ):
        self.max_dist = max_dist
        self.with_lidar_bboxes = with_lidar_bboxes
        self.with_ego_bboxes = with_ego_bboxes
        self.bev_range = bev_range
        self.with_bev_bboxes = with_bev_bboxes

    def __call__(
        self, sample: NuscenesSample, test_mode: bool = False
    ) -> dict:
        """Assemble model inputs from one packed sample.

        Args:
            sample: Decoded NuScenes sample.

        Returns:
            dict: Keys include ``img``, ``lidar2img``, ``lidar2global``,
                ``camera_intrinsic``, and ``meta``. GT box keys are populated
                only when the corresponding ``with_*`` flag is True.
        """
        ego_bboxes_labels = None
        if self.with_ego_bboxes:
            if self.bev_range is None:
                raise ValueError(
                    "bev_range is required when with_ego_bboxes=True"
                )
            ego_bboxes_labels = sample.get_ego_bboxes(self.bev_range)

        imgs = []
        ego2img = []
        cam2ego = []
        cam_intrin = []
        cam_global2img = []
        cam_names = []
        img_names = []
        cam_ego2img = []
        for name in CAM_NAMES:
            cam = sample.get_cam_by_name(name)
            imgs.append(cam["img"])
            img_names.append(cam.get("img_path", name))
            cam_names.append(name)
            cam_intrin.append(cam["camera_intrinsic"])
            cam2ego.append(cam["cam2ego"])
            ego2img.append(cam["ego2img"])
            cam_global2img.append(cam["global2img"])
            cam_ego2img.append(cam["ego2img"])

        lidar2ego = sample.get_lidar2ego()
        ego2global = sample.get_ego2gloabl()
        lidar2global = sample.get_lidar2global(lidar2ego, ego2global)
        cam_ego2global = sample._cam_ego2global(sample.sample["cam"][0])

        data = {
            "img": imgs,
            "img_name": img_names,
            "cam_name": cam_names,
            "camera_intrinsic": np.array(cam_intrin),
            "cam2ego": cam2ego,
            "layout": "chw",
            "color_space": "rgb",
            "timestamp": sample.get_timestamp(),
            "lidar2ego": lidar2ego,
            "lidar2global": lidar2global,
            "cam_ego2global": cam_ego2global,
            "lidar2img": sample.get_lidar2img(cam_global2img, lidar2global),
            "ego2global": ego2global,
            "scene": sample.get_scene(),
            "sample_token": sample.get_token(),
            "meta": sample.get_meta(),
        }
        if self.with_lidar_bboxes:
            lidar_bboxes_labels, instance_ids = sample.get_lidar_bboxes(
                max_dist=self.max_dist,
                return_instance_ids=True,
            )
            data["lidar_bboxes_labels"] = lidar_bboxes_labels
            data["instance_ids"] = instance_ids
        if self.with_ego_bboxes:
            data["ego_bboxes_labels"] = ego_bboxes_labels
            data["ego2img"] = np.array(cam_ego2img)
        else:
            data["ego2img"] = ego2img
        return data


class NuscenesPackParser(object):
    """Parse raw NuScenes into msgpack-friendly dicts for bev_vtv2 packing.

    Args:
        version: NuScenes version (e.g. ``v1.0-trainval``).
        src_data_dir: Raw NuScenes root directory.
        split_name: ``train`` or ``val`` split name.
    """

    @require_packages(
        "nuscenes", raise_msg="Please `pip3 install nuscenes-devkit`"
    )
    def __init__(
        self,
        version: str,
        src_data_dir: str,
        split_name: str = "val",
        max_sweeps: int = 10,
    ):
        self.nusc = NuScenes(
            version=version, dataroot=src_data_dir, verbose=True
        )
        self.root_path = src_data_dir
        self.max_sweeps = max_sweeps
        self.test = "test" in version
        available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
        assert version in available_vers
        if version == "v1.0-trainval":
            train_scenes, val_scenes = splits.train, splits.val
        elif version == "v1.0-test":
            train_scenes, val_scenes = splits.test, []
        else:
            train_scenes, val_scenes = splits.mini_train, splits.mini_val

        scene_names = train_scenes if split_name == "train" else val_scenes
        self.scenes = [
            scene["token"]
            for scene in self.nusc.scene
            if scene["name"] in scene_names
        ]

        self.samples = []
        self.scene2len = {}
        for scene_token in self.scenes:
            scene_len = 0
            scene = self.nusc.get("scene", scene_token)
            loc = self.nusc.get("log", scene["log_token"])["location"]
            next_sample_token = scene["first_sample_token"]
            logger.info("process scene %s %s ...", loc, scene_token)
            while next_sample_token:
                self.samples.append(next_sample_token)
                sample = self.nusc.get("sample", next_sample_token)
                next_sample_token = sample["next"]
                scene_len += 1
            self.scene2len[scene_token] = scene_len

    def _get_meta(self):
        """Pack metadata written to ``__meta__`` in the LMDB.

        Returns:
            dict: ``scene_info`` maps scene token to sample count per scene;
                ``scene_len`` is the number of scenes; ``data_len`` is total
                samples across all scenes.
        """
        return {
            "scene_info": self.scene2len,
            "scene_len": len(self.scenes),
            "data_len": len(self.samples),
            "max_sweeps": self.max_sweeps,
        }

    def _load_img(self, img_path):
        img_path = os.path.join(self.root_path, img_path)
        img = Image.open(img_path).convert("RGB")
        return cv2.imencode(".jpg", np.array(img))[1].tobytes()

    def _gen_anns(self, sample, info):
        gt_bboxes = []
        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)
            center = np.array(ann["translation"])
            wlh = np.array(ann["size"])
            rot = np.array(Quaternion(ann["rotation"]).elements.tolist())
            attr_tokens = ann["attribute_tokens"]
            if attr_tokens:
                attr_name = self.nusc.get("attribute", attr_tokens[0])["name"]
            else:
                attr_name = "None"
            attr_id = Attributes.index(attr_name)
            instance_id = self.nusc.getind("instance", ann["instance_token"])
            velocity = self.nusc.box_velocity(ann_token)[:2].tolist()
            gt_bboxes.append(
                {
                    "bbox": np.concatenate([center, wlh], axis=0).tolist(),
                    "rot": rot.tolist(),
                    "attr_name": attr_name,
                    "attr_id": attr_id,
                    "cat": ann["category_name"],
                    "token": ann_token,
                    "velocity": velocity,
                    "num_lidar_pts": ann["num_lidar_pts"],
                    "num_radar_pts": ann["num_radar_pts"],
                    "instance_id": instance_id,
                }
            )
        info["gt_bboxes"] = gt_bboxes

    def _gen_cam_info(self, sample, info):
        cams = []
        for name, token in sample["data"].items():
            if name not in PACK_CAMERA_TYPES:
                continue
            sd_rec = self.nusc.get("sample_data", token)
            pose_rec = self.nusc.get("ego_pose", sd_rec["ego_pose_token"])
            cs_rec = self.nusc.get(
                "calibrated_sensor", sd_rec["calibrated_sensor_token"]
            )
            cams.append(
                {
                    "img_path": sd_rec["filename"],
                    "img": self._load_img(sd_rec["filename"]),
                    "name": name,
                    "token": token,
                    "camera_intrinsic": cs_rec["camera_intrinsic"],
                    "sensor2ego_translation": cs_rec["translation"],
                    "sensor2ego_rotation": cs_rec["rotation"],
                    "ego2global_translation": pose_rec["translation"],
                    "ego2global_rotation": pose_rec["rotation"],
                    "timestamp": sd_rec["timestamp"],
                }
            )
        info["cam"] = cams

    def _gen_lidar_info(self, sample, info):
        """Store lidar-sensor pose only; point clouds use lidar LMDB."""
        lidar_token = sample["data"]["LIDAR_TOP"]
        sd_rec = self.nusc.get("sample_data", lidar_token)
        cs_rec = self.nusc.get(
            "calibrated_sensor", sd_rec["calibrated_sensor_token"]
        )
        pose_rec = self.nusc.get("ego_pose", sd_rec["ego_pose_token"])
        info["sensor2ego_translation"] = cs_rec["translation"]
        info["sensor2ego_rotation"] = cs_rec["rotation"]
        info["ego2global_translation"] = pose_rec["translation"]
        info["ego2global_rotation"] = pose_rec["rotation"]

    def _gen_info(self, sample_token):
        logger.info("process sample %s ...", sample_token)
        sample = self.nusc.get("sample", sample_token)
        scene = self.nusc.get("scene", sample["scene_token"])
        loc = self.nusc.get("log", scene["log_token"])["location"]
        info = {
            "scene": sample["scene_token"],
            "location": loc,
            "sample_token": sample_token,
            "prev": sample["prev"],
            "next": sample["next"],
            "scene_token": sample["scene_token"],
            "timestamp": sample["timestamp"],
        }
        self._gen_anns(sample, info)
        self._gen_cam_info(sample, info)
        self._gen_lidar_info(sample, info)
        return info

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        return self._gen_info(self.samples[index])


class NuscenesPacker(Packer):
    """Pack raw NuScenes into LMDB/msgpack for :class:`NuscenesBevDataset`.

    Args:
        version: NuScenes version string.
        src_data_dir: Raw NuScenes root directory.
        target_data_dir: Output pack directory.
        split_name: ``train`` or ``val``.
        num_workers: Parallel workers for packing.
        pack_type: Pack backend (e.g. ``lmdb``).
        **kwargs: Extra kwargs for :class:`Packer`.
    """

    @require_packages(
        "nuscenes", raise_msg="Please `pip3 install nuscenes-devkit`"
    )
    def __init__(
        self,
        version: str,
        src_data_dir: str,
        target_data_dir: str,
        split_name: str,
        num_workers: int,
        pack_type: str,
        **kwargs,
    ):
        self.dataset = NuscenesPackParser(
            version=version,
            src_data_dir=src_data_dir,
            split_name=split_name,
            max_sweeps=kwargs.pop("max_sweeps", 10),
        )
        self.packer_meta = self.dataset._get_meta()
        num_samples = len(self.dataset)
        logger.info(
            "Packing NuScenes %s %s, total samples %d",
            version,
            split_name,
            num_samples,
        )
        super().__init__(
            target_data_dir, num_samples, pack_type, num_workers, **kwargs
        )

    def _write_length(self, num):
        try:
            self.pack_file.write("__len__", "{}".format(num).encode("ascii"))
            if self.packer_meta is not None:
                self.pack_file.write(
                    "__meta__", msgpack.packb(self.packer_meta)
                )
        except ValueError:
            logger.warning(
                "%s does not support writing string keys for length.",
                self.target_pack_type,
            )
        self.pack_file.close()

    def pack_data(self, idx):
        return msgpack.packb(self.dataset[idx])


@OBJECT_REGISTRY.register
class NuscenesMonoDataset(NuscenesDataset):
    """Mono-camera 3D detection dataset from packed NuScenes data.

    Each sample is expanded into ``num_cams`` items (one per camera view).
    """

    def __init__(self, **kwargs):
        super(NuscenesMonoDataset, self).__init__(**kwargs)
        self.num_cams = len(CAM_NAMES)

    def __len__(self):
        return len(self.samples) * self.num_cams

    def __getitem__(self, item):

        index = item // self.num_cams
        cam_idx = item % self.num_cams

        sample = self._decode(self.pack_file, self.samples[index])
        sample = NuscenesSample(sample)
        data = sample.get_mono_by_index(cam_idx)
        data["img"] = np.array(data["img"])
        data["scale_factor"] = 1.0
        data["gt_bboxes_3d"] = np.array(data["gt_bboxes_3d"]).reshape((-1, 10))
        data["gt_labels_3d"] = torch.from_numpy(data["gt_bboxes_3d"][:, -1])
        data["gt_bboxes_3d"] = torch.from_numpy(data["gt_bboxes_3d"][:, :-1])
        data["gt_bboxes_3d"] = CameraInstance3DBoxes(
            data["gt_bboxes_3d"],
            box_dim=data["gt_bboxes_3d"].shape[-1],
            origin=(0.5, 0.5, 0.5),
        )
        data["gt_labels"] = data["gt_labels_3d"]
        data["depths"] = torch.from_numpy(np.array(data.pop("depth")))
        data["cam2img"] = data.pop("camera_intrinsic")
        data["centers2d"] = torch.from_numpy(np.array(data.pop("center2d")))
        data["attr_labels"] = torch.tensor(data["attr_labels"])
        data["gt_bboxes"] = torch.from_numpy(np.array(data["gt_bboxes"]))
        data["layout"] = "hwc"
        data["color_space"] = "rgb"
        data["filename"] = data["file_name"]
        if self.transforms is not None:
            data = self.transforms(data)

        return data
