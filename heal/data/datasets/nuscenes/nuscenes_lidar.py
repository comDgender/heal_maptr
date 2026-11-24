# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
"""Packed NuScenes lidar points + 3D annotations for lidar-only detection."""

import logging
import os
from typing import Callable, List, Optional

import msgpack
import numpy as np
from hat_engine.data.datasets.data_packer import Packer
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.pack_type import PackTypeMapper, get_packtype_from_path
from hat_engine.utils.package_helper import require_packages
from torch.utils.data import Dataset

try:
    from nuscenes.eval.common.utils import Quaternion
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits
except ImportError:
    Quaternion = None
    NuScenes = None
    splits = None

from heal.data.datasets.nuscenes.nuscenes_dataset import CLASSES, NameMapping

logger = logging.getLogger(__name__)

__all__ = [
    "NuscenesLidarPackParser",
    "NuscenesLidarPacker",
    "NuscenesLidarDataset",
]

LIDAR_LOAD_DIM = 5


class NuscenesLidarSample(object):
    """One decoded lidar sample from packed NuScenes LMDB."""

    def __init__(self, sample: dict, load_dim: int = LIDAR_LOAD_DIM):
        self.sample = sample
        self.load_dim = load_dim

    def _points_from_packed(self, raw) -> np.ndarray:
        if isinstance(raw, (bytes, bytearray)):
            return np.frombuffer(raw, dtype=np.float32).copy()
        return np.asarray(raw, dtype=np.float32).copy()

    def get_token(self):
        return self.sample["sample_token"]

    def _remove_close(self, points, radius=1.0):
        points_numpy = points
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def get_lidar_points(
        self,
        num_sweeps,
        load_dim,
        use_dim,
        time_dim=4,
        pad_empty_sweeps=True,
        remove_close=True,
        test_mode=False,
    ):
        points = self._points_from_packed(self.sample["lidar_points"])
        ts = self.sample["timestamp"] / 1e6

        points = points.reshape(-1, load_dim)
        points[:, time_dim] = 0

        sweeps = self.sample.get("sweeps", [])
        points_list = [points]

        if num_sweeps > 0:
            if pad_empty_sweeps and len(sweeps) == 0:
                for _ in range(num_sweeps):
                    if remove_close:
                        points_list.append(self._remove_close(points))
                    else:
                        points_list.append(points)
            else:
                if len(sweeps) <= num_sweeps:
                    choices = np.arange(len(sweeps))
                elif test_mode:
                    choices = np.arange(num_sweeps)
                else:
                    choices = np.random.choice(
                        len(sweeps), num_sweeps, replace=False
                    )
                for idx in choices:
                    sweep = sweeps[idx]
                    points_sweep = self._points_from_packed(
                        sweep["lidar_points"]
                    )
                    points_sweep = points_sweep.reshape(-1, load_dim)
                    if remove_close:
                        points_sweep = self._remove_close(points_sweep)
                    sweep_ts = sweep["timestamp"] / 1e6
                    points_sweep[:, :3] = (
                        points_sweep[:, :3]
                        @ np.array(sweep["sensor2lidar_rotation"]).T
                    )
                    points_sweep[:, :3] += np.array(
                        sweep["sensor2lidar_translation"]
                    )
                    points_sweep[:, time_dim] = ts - sweep_ts
                    points_list.append(points_sweep)
            points = np.concatenate(points_list)
        points = points[:, use_dim]
        return points

    def get_lidar_ann_info(
        self, use_valid_flag=False, with_velocity=True, classes=None
    ):
        info = self.sample
        if use_valid_flag:
            mask = np.array(info["valid_flag"], dtype=bool)
        else:
            mask = np.array(info["num_lidar_pts"]) > 0
        gt_bboxes_3d = np.array(info["gt_boxes"]).reshape(-1, 7)[mask]

        gt_names_3d = []
        for obj in info["gt_names"]:
            if isinstance(obj, bytes):
                obj = obj.decode("utf-8")
            gt_names_3d.append(obj)
        gt_names_3d = np.array(gt_names_3d)[mask]

        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in classes:
                gt_labels_3d.append(classes.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if with_velocity:
            gt_velocity = np.array(info["gt_velocity"]).reshape(-1, 2)[mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        locs = gt_bboxes_3d[:, :3]
        dims = gt_bboxes_3d[:, 3:6]
        rots = gt_bboxes_3d[:, 6:7]
        vel = gt_bboxes_3d[:, 7:9]
        gt_bboxes_3d = np.concatenate(
            [locs, dims[:, [1, 0, 2]], vel, -rots - np.pi / 2], axis=-1
        )

        return dict(  # noqa: C408
            boxes=gt_bboxes_3d,
            labels=gt_labels_3d,
            names=gt_names_3d,
        )


class NuscenesLidarPackParser(object):
    """Parse lidar points + 3D GT, aligned with the base NuScenes pack order.

    Args:
        version: NuScenes version (e.g. ``v1.0-trainval``).
        src_data_dir: Raw NuScenes root directory.
        split_name: ``train`` or ``val`` split name.
        max_sweeps: Max number of historical lidar sweeps stored per sample.
        load_dim: Raw point feature dimension in ``.pcd.bin`` files.
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
        load_dim: int = LIDAR_LOAD_DIM,
    ):
        self.nusc = NuScenes(
            version=version, dataroot=src_data_dir, verbose=True
        )
        self.root_path = src_data_dir
        self.max_sweeps = max_sweeps
        self.load_dim = load_dim
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
            logger.info("process lidar scene %s %s ...", loc, scene_token)
            while next_sample_token:
                self.samples.append(next_sample_token)
                sample = self.nusc.get("sample", next_sample_token)
                next_sample_token = sample["next"]
                scene_len += 1
            self.scene2len[scene_token] = scene_len

    def _get_meta(self):
        return {
            "scene_info": self.scene2len,
            "scene_len": len(self.scenes),
            "data_len": len(self.samples),
            "load_dim": self.load_dim,
            "max_sweeps": self.max_sweeps,
        }

    def _read_points(self, lidar_path: str) -> np.ndarray:
        lidar_file = os.path.join(self.root_path, lidar_path)
        return np.fromfile(lidar_file, dtype=np.float32)

    def _obtain_sensor2top(
        self,
        sensor_token,
        l2e_t,
        l2e_r_mat,
        e2g_t,
        e2g_r_mat,
        sensor_type="lidar",
    ):
        sd_rec = self.nusc.get("sample_data", sensor_token)
        cs_record = self.nusc.get(
            "calibrated_sensor", sd_rec["calibrated_sensor_token"]
        )
        pose_record = self.nusc.get("ego_pose", sd_rec["ego_pose_token"])
        points = self._read_points(sd_rec["filename"])

        sens2ego_t = cs_record["translation"]
        sens2ego_q = cs_record["rotation"]
        ego2glob_t = pose_record["translation"]
        ego2glob_q = pose_record["rotation"]

        sweep = {
            "lidar_points": points.tobytes(),
            "type": sensor_type,
            "sample_data_token": sd_rec["token"],
            "sensor2ego_translation": sens2ego_t,
            "sensor2ego_rotation": sens2ego_q,
            "ego2global_translation": ego2glob_t,
            "ego2global_rotation": ego2glob_q,
            "timestamp": sd_rec["timestamp"],
        }

        # Row-vector convention: p' = p @ R. Compose sweep->global->ref-lidar.
        sens2ego_R = Quaternion(sens2ego_q).rotation_matrix
        ego2glob_R = Quaternion(ego2glob_q).rotation_matrix
        glob2ego_ref_R = np.linalg.inv(e2g_r_mat).T
        ego2lidar_ref_R = np.linalg.inv(l2e_r_mat).T
        glob2lidar_ref = glob2ego_ref_R @ ego2lidar_ref_R

        sweep_to_glob_R = sens2ego_R.T @ ego2glob_R.T
        R = sweep_to_glob_R @ glob2lidar_ref
        sweep_origin_glob = sens2ego_t @ ego2glob_R.T + ego2glob_t
        T = sweep_origin_glob @ glob2lidar_ref
        ref_origin = e2g_t @ glob2lidar_ref + l2e_t @ ego2lidar_ref_R
        T = T - ref_origin

        sweep["sensor2lidar_rotation"] = R.T.tolist()
        sweep["sensor2lidar_translation"] = T.tolist()
        return sweep

    def _collect_prev_sweeps(self, sd_rec, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat):
        sweeps = []
        cur = sd_rec
        while len(sweeps) < self.max_sweeps:
            prev_token = cur.get("prev") or ""
            if not prev_token:
                break
            sweeps.append(
                self._obtain_sensor2top(
                    prev_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, "lidar"
                )
            )
            cur = self.nusc.get("sample_data", prev_token)
        return sweeps

    def _boxes_to_gt_fields(self, boxes, sample, e2g_r_mat, l2e_r_mat):
        annotations = [
            self.nusc.get("sample_annotation", token)
            for token in sample["anns"]
        ]
        centers = np.asarray(
            [b.center for b in boxes], dtype=np.float64
        ).reshape(-1, 3)
        # nuScenes Box.wlh is (w, l, h); store as (l, w, h).
        sizes_wlh = np.asarray(
            [b.wlh for b in boxes], dtype=np.float64
        ).reshape(-1, 3)
        yaws = np.asarray(
            [b.orientation.yaw_pitch_roll[0] for b in boxes], dtype=np.float64
        ).reshape(-1, 1)

        vel_xy = np.asarray(
            [self.nusc.box_velocity(token)[:2] for token in sample["anns"]],
            dtype=np.float64,
        )
        # Rotate global XY velocity into the reference lidar frame.
        glob2lidar = np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
        if vel_xy.size:
            vel_xyz = np.concatenate(
                [vel_xy, np.zeros((vel_xy.shape[0], 1), dtype=np.float64)],
                axis=1,
            )
            vel_xy = (vel_xyz @ glob2lidar)[:, :2]

        mapped_names = [NameMapping.get(box.name, box.name) for box in boxes]
        gt_boxes = np.concatenate(
            [centers, sizes_wlh[:, [1, 0, 2]], yaws], axis=1
        )
        has_pts = np.asarray(
            [
                (anno["num_lidar_pts"] + anno["num_radar_pts"]) > 0
                for anno in annotations
            ],
            dtype=bool,
        ).reshape(-1)

        return {
            "gt_boxes": gt_boxes.tolist(),
            "gt_names": np.asarray(mapped_names).tolist(),
            "gt_velocity": vel_xy.reshape(-1, 2).tolist(),
            "num_lidar_pts": [a["num_lidar_pts"] for a in annotations],
            "num_radar_pts": [a["num_radar_pts"] for a in annotations],
            "valid_flag": has_pts.tolist(),
        }

    def _gen_lidar_info(self, sample: dict, info: dict):
        lidar_token = sample["data"]["LIDAR_TOP"]
        sd_rec = self.nusc.get("sample_data", lidar_token)
        points = self._read_points(sd_rec["filename"])
        info["lidar_points"] = points.tobytes()
        info["load_dim"] = self.load_dim

        cs_rec = self.nusc.get(
            "calibrated_sensor", sd_rec["calibrated_sensor_token"]
        )
        pose_rec = self.nusc.get("ego_pose", sd_rec["ego_pose_token"])
        info["lidar2ego_translation"] = cs_rec["translation"]
        info["lidar2ego_rotation"] = cs_rec["rotation"]
        info["ego2global_translation"] = pose_rec["translation"]
        info["ego2global_rotation"] = pose_rec["rotation"]

        l2e_r_mat = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        e2g_r_mat = Quaternion(info["ego2global_rotation"]).rotation_matrix
        l2e_t = info["lidar2ego_translation"]
        e2g_t = info["ego2global_translation"]

        info["sweeps"] = self._collect_prev_sweeps(
            sd_rec, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat
        )

        if self.test:
            return

        _, boxes, _ = self.nusc.get_sample_data(lidar_token)
        info.update(
            self._boxes_to_gt_fields(boxes, sample, e2g_r_mat, l2e_r_mat)
        )

    def _gen_info(self, sample_token: str):
        logger.info("process lidar sample %s ...", sample_token)
        sample = self.nusc.get("sample", sample_token)
        scene = self.nusc.get("scene", sample["scene_token"])
        loc = self.nusc.get("log", scene["log_token"])["location"]
        info = {
            "sample_token": sample_token,
            "scene_token": sample["scene_token"],
            "scene": sample["scene_token"],
            "location": loc,
            "prev": sample["prev"],
            "next": sample["next"],
            "timestamp": sample["timestamp"],
        }
        self._gen_lidar_info(sample, info)
        return info

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        return self._gen_info(self.samples[index])


class NuscenesLidarPacker(Packer):
    """Pack NuScenes lidar points + 3D GT into a separate LMDB.

    Args:
        version: NuScenes version string.
        src_data_dir: Raw NuScenes root directory.
        target_data_dir: Output pack directory.
        split_name: ``train`` or ``val``.
        num_workers: Parallel workers for packing.
        pack_type: Pack backend (e.g. ``lmdb``).
        max_sweeps: Max sweeps stored per sample.
        load_dim: Raw point dimension in ``.pcd.bin`` files.
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
        max_sweeps: int = 10,
        load_dim: int = LIDAR_LOAD_DIM,
        **kwargs,
    ):
        self.dataset = NuscenesLidarPackParser(
            version=version,
            src_data_dir=src_data_dir,
            split_name=split_name,
            max_sweeps=max_sweeps,
            load_dim=load_dim,
        )
        self.packer_meta = self.dataset._get_meta()
        num_samples = len(self.dataset)
        logger.info(
            "Packing NuScenes lidar %s %s, total samples %d",
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
class NuscenesLidarDataset(Dataset):
    """Lidar-only dataset reading the separate lidar LMDB pack.

    Each sample yields multi-sweep point clouds and 3D box annotations in
    lidar frame, following HAT ``NuscenesLidarDataset`` conventions.

    Args:
        data_path: Path to the packed lidar LMDB directory.
        num_sweeps: Number of sweeps loaded at runtime.
        load_dim: Dimension of raw packed points.
        use_dim: Subset of point dimensions to use.
        time_dim: Timestamp channel index in raw points.
        pad_empty_sweeps: Repeat keyframe when no sweeps are available.
        remove_close: Remove points near ego origin in each sweep.
        use_valid_flag: Filter boxes with ``valid_flag`` instead of lidar pts.
        with_velocity: Include velocity in box tensors.
        classes: Detection class names.
        test_mode: Use deterministic sweep selection and skip empty-GT filter.
        filter_empty_gt: Resample when no GT boxes remain after filtering.
        transforms: Optional transform pipeline on the output dict.
        pack_type: Pack backend when it cannot be inferred from path.
        pack_kwargs: Extra kwargs for the pack backend.
    """

    def __init__(
        self,
        data_path: str,
        num_sweeps: int = 10,
        load_dim: int = LIDAR_LOAD_DIM,
        use_dim: Optional[List[int]] = None,
        time_dim: int = 4,
        pad_empty_sweeps: bool = True,
        remove_close: bool = True,
        use_valid_flag: bool = False,
        with_velocity: bool = True,
        classes: Optional[List[str]] = None,
        test_mode: bool = False,
        filter_empty_gt: bool = True,
        transforms: Optional[Callable] = None,
        pack_type: Optional[str] = None,
        pack_kwargs: Optional[dict] = None,
    ):
        self.root = data_path
        self.transforms = transforms
        self.pack_type_name = pack_type
        self.kwargs = {} if pack_kwargs is None else pack_kwargs
        self.pack_type = self._resolve_pack_type(data_path)
        self.pack_file = self._open_pack(data_path)
        self.samples = self.pack_file.get_keys()
        self.meta = self._decode_meta()

        self.num_sweeps = num_sweeps
        self.load_dim = (
            load_dim if load_dim is not None else self._meta_load_dim()
        )
        self.use_dim = use_dim if use_dim is not None else [0, 1, 2, 4]
        self.time_dim = time_dim
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.use_valid_flag = use_valid_flag
        self.with_velocity = with_velocity
        self.CLASSES = self.get_classes(classes)
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt

        assert (
            max(self.use_dim) < self.load_dim
        ), f"Expect all used dimensions < {self.load_dim}, got {self.use_dim}"
        assert (
            self.time_dim < self.load_dim
        ), f"Expect timestamp dimension < {self.load_dim}, got {self.time_dim}"
        if not self.test_mode:
            self.flag = np.zeros(len(self), dtype=np.uint8)

    def _meta_load_dim(self) -> int:
        if self.meta and "load_dim" in self.meta:
            return self.meta["load_dim"]
        return LIDAR_LOAD_DIM

    def _resolve_pack_type(self, path):
        try:
            return get_packtype_from_path(path)
        except NotImplementedError:
            assert self.pack_type_name is not None, (
                "pack_type must be given when it cannot be inferred from the "
                f"path: {path}"
            )
            return PackTypeMapper[self.pack_type_name.lower()]

    def _open_pack(self, path):
        pack_file = self.pack_type(path, writable=False, **self.kwargs)
        pack_file.open()
        return pack_file

    def _decode_meta(self):
        raw = self.pack_file.read("__meta__")
        if raw is None:
            return None
        return msgpack.unpackb(raw, raw=False)

    def _decode(self, key):
        def _decode_hook(obj):
            new_obj = {}
            for k, v in obj.items():
                k = k.decode("utf-8") if isinstance(k, bytes) else k
                if k in ("lidar_points",):
                    new_obj[k] = v
                elif k == "sweeps" and isinstance(v, list):
                    sweeps = []
                    for sweep in v:
                        sweep_obj = {}
                        for sk, sv in sweep.items():
                            sk = (
                                sk.decode("utf-8")
                                if isinstance(sk, bytes)
                                else sk
                            )
                            if sk == "lidar_points":
                                sweep_obj[sk] = sv
                            elif isinstance(sv, bytes):
                                sweep_obj[sk] = sv.decode("utf-8")
                            else:
                                sweep_obj[sk] = sv
                        sweeps.append(sweep_obj)
                    new_obj[k] = sweeps
                elif isinstance(v, bytes):
                    new_obj[k] = v.decode("utf-8")
                else:
                    new_obj[k] = v
            return new_obj

        raw = self.pack_file.read(key)
        if raw is None:
            return None
        return msgpack.unpackb(raw, object_hook=_decode_hook, raw=True)

    def _decode_ann_for_cat_ids(self, key):
        """Decode only annotation fields needed by ``get_cat_ids``.

        Skips keeping ``lidar_points`` / ``sweeps`` payloads so CBGS histogram
        builds spend less CPU/RAM. LMDB still returns the full value blob.
        """

        keep = {"gt_names", "valid_flag"}

        def _decode_hook(obj):
            new_obj = {}
            for k, v in obj.items():
                k = k.decode("utf-8") if isinstance(k, bytes) else k
                if k not in keep:
                    continue
                if isinstance(v, bytes):
                    new_obj[k] = v.decode("utf-8")
                elif isinstance(v, list):
                    decoded = []
                    for item in v:
                        if isinstance(item, bytes):
                            decoded.append(item.decode("utf-8"))
                        else:
                            decoded.append(item)
                    new_obj[k] = decoded
                else:
                    new_obj[k] = v
            return new_obj

        raw = self.pack_file.read(key)
        if raw is None:
            return None
        return msgpack.unpackb(raw, object_hook=_decode_hook, raw=True)

    @classmethod
    def get_classes(cls, classes=None):
        if classes is None:
            return CLASSES
        if isinstance(classes, (tuple, list)):
            return classes
        raise ValueError(f"Unsupported type {type(classes)} of classes.")

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def prepare_data(self, item):
        sample = self._decode(self.samples[item])
        data = {
            "sample_token": sample["sample_token"],
            "lidar": {
                "points": None,
                "annotations": None,
            },
            "metadata": {
                "lidar2ego_translation": sample["lidar2ego_translation"],
                "lidar2ego_rotation": sample["lidar2ego_rotation"],
                "ego2global_translation": sample["ego2global_translation"],
                "ego2global_rotation": sample["ego2global_rotation"],
                "sample_token": sample["sample_token"],
                "image_prefix": None,
                "num_point_features": len(self.use_dim),
            },
            "mode": "val" if self.test_mode else "train",
        }

        lidar_sample = NuscenesLidarSample(sample, load_dim=self.load_dim)
        data["lidar"]["points"] = lidar_sample.get_lidar_points(
            self.num_sweeps,
            self.load_dim,
            self.use_dim,
            self.time_dim,
            self.pad_empty_sweeps,
            self.remove_close,
            self.test_mode,
        )
        if "gt_boxes" in sample:
            data["lidar"]["annotations"] = lidar_sample.get_lidar_ann_info(
                self.use_valid_flag,
                self.with_velocity,
                self.CLASSES,
            )
        else:
            data["lidar"]["annotations"] = dict(  # noqa: C408
                boxes=np.zeros((0, 9), dtype=np.float32),
                labels=np.zeros((0,), dtype=np.int64),
                names=np.array([]),
            )

        if self.transforms is not None:
            data = self.transforms(data)
        return data

    def _gt_boxes_from_sample(self, data):
        annotations = data.get("annotations")
        if isinstance(annotations, dict):
            for key in ("gt_boxes", "boxes"):
                if key in annotations:
                    return annotations[key]

        lidar = data.get("lidar")
        if isinstance(lidar, dict):
            lidar_ann = lidar.get("annotations")
            if isinstance(lidar_ann, dict):
                for key in ("boxes", "gt_boxes"):
                    if key in lidar_ann:
                        return lidar_ann[key]

        return np.zeros((0, 9), dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def get_cat_ids(self, idx: int):
        sample = self._decode_ann_for_cat_ids(self.samples[idx])
        gt_names_3d = []
        for obj in sample.get("gt_names", []):
            if isinstance(obj, bytes):
                obj = obj.decode("utf-8")
            gt_names_3d.append(obj)
        gt_names_3d = np.array(gt_names_3d)
        if self.use_valid_flag:
            mask = np.array(sample.get("valid_flag", []), dtype=bool)
            if len(mask) == len(gt_names_3d):
                gt_names = set(gt_names_3d[mask])
            else:
                gt_names = set(gt_names_3d)
        else:
            gt_names = set(gt_names_3d)

        return [self.cat2id[name] for name in gt_names if name in self.CLASSES]

    def __getitem__(self, item):
        if isinstance(item, dict):
            idx = item["idx"]
        else:
            idx = item

        if self.test_mode:
            return self.prepare_data(idx)

        while True:
            data = self.prepare_data(idx)
            if (
                self.filter_empty_gt
                and len(self._gt_boxes_from_sample(data)) == 0
            ):
                idx = self._rand_another(idx)
                continue
            return data

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.pack_file is not None:
            self.pack_file.close()
        state["pack_file"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.pack_file = self._open_pack(self.root)
        self.samples = self.pack_file.get_keys()
