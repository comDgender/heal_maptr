# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
"""Precomputed NuScenes map vector GT pack (aligned with base/lidar packs)."""

import logging
from typing import Optional, Sequence, Tuple

import msgpack
import numpy as np
from hat_engine.data.datasets.data_packer import Packer
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.pack_type import PackTypeMapper, get_packtype_from_path
from hat_engine.utils.package_helper import require_packages
from shapely.geometry import LineString
from torch.utils.data import Dataset

from heal.data.datasets.nuscenes.nuscenes_map_geometry import (
    LiDARInstanceLines,
    VectorizedLocalMap,
)

try:
    from nuscenes.eval.common.utils import Quaternion
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits
except ImportError as e:  # pragma: no cover - surfaces the real failing import
    raise ImportError(
        "nuscenes-devkit is required for the map GT packer "
        "(NuscenesMapGtPacker / NuscenesMapGtDataset). The previous code "
        "silently set Quaternion=None here, which later crashed as "
        "'NoneType is not callable'. Install nuscenes-devkit==1.1.9 with a "
        "compatible numpy/pandas."
    ) from e

logger = logging.getLogger(__name__)

__all__ = [
    "NuscenesMapGtPackParser",
    "NuscenesMapGtPacker",
    "NuscenesMapGtDataset",
    "build_lidar_instance_lines",
    "decode_map_gt_sample",
]


def build_lidar_instance_lines(
    instance_coords: Sequence,
    patch_size: Tuple[float, float],
    fixed_num: int,
    padding_value: int = -10000,
    sample_dist: int = 1,
    num_samples: int = 250,
    padding: bool = False,
) -> LiDARInstanceLines:
    """Rebuild :class:`LiDARInstanceLines` from packed line coordinates."""
    instance_list = [
        LineString(np.array(coords, dtype=np.float64))
        for coords in instance_coords
    ]
    return LiDARInstanceLines(
        instance_list,
        sample_dist=sample_dist,
        num_samples=num_samples,
        padding=padding,
        fixed_num=fixed_num,
        padding_value=padding_value,
        patch_size=patch_size,
    )


def decode_map_gt_sample(raw: bytes) -> dict:
    """Decode one map-GT pack entry from msgpack bytes."""

    def _decode_hook(obj):
        new_obj = {}
        for key, value in obj.items():
            key = key.decode("utf-8") if isinstance(key, bytes) else key
            if isinstance(value, bytes):
                new_obj[key] = value.decode("utf-8")
            else:
                new_obj[key] = value
        return new_obj

    return msgpack.unpackb(raw, object_hook=_decode_hook, raw=True)


class NuscenesMapGtPackParser(object):
    """Parse map vector GT using the same sample order as ``NuscenesPackParser``.

    Only vector GT (``gt_labels_map`` / line coordinates) is packed. PV seg
    masks stay online because they depend on image augmentations at train time.
    """

    @require_packages(
        "nuscenes", raise_msg="Please `pip3 install nuscenes-devkit`"
    )
    def __init__(
        self,
        version: str,
        src_data_dir: str,
        split_name: str = "val",
        map_path: Optional[str] = None,
        pc_range: Optional[Sequence[float]] = None,
        map_classes: Sequence[str] = ("divider", "ped_crossing", "boundary"),
        fixed_ptsnum_per_line: int = 20,
        bev_size: Tuple[int, int] = (100, 50),
        padding_value: int = -10000,
        use_lidar_gt: bool = True,
    ):
        self.nusc = NuScenes(
            version=version, dataroot=src_data_dir, verbose=True
        )
        self.root_path = src_data_dir
        self.use_lidar_gt = use_lidar_gt
        self.pc_range = list(pc_range)
        self.map_classes = tuple(map_classes)
        self.fixed_ptsnum_per_line = fixed_ptsnum_per_line
        self.bev_size = tuple(bev_size)
        self.padding_value = padding_value

        patch_h = self.pc_range[4] - self.pc_range[1]
        patch_w = self.pc_range[3] - self.pc_range[0]
        self.patch_size = (patch_h, patch_w)
        self.vector_map = VectorizedLocalMap(
            map_path,
            patch_size=self.patch_size,
            map_classes=self.map_classes,
            fixed_ptsnum_per_line=fixed_ptsnum_per_line,
            padding_value=padding_value,
            canvas_size=bev_size,
            aux_seg=None,
        )

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
            logger.info("process map-gt scene %s %s ...", loc, scene_token)
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
            "pc_range": self.pc_range,
            "map_classes": list(self.map_classes),
            "fixed_ptsnum_per_line": self.fixed_ptsnum_per_line,
            "bev_size": list(self.bev_size),
            "padding_value": self.padding_value,
            "use_lidar_gt": self.use_lidar_gt,
        }

    def _lidar_pose(self, sample: dict):
        lidar_token = sample["data"]["LIDAR_TOP"]
        sd_rec = self.nusc.get("sample_data", lidar_token)
        cs_rec = self.nusc.get(
            "calibrated_sensor", sd_rec["calibrated_sensor_token"]
        )
        pose_rec = self.nusc.get("ego_pose", sd_rec["ego_pose_token"])

        lidar2ego = np.eye(4, dtype=np.float64)
        lidar2ego[:3, :3] = Quaternion(cs_rec["rotation"]).rotation_matrix
        lidar2ego[:3, 3] = cs_rec["translation"]

        ego2global = np.eye(4, dtype=np.float64)
        ego2global[:3, :3] = Quaternion(pose_rec["rotation"]).rotation_matrix
        ego2global[:3, 3] = pose_rec["translation"]

        lidar2global = ego2global @ lidar2ego
        translation = list(lidar2global[:3, 3])
        rotation = list(Quaternion(matrix=lidar2global).q)
        return translation, rotation

    def _ego_pose(self, sample: dict):
        lidar_token = sample["data"]["LIDAR_TOP"]
        sd_rec = self.nusc.get("sample_data", lidar_token)
        pose_rec = self.nusc.get("ego_pose", sd_rec["ego_pose_token"])
        return pose_rec["translation"], pose_rec["rotation"]

    def _gen_info(self, sample_token: str):
        logger.info("process map-gt sample %s ...", sample_token)
        sample = self.nusc.get("sample", sample_token)
        scene = self.nusc.get("scene", sample["scene_token"])
        location = self.nusc.get("log", scene["log_token"])["location"]

        if self.use_lidar_gt:
            translation, rotation = self._lidar_pose(sample)
        else:
            translation, rotation = self._ego_pose(sample)

        anns_results = self.vector_map.gen_vectorized_samples(
            location,
            translation,
            rotation,
            img_shape=None,
            homo2img=None,
            num_cam=6,
        )
        gt_instance = anns_results["gt_vecs_pts_loc"].instance_list
        gt_instance_coords = [
            np.array(line.coords, dtype=np.float64).tolist()
            for line in gt_instance
        ]
        return {
            "sample_token": sample_token,
            "scene_token": sample["scene_token"],
            "gt_labels_map": anns_results["gt_vecs_label"],
            "gt_instance_coords": gt_instance_coords,
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        return self._gen_info(self.samples[index])


class NuscenesMapGtPacker(Packer):
    """Pack precomputed map vector GT into a separate LMDB."""

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
        map_path: Optional[str] = None,
        pc_range: Optional[Sequence[float]] = None,
        map_classes: Sequence[str] = ("divider", "ped_crossing", "boundary"),
        fixed_ptsnum_per_line: int = 20,
        bev_size: Tuple[int, int] = (100, 50),
        padding_value: int = -10000,
        use_lidar_gt: bool = True,
        **kwargs,
    ):
        self.dataset = NuscenesMapGtPackParser(
            version=version,
            src_data_dir=src_data_dir,
            split_name=split_name,
            map_path=map_path,
            pc_range=pc_range,
            map_classes=map_classes,
            fixed_ptsnum_per_line=fixed_ptsnum_per_line,
            bev_size=bev_size,
            padding_value=padding_value,
            use_lidar_gt=use_lidar_gt,
        )
        self.packer_meta = self.dataset._get_meta()
        num_samples = len(self.dataset)
        logger.info(
            "Packing NuScenes map GT %s %s, total samples %d",
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
class NuscenesMapGtDataset(Dataset):
    """Map-GT-only dataset for use with :class:`MergeDataset`.

    Yields ``sample_token``, ``gt_labels_map``, and ``gt_instances``. PV seg
    should still be generated online in :class:`NuscenesMapDataset` when
    ``map_gt_path`` is set.
    """

    def __init__(
        self,
        data_path: str,
        pc_range: Sequence[float],
        fixed_ptsnum_per_line: int = 20,
        padding_value: int = -10000,
        pack_type: Optional[str] = None,
        pack_kwargs: Optional[dict] = None,
    ):
        self.root = data_path
        self.pc_range = list(pc_range)
        patch_h = self.pc_range[4] - self.pc_range[1]
        patch_w = self.pc_range[3] - self.pc_range[0]
        self.patch_size = (patch_h, patch_w)
        self.fixed_ptsnum_per_line = fixed_ptsnum_per_line
        self.padding_value = padding_value
        self.pack_type_name = pack_type
        self.kwargs = {} if pack_kwargs is None else pack_kwargs
        self.pack_type = self._resolve_pack_type(data_path)
        self.pack_file = self._open_pack(data_path)
        self.samples = self.pack_file.get_keys()

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

    def _decode(self, key):
        raw = self.pack_file.read(key)
        if raw is None:
            return None
        return decode_map_gt_sample(raw)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self._decode(self.samples[idx])
        gt_instances = build_lidar_instance_lines(
            sample["gt_instance_coords"],
            patch_size=self.patch_size,
            fixed_num=self.fixed_ptsnum_per_line,
            padding_value=self.padding_value,
        )
        return {
            "sample_token": sample["sample_token"],
            "gt_labels_map": sample["gt_labels_map"],
            "gt_instances": gt_instances,
        }

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
