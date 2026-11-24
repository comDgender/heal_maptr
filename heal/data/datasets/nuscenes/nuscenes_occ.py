# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import logging
import os

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

logger = logging.getLogger(__name__)

__all__ = [
    "NuscenesOccPackParser",
    "NuscenesOccPacker",
    "NuscenesOccDataset",
]

CAM_NAMES = (
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
)

# occ voxel grid shape (semantics / masks).
OCC_SHAPE = (200, 200, 16)


def _ego2img_matrix(
    camera_intrinsic, sensor2ego_translation, sensor2ego_rotation
):
    """Ego -> image homography for one camera (matches old occ sampler)."""
    rotation = Quaternion(sensor2ego_rotation).rotation_matrix
    ego2sensor_r = np.linalg.inv(rotation)
    ego2sensor_t = np.array(sensor2ego_translation) @ ego2sensor_r.T
    ego2sensor = np.eye(4)
    ego2sensor[:3, :3] = ego2sensor_r.T
    ego2sensor[3, :3] = -np.array(ego2sensor_t)
    intrinsic = np.array(camera_intrinsic)
    viewpad = np.eye(4)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
    return viewpad @ ego2sensor.T


def _cam2ego_matrix(sensor2ego_translation, sensor2ego_rotation):
    """Camera -> ego transform for one camera (matches old occ sampler)."""
    cam2ego = np.eye(4)
    cam2ego[:3, :3] = Quaternion(sensor2ego_rotation).rotation_matrix
    cam2ego[:3, 3] = sensor2ego_translation
    return cam2ego


class NuscenesOccPackParser(object):
    """Parse occ3d GT + camera geometry, aligned with the base pack order.

    Args:
        version: NuScenes version (e.g. ``v1.0-trainval``).
        src_data_dir: Raw NuScenes root (also holds ``occ_gt_dir``).
        split_name: ``train`` or ``val`` split name.
        occ_gt_dir: Directory under ``src_data_dir`` holding occ labels, laid
            out as ``<occ_gt_dir>/<scene_name>/<sample_token>/labels.npz``.
    """

    @require_packages(
        "nuscenes", raise_msg="Please `pip3 install nuscenes-devkit`"
    )
    def __init__(
        self,
        version: str,
        src_data_dir: str,
        split_name: str = "val",
        occ_gt_dir: str = "occ3d/gts",
    ):
        self.nusc = NuScenes(
            version=version, dataroot=src_data_dir, verbose=True
        )
        self.root_path = src_data_dir
        self.occ_gt_dir = occ_gt_dir
        available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini"]
        assert version in available_vers
        if version == "v1.0-trainval":
            train_scenes, val_scenes = splits.train, splits.val
        elif version == "v1.0-test":
            train_scenes, val_scenes = splits.test, []
        else:
            train_scenes, val_scenes = splits.mini_train, splits.mini_val

        scene_names = train_scenes if split_name == "train" else val_scenes
        # Same iteration order as NuscenesPackParser => index-aligned packs.
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
            logger.info("process occ scene %s %s ...", loc, scene_token)
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
        }

    def _gen_occ_gt(self, scene: dict, info: dict):
        occ_gt_path = os.path.join(
            self.root_path,
            self.occ_gt_dir,
            scene["name"],
            info["sample_token"],
            "labels.npz",
        )
        occ_labels = np.load(occ_gt_path)
        info["voxel_semantics"] = occ_labels["semantics"].tobytes()
        info["mask_lidar"] = occ_labels["mask_lidar"].tobytes()
        info["mask_camera"] = occ_labels["mask_camera"].tobytes()

    def _gen_cam_geometry(self, sample: dict, info: dict):
        """Compute per-camera ego2img / cam2ego in CAM_NAMES order."""
        ego2img = []
        cam2ego = []
        for name in CAM_NAMES:
            token = sample["data"][name]
            sd_rec = self.nusc.get("sample_data", token)
            cs_rec = self.nusc.get(
                "calibrated_sensor", sd_rec["calibrated_sensor_token"]
            )
            intrinsic = cs_rec["camera_intrinsic"]
            s2e_t = cs_rec["translation"]
            s2e_r = cs_rec["rotation"]
            ego2img.append(_ego2img_matrix(intrinsic, s2e_t, s2e_r).tolist())
            cam2ego.append(_cam2ego_matrix(s2e_t, s2e_r).tolist())
        info["ego2img"] = ego2img
        info["cam2ego"] = cam2ego

    def _gen_info(self, sample_token: str):
        logger.info("process occ sample %s ...", sample_token)
        sample = self.nusc.get("sample", sample_token)
        scene = self.nusc.get("scene", sample["scene_token"])
        info = {
            "sample_token": sample_token,
            "scene_token": sample["scene_token"],
        }
        self._gen_occ_gt(scene, info)
        self._gen_cam_geometry(sample, info)
        return info

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        return self._gen_info(self.samples[index])


class NuscenesOccPacker(Packer):
    """Pack occ3d GT + camera geometry into a separate LMDB aligned with base.

    Args:
        version: NuScenes version string.
        src_data_dir: Raw NuScenes root directory.
        target_data_dir: Output pack directory.
        split_name: ``train`` or ``val``.
        num_workers: Parallel workers for packing.
        pack_type: Pack backend (e.g. ``lmdb``).
        occ_gt_dir: Directory under ``src_data_dir`` holding occ labels.
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
        occ_gt_dir: str = "occ3d/gts",
        **kwargs,
    ):
        self.dataset = NuscenesOccPackParser(
            version=version,
            src_data_dir=src_data_dir,
            split_name=split_name,
            occ_gt_dir=occ_gt_dir,
        )
        self.packer_meta = self.dataset._get_meta()
        num_samples = len(self.dataset)
        logger.info(
            "Packing NuScenes occ %s %s, total samples %d",
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
class NuscenesOccDataset(Dataset):
    """Occ-only dataset: reads the separate occ pack for flashocc.

    Each sample yields ``sample_token`` + ``scene_token`` (meta), the decoded
    ``gt_occ_info`` (``voxel_semantics`` / ``mask_lidar`` / ``mask_camera``),
    and the per-camera ``ego2img`` / ``cam2ego`` that the base
    ``NuscenesBevDataset`` does not surface. Designed to be combined with the
    base camera dataset via
    :class:`heal.data.datasets.merge_dataset.MergeDataset` (join on
    ``sample_token``).

    Args:
        data_path: Path to the packed occ LMDB directory.
        transforms: Unused here (kept for API symmetry); transforms run once at
            the :class:`MergeDataset` level after merge. Leave as ``None``.
        pack_type: Pack backend name when it cannot be inferred from the path.
        pack_kwargs: Extra kwargs passed to the pack file backend.
        occ_shape: Voxel grid shape used to reshape occ buffers.
    """

    def __init__(
        self,
        data_path: str,
        transforms=None,
        pack_type=None,
        pack_kwargs=None,
        occ_shape=OCC_SHAPE,
    ):
        self.root = data_path
        self.transforms = transforms
        self.pack_type_name = pack_type
        self.kwargs = {} if pack_kwargs is None else pack_kwargs
        self.occ_shape = tuple(occ_shape)
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
        def _decode_hook(obj):
            new_obj = {}
            for k, v in obj.items():
                k = k.decode("utf-8") if isinstance(k, bytes) else k
                if k in ("voxel_semantics", "mask_lidar", "mask_camera"):
                    new_obj[k] = v
                elif isinstance(v, bytes):
                    new_obj[k] = v.decode("utf-8")
                else:
                    new_obj[k] = v
            return new_obj

        raw = self.pack_file.read(key)
        if raw is None:
            return None
        return msgpack.unpackb(raw, object_hook=_decode_hook, raw=True)

    def _decode_occ(self, sample):
        return {
            name: np.frombuffer(sample[name], dtype=np.uint8).reshape(
                self.occ_shape
            )
            for name in ("voxel_semantics", "mask_lidar", "mask_camera")
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self._decode(self.samples[idx])
        data = {
            "sample_token": sample["sample_token"],
            "scene_token": sample["scene_token"],
            "gt_occ_info": self._decode_occ(sample),
            "ego2img": [np.array(m) for m in sample["ego2img"]],
            "cam2ego": [np.array(m) for m in sample["cam2ego"]],
        }
        if self.transforms is not None:
            data = self.transforms(data)
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
