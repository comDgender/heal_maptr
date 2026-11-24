# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from __future__ import annotations
import logging
import os
import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import msgpack
import msgpack_numpy
import numpy as np
import yaml
from hat_engine.data.datasets.data_packer import Packer
from tqdm import tqdm

msgpack_numpy.patch()

logger = logging.getLogger(__name__)


class NavsimPerceptionParser(object):
    def __init__(
        self,
        data_root: str,
        filter_file_path: str,
        mode: str = "train",
        split: str = "trainval",
        sensor_names: Optional[List[str]] = None,
        navsim_version: str = "v1",
        world_size: int = -1,
        rank_id: int = -1,
    ):
        self.root = Path(data_root)
        self.sensor_names = sensor_names
        self.scene_length_dict = OrderedDict()
        self.frame_meta_all = []
        self.log_file_paths = {}
        self.extend_scenes_dict = {}
        self.scene_meta_dict = {}
        self.scene_info = {}
        self._scenes_cache = OrderedDict()
        self._max_cache_size = 5
        self.filter_file_path = filter_file_path
        self.mode = mode
        assert mode in [
            "train",
            "val",
            "test",
        ], "`mode` must be one of [train, val, test]"
        world_size = int(world_size)
        rank_id = int(rank_id)

        if navsim_version == "v1":
            self.logs_path = self.root / "navsim_logs" / split
            self.blobs_path = self.root / "sensor_blobs" / split

            if not self.logs_path.exists():
                raise FileNotFoundError(
                    f"NavSim logs not found at {self.logs_path}"
                )
            if self.filter_file_path and os.path.exists(self.filter_file_path):
                with open(self.filter_file_path, "r") as f:
                    filter_cfg = yaml.safe_load(f)
                field = (
                    "train_log_names"
                    if self.mode == "train"
                    else "val_log_names"
                )
                log_names = filter_cfg.get(field, [])
                all_log_files = sorted(
                    [self.logs_path / f"{name}.pkl" for name in log_names]
                )
                logger.info(
                    "Filter %s field=%s -> %d log names from %s",
                    self.mode,
                    field,
                    len(log_names),
                    self.filter_file_path,
                )
            else:
                all_log_files = sorted(
                    [f for f in self.logs_path.iterdir() if f.suffix == ".pkl"]
                )
                logger.info(
                    "No filter file (path=%s), use all %d pkl under %s",
                    self.filter_file_path or "<empty>",
                    len(all_log_files),
                    self.logs_path,
                )
            # Distribute log_files across ranks when world_size > 1
            if world_size > 1:
                total_logs = len(all_log_files)
                logs_per_rank = (total_logs + world_size - 1) // world_size
                start_idx = rank_id * logs_per_rank
                end_idx = min(start_idx + logs_per_rank, total_logs)
                self.log_files = all_log_files[start_idx:end_idx]
                logger.info(
                    "Rank %d/%d: %d/%d logs (indices %d-%d, ~%d logs/rank)",
                    rank_id,
                    world_size,
                    len(self.log_files),
                    total_logs,
                    start_idx,
                    end_idx - 1 if end_idx > start_idx else start_idx,
                    logs_per_rank,
                )
            else:
                self.log_files = all_log_files
                logger.info(
                    "Rank %d/%d: using all %d logs (no sharding)",
                    rank_id,
                    world_size,
                    len(self.log_files),
                )
            self.token_list_dict = OrderedDict()

            for log_file in tqdm(
                self.log_files, desc=f"rank{rank_id}-scan-logs"
            ):
                # Load temporarily to extract metadata, then release
                with open(log_file, "rb") as f:
                    scene_data = pickle.load(f)
                self.log_file_paths[log_file.stem] = log_file
                frame_info_list = []
                self.scene_info[log_file.stem] = len(scene_data)
                for i, frame in enumerate(scene_data):
                    meta = self._get_meta_of_frame(i, frame)
                    meta["log_key"] = log_file.stem
                    frame_info_list.append(meta)
                token_dict_frame_info = self.list_to_token_dict(
                    frame_info_list
                )
                self.token_list_dict[log_file.stem] = token_dict_frame_info
                self.frame_meta_all.extend(frame_info_list)
                del scene_data
            logger.info(
                "Rank %d parse done: %d frames from %d logs",
                rank_id,
                len(self.frame_meta_all),
                len(self.log_files),
            )
        else:
            raise ValueError(f"Unsupported navsim_version: {navsim_version}")

    def list_to_token_dict(self, frame_info):
        return {f["token"]: f for f in frame_info}

    def _get_meta_of_frame(
        self,
        idx,
        frame: Dict[str, Any],
    ) -> Dict:

        meta_keys = [
            "token",
            "frame_idx",
            "timestamp",
            "log_name",
            "map_location",
            "scene_token",
        ]
        filtered_meta = {key: frame[key] for key in meta_keys if key in frame}
        filtered_meta["has_route"] = len(frame.get("roadblock_ids", [])) > 0
        filtered_meta["scene_idx"] = idx
        return filtered_meta

    def _load_scenes_dict(self, log_key: str):
        if log_key in self._scenes_cache:
            self._scenes_cache.move_to_end(log_key)
            return self._scenes_cache[log_key]
        log_file = self.log_file_paths[log_key]
        with open(log_file, "rb") as f:
            scene_data = pickle.load(f)
        self._scenes_cache[log_key] = scene_data
        if len(self._scenes_cache) > self._max_cache_size:
            self._scenes_cache.popitem(last=False)
        return scene_data

    def __len__(self):
        return len(self.frame_meta_all) + len(self.extend_scenes_dict)

    def _get_meta(self):
        """Get meta information of dataset."""
        meta = {
            "scene_info": self.scene_info,
            "scene_info_list": self.frame_meta_all,
            "extend_scenes_list": list(self.extend_scenes_dict.keys()),
            "scene_meta_dict": self.scene_meta_dict,
        }
        return meta

    def __getitem__(self, idx: int):
        if idx >= len(self.frame_meta_all):
            extend_idx = idx - len(self.frame_meta_all)
            extend_log_name = list(self.extend_scenes_dict.keys())[extend_idx]
            extend_data = self.extend_scenes_dict[extend_log_name]
            return extend_data
        meta = self.frame_meta_all[idx]
        log_key = meta["log_key"]
        scene_idx = meta["scene_idx"]
        scene_dict_list = self._load_scenes_dict(log_key)
        single_frame_data = scene_dict_list[scene_idx]
        sensor_names = self.sensor_names or []
        if "cams" in single_frame_data:
            camera_dict = single_frame_data["cams"]
        elif "camera_dict" in single_frame_data:
            camera_dict = single_frame_data["camera_dict"]
        cameras = {}
        for camera_name in camera_dict.keys():
            camera_identifier = camera_name.lower()
            camera_dict[camera_name]["data_path"] = str(
                camera_dict[camera_name]["data_path"]
            )
            if camera_identifier in sensor_names:
                image_path = (
                    self.blobs_path / camera_dict[camera_name]["data_path"]
                )

                with open(image_path, "rb") as f:
                    cameras[camera_name] = f.read()
        single_frame_data["camera_data"] = cameras

        if "lidar_pc" in sensor_names:
            lidar_path = str(self.blobs_path / single_frame_data["lidar_path"])

            with open(lidar_path, "rb") as fp:
                lidar_data = fp.read()
            single_frame_data["lidar_data"] = lidar_data
        if "occ_gt_final_path" in single_frame_data:
            if single_frame_data["occ_gt_final_path"] is None:
                single_frame_data["voxel_semantics"] = None
            if not isinstance(single_frame_data["occ_gt_final_path"], str):
                single_frame_data["voxel_semantics"] = None
            else:
                occ_path = os.path.join(
                    self.root, single_frame_data["occ_gt_final_path"]
                )
                if os.path.exists(occ_path):
                    single_frame_data["voxel_semantics"] = np.load(
                        occ_path
                    ).tobytes()
                else:
                    single_frame_data["voxel_semantics"] = None
        else:
            single_frame_data["voxel_semantics"] = None
        if "lidar_path" in single_frame_data and isinstance(
            single_frame_data["lidar_path"], Path
        ):
            single_frame_data["lidar_path"] = str(
                single_frame_data["lidar_path"]
            )
        return single_frame_data


class NavsimPerceptionPacker(Packer):
    def __init__(
        self,
        src_data_dir: str,
        target_data_dir: str,
        split_name: str,
        filter_file_path: str,
        mode: str = "train",
        num_workers: int = 20,
        pack_type: str = "raw",
        navsim_version: str = "v1",
        max_samples: int = None,
        world_size: int = -1,
        rank_id: int = -1,
        **kwargs,
    ):
        assert mode in [
            "train",
            "val",
            "test",
        ], "`mode` must be one of [train, val, test]"
        sensor_names = [
            "cam_f0",
            "cam_l0",
            "cam_l1",
            "cam_l2",
            "cam_r0",
            "cam_r1",
            "cam_r2",
            "cam_b0",
            # "lidar_pc"
        ]
        self.dataset = NavsimPerceptionParser(
            data_root=src_data_dir,
            filter_file_path=filter_file_path,
            mode=mode,
            split=split_name,
            sensor_names=sensor_names,
            navsim_version=navsim_version,
            world_size=world_size,
            rank_id=rank_id,
        )
        self.navsim_version = navsim_version
        self.packer_meta = self.dataset._get_meta()
        num_samples = len(self.dataset)
        if max_samples is not None:
            num_samples = min(num_samples, max_samples)
        kwargs["map_size"] = 1024**4 * 10
        super(NavsimPerceptionPacker, self).__init__(
            target_data_dir, num_samples, pack_type, num_workers, **kwargs
        )

    def _write_length(self, num):
        """
        Write max length to target format file.

        Args:
            num (int): Max length for packing data.
        """
        if self.packer_meta is not None:
            try:
                self.pack_file.write(
                    "__meta__", msgpack.packb(self.packer_meta)
                )
                self.pack_file.write(
                    "__len__", "{}".format(num).encode("ascii")
                )
            except ValueError:
                logger.warning(
                    f"{self.target_pack_type} is not supported "
                    + "for write length mainly due to the string key.",
                )

    def pack_data(self, idx):
        sample = self.dataset[idx]
        try:
            if self.navsim_version == "v1":
                data = msgpack.packb(sample, default=msgpack_numpy.encode)
            else:
                data = pickle.dumps(sample, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            keys = list(sample.keys())
            type_list = [type(sample[k]) for k in sample.keys()]
            print(keys)
            print(type_list)

            data = msgpack.packb(sample, default=msgpack_numpy.encode)
        return data


def save_token_to_idx_pkl(
    lmdb_dirs: Sequence[str],
    pkl_path: Optional[str] = None,
) -> Dict[str, int]:
    """Build ``{sample_token: global_idx}`` from one or more perception LMDBs.

    Indices follow ``NavsimPerceptionMultitaskDataset`` order: each LMDB
    contributes samples ``0 .. n-1`` in turn, then the next shard is
    appended. Prefer reading ``__meta__`` / ``scene_info_list`` (fast);
    fall back to scanning sample records if meta is missing.

    When ``lmdb_dirs`` has multiple ``rank_*`` shards, write the pkl next
    to them (parent of the first dir) unless ``pkl_path`` is set.
    """
    import lmdb as lmdb_lib

    if not lmdb_dirs:
        raise ValueError("lmdb_dirs must not be empty")

    if pkl_path is None:
        if len(lmdb_dirs) == 1:
            pkl_path = os.path.join(lmdb_dirs[0], "token_to_idx.pkl")
        else:
            pkl_path = os.path.join(
                os.path.dirname(os.path.abspath(lmdb_dirs[0])),
                "token_to_idx.pkl",
            )

    token_to_idx: Dict[str, int] = {}
    global_offset = 0

    logger.info(
        "Building perception token->index from %d LMDB(s) ...",
        len(lmdb_dirs),
    )

    for lmdb_path in lmdb_dirs:
        if not os.path.isdir(lmdb_path):
            raise FileNotFoundError(f"LMDB dir not found: {lmdb_path}")

        env = lmdb_lib.open(lmdb_path, readonly=True, lock=False)
        n_local = 0
        with env.begin() as txn:
            meta_raw = txn.get(b"__meta__")
            tokens: Optional[List[str]] = None
            if meta_raw is not None:
                try:
                    meta = msgpack.unpackb(meta_raw, raw=False)
                    scene_list = meta.get("scene_info_list") or []
                    extend_list = meta.get("extend_scenes_list") or []
                    tokens = [
                        frame["token"]
                        for frame in scene_list
                        if isinstance(frame, dict) and "token" in frame
                    ]
                    # extend scenes are rare; keys are tokens if present
                    for item in extend_list:
                        if isinstance(item, str):
                            tokens.append(item)
                        elif isinstance(item, dict) and "token" in item:
                            tokens.append(item["token"])
                except Exception as e:
                    logger.warning(
                        "Failed to parse __meta__ in %s: %s; "
                        "falling back to sample scan",
                        lmdb_path,
                        e,
                    )
                    tokens = None

            if tokens is not None:
                for local_idx, token in enumerate(tokens):
                    token_to_idx[token] = global_offset + local_idx
                n_local = len(tokens)
            else:
                local_keys: List[int] = []
                for key, _ in txn.cursor():
                    try:
                        local_keys.append(int(key))
                    except (TypeError, ValueError):
                        continue
                local_keys.sort()
                for local_idx in local_keys:
                    raw = txn.get(str(local_idx).encode("ascii"))
                    if raw is None:
                        continue
                    try:
                        packed = msgpack.unpackb(raw, raw=False)
                    except Exception:
                        try:
                            packed = pickle.loads(raw)
                        except Exception:
                            logger.warning(
                                "Failed to decode %s key=%s",
                                lmdb_path,
                                local_idx,
                            )
                            continue
                    if not isinstance(packed, dict):
                        continue
                    token = packed.get("token")
                    if token is None:
                        continue
                    token_to_idx[token] = global_offset + local_idx
                n_local = len(local_keys)

        env.close()
        logger.info(
            "  %s: %d samples (global [%d, %d))",
            lmdb_path,
            n_local,
            global_offset,
            global_offset + n_local,
        )
        global_offset += n_local

    out_dir = os.path.dirname(os.path.abspath(pkl_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(pkl_path, "wb") as f:
        pickle.dump(token_to_idx, f)

    logger.info(
        "Saved token->index mapping (%d entries) to %s",
        len(token_to_idx),
        pkl_path,
    )
    return token_to_idx
