# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from __future__ import annotations
import logging
import pickle
from collections import OrderedDict
from collections.abc import ItemsView, KeysView, ValuesView
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import msgpack
import msgpack_numpy
import numpy as np
import torch
from hat_engine.data.datasets.data_packer import Packer
from tqdm import tqdm

from heal.data.datasets.navsim.navsim_dataset import (
    _require_navsim,
    load_scene_filter,
)

msgpack_numpy.patch()

logger = logging.getLogger(__name__)


class NavsimParser(object):
    """Unified NavSim raw parser for navtrain/navtest/navhard."""

    def list_to_token_dict(
        self, frame_info: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        return {f["token"]: f for f in frame_info}

    def _get_meta_of_frame(
        self, idx: int, frame: Dict[str, Any]
    ) -> Dict[str, Any]:
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

    def __init__(
        self,
        data_root: str,
        split: str = "navtrain",
        sensor_names: Optional[List[str]] = None,
        scene_filter: Optional[Union[str, Any]] = None,
        synth_data_path: str = None,
    ):
        _require_navsim()
        split_key = split.lower()
        if split_key not in {"navtrain", "navtest", "navhard_two_stage"}:
            raise ValueError(
                f"Unsupported split: {split}. Expected one of "
                "'navtrain', 'navtest', 'navhard_two_stage'."
            )

        self.root = Path(data_root)
        self.split = split
        self.sensor_names = sensor_names

        self._split_key = split_key
        self._scene_filter = scene_filter

        self._scene_info_dict = OrderedDict()
        self._token_frame_dict = OrderedDict()
        self._token_frame_values: List[Dict[str, Any]] = []

        self._synth_extend_data_list = []
        self._synth_scene_meta_dict = OrderedDict()
        self._synth_scene_info_dict = OrderedDict()
        self._synth_token_frame_dict = OrderedDict()
        self._synth_token_frame_values: List[Dict[str, Any]] = []

        storage_subdir = {
            "navtest": "test",
            "navtrain": "trainval",
            "navhard_two_stage": "test",
        }.get(self._split_key, self._split_key)
        self.logs_path = self.root / "navsim_logs" / storage_subdir
        if not self.logs_path.exists():
            raise FileNotFoundError(f"Logs path {self.logs_path} not found.")

        self.blobs_path = self.root / "sensor_blobs" / storage_subdir
        if not self.blobs_path.exists():
            raise FileNotFoundError(f"Blobs path {self.blobs_path} not found.")

        self.include_synthetic_scenes = (
            self._scene_filter.include_synthetic_scenes
        )
        if self.include_synthetic_scenes is True:
            if synth_data_path is None:
                raise ValueError(
                    "synth_data_path is required when include_synthetic_scenes is True"
                )
            synth_path = Path(synth_data_path)
            self.synth_meta_path = synth_path / "synthetic_scene_pickles"
            if not self.synth_meta_path.exists():
                raise FileNotFoundError(
                    f"Synthetic meta path {self.synth_meta_path} not found."
                )
            self.synth_blobs_path = synth_path / "sensor_blobs"
            if not self.synth_blobs_path.exists():
                raise FileNotFoundError(
                    f"Synthetic blobs path {self.synth_blobs_path} not found."
                )
        self._parse()

    def _parse(self) -> None:
        filtered_scene_frame_tokens: Dict[str, List[str]] = {}
        token_to_frame_all: Dict[str, Dict[str, Any]] = {}
        stage1_final_frame_tokens: List[str] = []
        stop_loading = False
        log_files = sorted(
            log_file
            for log_file in self.logs_path.iterdir()
            if log_file.suffix == ".pkl"
        )
        log_files = self._filter_log_files(log_files)

        for log_file in tqdm(log_files, desc="Parsing Raw Meta"):
            with open(log_file, "rb") as f:
                scene_frames = pickle.load(f)
            if not isinstance(scene_frames, list):
                raise AssertionError(
                    f"v1 scene should be list, but got: {type(scene_frames)}"
                )

            log_key = log_file.stem

            frame_info_list: List[Dict[str, Any]] = []
            for frame_idx, frame in enumerate(scene_frames):
                token = frame.get("token")
                if token is not None:
                    token_to_frame_all.setdefault(token, frame)
                frame_info_list.append(
                    self._get_meta_of_frame(frame_idx, frame)
                )
            stop_loading = self._filter_scenes_for_log(
                frame_info_list=frame_info_list,
                filtered_scenes=filtered_scene_frame_tokens,
                final_frame_tokens=stage1_final_frame_tokens,
            )

            if stop_loading:
                break
        self._scene_info_dict = filtered_scene_frame_tokens
        self._token_frame_dict = OrderedDict(
            (token, token_to_frame_all[token])
            for token in chain.from_iterable(
                filtered_scene_frame_tokens.values()
            )
            if token in token_to_frame_all
        )
        if self.include_synthetic_scenes:
            self._parse_synthetic_scenes(stage1_final_frame_tokens)
        self._token_frame_values = list(self._token_frame_dict.values())
        self._synth_token_frame_values = list(
            self._synth_token_frame_dict.values()
        )

    def _filter_log_files(self, log_files: List[Path]) -> List[Path]:
        allowed_log_names = set(
            getattr(self._scene_filter, "log_names", None) or []
        )
        if not allowed_log_names:
            return log_files
        return [
            log_file
            for log_file in log_files
            if log_file.stem in allowed_log_names
        ]

    def _parse_synthetic_scenes(self, stage1_final_frame_tokens) -> None:
        synth_files = sorted(self.synth_meta_path.iterdir())
        stage1_final_set = set(stage1_final_frame_tokens)
        for synth_file in tqdm(synth_files, desc="Parsing Synthetic Meta"):
            raw_stem = synth_file.stem
            with open(synth_file, "rb") as f:
                scene_frames = pickle.load(f)
            if not (
                isinstance(scene_frames, dict) and "frames" in scene_frames
            ):
                raise AssertionError(
                    f"Synthetic scene should be dict with `frames`, but got: {type(scene_frames)}"
                )
            scene_meta = scene_frames.get("scene_metadata", {})
            if not self.filter_synthetic_scenes(
                self._scene_filter,
                stage1_final_set,
                scene_meta,
            ):
                continue
            initial_token = scene_meta.get("initial_token")
            self._synth_scene_meta_dict[initial_token] = scene_meta
            for frame in scene_frames["frames"]:
                self._synth_token_frame_dict[frame["token"]] = frame
            self._synth_scene_info_dict[initial_token] = [
                frame["token"] for frame in scene_frames["frames"]
            ]
            extend_data = {
                "extended_traffic_light_data": scene_frames.get(
                    "extended_traffic_light_data"
                ),
                "extended_detections_tracks": scene_frames.get(
                    "extended_detections_tracks"
                ),
            }
            self._synth_extend_data_list.append(extend_data)

    def filter_synthetic_scenes(
        self,
        scene_filter: Any,
        stage1_scenes_final_frames_tokens: set[str],
        scene_meta: Dict[str, Any],
    ) -> bool:
        """Decide whether one synthetic scene should be kept."""
        filter_logs = scene_filter.log_names is not None
        filter_tokens = scene_filter.synthetic_scene_tokens is not None

        initial_token = scene_meta.get("initial_token")
        curr_log_name = scene_meta.get("log_name")
        corresponding_original_scene = scene_meta.get(
            "corresponding_original_scene"
        )

        if (
            filter_tokens
            and initial_token not in scene_filter.synthetic_scene_tokens
        ):
            return False
        if filter_logs and curr_log_name not in scene_filter.log_names:
            return False
        if (not filter_tokens) and (
            corresponding_original_scene
            not in stage1_scenes_final_frames_tokens
        ):
            return False
        return True

    def _split_list(
        self, input_list: List[Any], num_frames: int, frame_interval: int
    ) -> List[List[Any]]:
        return [
            input_list[i : i + num_frames]
            for i in range(0, len(input_list), frame_interval)
        ]

    def _filter_scenes_for_log(
        self,
        frame_info_list: List[Dict[str, Any]],
        filtered_scenes: Dict[str, List[str]],
        final_frame_tokens: List[str],
    ) -> bool:
        scene_filter = self._scene_filter
        if scene_filter is None:
            return False

        filter_tokens = scene_filter.tokens is not None
        tokens_allowlist = set(scene_filter.tokens or [])

        target_len = (
            scene_filter.num_history_frames + scene_filter.num_future_frames
        )
        interval = scene_filter.frame_interval or 1
        chunked_lists = self._split_list(frame_info_list, target_len, interval)
        for frame_chunk in chunked_lists:
            if len(frame_chunk) < target_len:
                continue
            curr_frame_info = frame_chunk[scene_filter.num_history_frames - 1]
            token = curr_frame_info["token"]
            if scene_filter.has_route and not curr_frame_info["has_route"]:
                continue
            if filter_tokens and token not in tokens_allowlist:
                continue

            filtered_scenes[token] = [frame["token"] for frame in frame_chunk]

            final_frame_tokens.append(frame_chunk[target_len - 1]["token"])
            if (
                scene_filter.max_scenes is not None
                and len(filtered_scenes) >= scene_filter.max_scenes
            ):
                return True
        return False

    def _load_scene_frames_for_log(self, log_key: str) -> List[Any]:
        path = self._scene_pickle_paths[log_key]
        with open(path, "rb") as f:
            loaded = pickle.load(f)
        if isinstance(loaded, list):
            if self._log_is_synthetic.get(log_key, False):
                raise AssertionError(
                    "v2 list-format scenes are not expected for synthetic logs"
                )
            return loaded
        if isinstance(loaded, dict) and "frames" in loaded:
            return loaded["frames"]
        raise AssertionError(
            f"Unsupported scene_frames type for lazy load: {type(loaded)}"
        )

    def __len__(self):
        return (
            len(self._token_frame_dict)
            + len(self._synth_token_frame_dict)
            + len(self._synth_extend_data_list)
        )

    def _get_meta(self):
        if len(self._synth_scene_info_dict) > 0:
            return {
                "scene_info_dict": self._scene_info_dict,
                "token_list": list(self._token_frame_dict.keys()),
                "synth_scene_meta_dict": self._synth_scene_meta_dict,
                "synth_token_list": list(self._synth_token_frame_dict.keys()),
                "synth_scene_info_dict": self._synth_scene_info_dict,
            }
        else:
            return {
                "scene_info_dict": self._scene_info_dict,
                "token_list": list(self._token_frame_dict.keys()),
            }

    def __getitem__(self, idx: int):
        raw_count = len(self._token_frame_dict)
        synth_count = len(self._synth_token_frame_dict)
        extend_count = len(self._synth_extend_data_list)
        total_count = raw_count + synth_count + extend_count

        if idx < 0 or idx >= total_count:
            raise IndexError(f"Index out of range: {idx}")

        if idx < raw_count:
            single_frame_data = self._token_frame_values[idx]
            blobs_path = self.blobs_path
        elif idx < raw_count + synth_count:
            single_frame_data = self._synth_token_frame_values[idx - raw_count]
            blobs_path = self.synth_blobs_path
        else:
            return self._synth_extend_data_list[idx - raw_count - synth_count]

        sensor_names = self.sensor_names or []
        if "cams" in single_frame_data:
            camera_dict = single_frame_data["cams"]
        elif "camera_dict" in single_frame_data:
            camera_dict = single_frame_data["camera_dict"]
        else:
            camera_dict = {}

        cameras = {}
        for camera_name in camera_dict.keys():
            camera_identifier = camera_name.lower()
            camera_dict[camera_name]["data_path"] = str(
                camera_dict[camera_name]["data_path"]
            )
            if camera_identifier in sensor_names:
                image_path = blobs_path / camera_dict[camera_name]["data_path"]
                try:
                    with open(image_path, "rb") as f:
                        cameras[camera_name] = f.read()
                except FileNotFoundError:
                    logger.warning(
                        "Camera file not found, skip loading: %s (camera=%s)",
                        image_path,
                        camera_name,
                    )
                    continue
        single_frame_data["camera_data"] = cameras

        if "lidar_path" in single_frame_data and isinstance(
            single_frame_data["lidar_path"], Path
        ):
            single_frame_data["lidar_path"] = str(
                single_frame_data["lidar_path"]
            )

        return single_frame_data


# ===================Packer ===================
class NavsimPacker(Packer):
    """
    Dataset packer for Navsim v1/v2.
    """

    def __init__(
        self,
        src_data_dir: str,
        target_data_dir: str,
        split_name: str,
        split_cfg_root: str,
        synth_data_path: str = None,
        num_workers: int = 20,
        pack_type: str = "raw",
        max_samples: int = None,
        **kwargs,
    ):
        _require_navsim()
        sensor_names = [
            "cam_f0",
            "cam_l0",
            "cam_l1",
            "cam_l2",
            "cam_r0",
            "cam_r1",
            "cam_r2",
            "cam_b0",
        ]

        if not split_cfg_root:
            raise ValueError("split_cfg_root is required for NavsimPacker.")

        split_cfg_path = Path(split_cfg_root).expanduser()
        if not split_cfg_path.exists():
            raise FileNotFoundError(
                f"split_cfg_root not found: {split_cfg_root}"
            )
        if split_cfg_path.suffix in {".yaml", ".yml"}:
            scene_filter = load_scene_filter(str(split_cfg_path))
        elif split_cfg_path.is_dir():
            scene_filter = load_scene_filter(str(split_cfg_path), split_name)
        else:
            raise ValueError(
                "split_cfg_root must be a yaml file path or a config directory, "
                f"got: {split_cfg_root}"
            )
        self.dataset = NavsimParser(
            src_data_dir,
            split=split_name,
            sensor_names=sensor_names,
            scene_filter=scene_filter,
            synth_data_path=synth_data_path,
        )
        self.packer_meta = self.dataset._get_meta()
        num_samples = len(self.dataset)
        if max_samples is not None:
            num_samples = min(num_samples, max_samples)
        kwargs.pop("navsim_version", None)
        kwargs["map_size"] = 1024**4 * 10
        super(NavsimPacker, self).__init__(
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
                packed_meta = self._to_msgpackable(self.packer_meta)
                self.pack_file.write("__meta__", msgpack.packb(packed_meta))
                self.pack_file.write(
                    "__len__", "{}".format(num).encode("ascii")
                )
            except ValueError:
                logger.warning(
                    f"{self.target_pack_type} is not supported "
                    + "for write length mainly due to the string key.",
                )

    def _tracked_object_to_dict(self, obj: Any) -> Dict[str, Any]:
        """Serialize one nuPlan tracked object to plain dict."""
        tracked_type = getattr(obj, "tracked_object_type", None)
        tracked_type_value = getattr(tracked_type, "value", tracked_type)
        metadata = getattr(obj, "metadata", None)
        center = getattr(obj, "center", None)
        box = getattr(obj, "box", None)

        result: Dict[str, Any] = {
            "type": tracked_type_value,
            "x": getattr(center, "x", None),
            "y": getattr(center, "y", None),
            "heading": getattr(center, "heading", None),
            "length": getattr(box, "length", None),
            "width": getattr(box, "width", None),
            "height": getattr(box, "height", None),
            "token": getattr(metadata, "token", None),
            "track_token": getattr(metadata, "track_token", None),
            "track_id": getattr(metadata, "track_id", None),
            "timestamp_us": getattr(metadata, "timestamp_us", None),
        }
        velocity = getattr(obj, "velocity", None)
        if velocity is not None:
            result["vx"] = getattr(velocity, "x", None)
            result["vy"] = getattr(velocity, "y", None)
        return result

    def _to_msgpackable(self, obj: Any) -> Any:
        """Recursively convert sample fields to msgpack-compatible objects."""
        if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
            return obj
        if isinstance(obj, dict):
            return {str(k): self._to_msgpackable(v) for k, v in obj.items()}
        if isinstance(obj, KeysView):
            return [self._to_msgpackable(v) for v in obj]
        if isinstance(obj, ValuesView):
            return [self._to_msgpackable(v) for v in obj]
        if isinstance(obj, ItemsView):
            return [
                [self._to_msgpackable(k), self._to_msgpackable(v)]
                for k, v in obj
            ]
        if isinstance(obj, (list, tuple, set)):
            return [self._to_msgpackable(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy()
        if isinstance(obj, np.ndarray):
            return obj

        # nuPlan DetectionsTracks-like object.
        if hasattr(obj, "tracked_objects"):
            tracked_objects = getattr(obj, "tracked_objects", None)
            if tracked_objects is not None:
                return [
                    self._tracked_object_to_dict(item)
                    for item in tracked_objects
                ]

        # nuPlan Agent / StaticObject-like object.
        if hasattr(obj, "tracked_object_type") and hasattr(obj, "metadata"):
            return self._tracked_object_to_dict(obj)

        # nuPlan traffic-light objects expose `.serialize()` returning plain dict.
        serialize_fn = getattr(obj, "serialize", None)
        if callable(serialize_fn):
            return self._to_msgpackable(serialize_fn())

        # e.g. TrafficLightStatuses(traffic_lights=[...])
        if hasattr(obj, "traffic_lights"):
            return {
                "__type__": "TrafficLightStatuses",
                "traffic_lights": [
                    self._to_msgpackable(item)
                    for item in getattr(obj, "traffic_lights", [])
                ],
            }

        # Generic enum fallback.
        if hasattr(obj, "value") and not callable(getattr(obj, "value")):
            return self._to_msgpackable(getattr(obj, "value"))

        # Last-resort stable representation.
        return str(obj)

    def _pack_data(self, sample):
        sample = self._to_msgpackable(sample)
        data = msgpack.packb(sample, default=msgpack_numpy.encode)
        return data

    def pack_data(self, idx):
        sample = self.dataset[idx]
        try:
            data = self._pack_data(sample)
        except:
            keys = list(sample.keys())
            type_list = [type(sample[k]) for k in sample.keys()]

            data = msgpack.packb(sample, default=msgpack_numpy.encode)
        return data
