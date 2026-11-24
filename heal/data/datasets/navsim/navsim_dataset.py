# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from __future__ import annotations
import io
import logging
import os
import pickle
import sys
from collections import OrderedDict
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import msgpack
import msgpack_numpy
import numpy as np
import torch

try:
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
except ImportError:
    compose = None
    initialize_config_dir = None
    instantiate = None

_NUPLAN_IMPORT_ERROR: Optional[BaseException] = None
_NAVSIM_IMPORT_ERROR: Optional[BaseException] = None
_HAS_NUPLAN = False
_HAS_NAVSIM = False

TrajectorySampling = None  # type: ignore[misc,assignment]
navsim = None  # type: ignore[assignment]
SceneLoader = None  # type: ignore[assignment]
Annotations = None  # type: ignore[assignment]
Camera = None  # type: ignore[assignment]
Cameras = None  # type: ignore[assignment]
EgoStatus = None  # type: ignore[assignment]
Frame = None  # type: ignore[assignment]
Lidar = None  # type: ignore[assignment]
Scene = None  # type: ignore[assignment]
SceneFilter = Any
SceneMetadata = None  # type: ignore[assignment]
SensorConfig = Any
AgentInput = Any
NavSimCommonScene = Any

try:
    from nuplan.planning.simulation.trajectory.trajectory_sampling import (
        TrajectorySampling as _TrajectorySampling,
    )

    TrajectorySampling = _TrajectorySampling
    _HAS_NUPLAN = True
except ImportError as exc:
    _NUPLAN_IMPORT_ERROR = exc

try:
    import navsim as _navsim_mod
    from navsim.agents.abstract_agent import AgentInput as _AgentInput
    from navsim.common.dataclasses import Annotations as _Annotations
    from navsim.common.dataclasses import Camera as _Camera
    from navsim.common.dataclasses import Cameras as _Cameras
    from navsim.common.dataclasses import EgoStatus as _EgoStatus
    from navsim.common.dataclasses import Frame as _Frame
    from navsim.common.dataclasses import Lidar as _Lidar
    from navsim.common.dataclasses import Scene as _NavSimCommonScene
    from navsim.common.dataclasses import Scene as _Scene
    from navsim.common.dataclasses import SceneFilter as _SceneFilter
    from navsim.common.dataclasses import SceneMetadata as _SceneMetadata
    from navsim.common.dataclasses import SensorConfig as _SensorConfig
    from navsim.common.dataloader import SceneLoader as _SceneLoader

    navsim = _navsim_mod
    SceneLoader = _SceneLoader
    Annotations = _Annotations
    Camera = _Camera
    Cameras = _Cameras
    EgoStatus = _EgoStatus
    Frame = _Frame
    Lidar = _Lidar
    Scene = _Scene
    SceneFilter = _SceneFilter
    SceneMetadata = _SceneMetadata
    SensorConfig = _SensorConfig
    AgentInput = _AgentInput
    NavSimCommonScene = _NavSimCommonScene
    _HAS_NAVSIM = True
except ImportError as exc:
    _NAVSIM_IMPORT_ERROR = exc

from hat_engine.data.datasets.data_packer import Packer
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.pack_type import PackTypeMapper, get_packtype_from_path
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from heal.data.datasets.navsim.navsim_metric_cache_dataset import (
    PACKABLE_FORMAT,
    NavsimMetricCacheLoader,
    NavsimMetricCachePacker,
    detections_tracks_from_list,
    metric_cache_from_packable,
    metric_cache_to_packable,
)

msgpack_numpy.patch()

logger = logging.getLogger(__name__)
_navsim_pkg_version = "0"
if _HAS_NAVSIM:
    try:
        _navsim_pkg_version = version("navsim")
    except Exception as exc:
        logger.warning(
            "Could not read installed navsim version via importlib.metadata (%s); "
            "assuming major version 0 (v1-style behaviors).",
            exc,
        )
_navsim_major_version = _navsim_pkg_version.split(".")[0]
NAVSIM_VERSION = (
    "v2"
    if _navsim_major_version.isdigit() and int(_navsim_major_version) >= 2
    else "v1"
)


def _require_nuplan() -> None:
    """Raise at runtime if nuplan is not available (module import stays lightweight)."""
    if _HAS_NUPLAN and TrajectorySampling is not None:
        return
    msg = (
        "heal.data.datasets.navsim.navsim_dataset: this operation requires `nuplan` "
        "(nuPlan devkit). Install it or add it to PYTHONPATH."
    )
    raise RuntimeError(msg) from _NUPLAN_IMPORT_ERROR


def _require_hydra() -> None:
    """Raise at runtime if hydra is not available."""
    if compose is not None and initialize_config_dir is not None:
        return
    raise RuntimeError(
        "heal.data.datasets.navsim.navsim_dataset: "
        "this operation requires `hydra`. "
        "Install hydra-core in this environment."
    )


def _require_navsim() -> None:
    """Raise at runtime if navsim is not available."""
    if _HAS_NAVSIM and navsim is not None:
        return
    msg = (
        "heal.data.datasets.navsim.navsim_dataset: this operation requires `navsim`. "
        "Install the NavSim package in this environment."
    )
    raise RuntimeError(msg) from _NAVSIM_IMPORT_ERROR


def _require_navsim_and_nuplan() -> None:
    _require_navsim()
    _require_nuplan()


def _sync_nuplan_maps_root(map_root: str) -> None:
    """Sync NUPLAN_MAPS_ROOT to env and navsim dataclasses snapshots (read at import)."""
    os.environ["NUPLAN_MAPS_ROOT"] = map_root
    for mod_name in (
        "navsim_v2.common.dataclasses",
        "navsim.common.dataclasses",
    ):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "NUPLAN_MAPS_ROOT"):
            mod.NUPLAN_MAPS_ROOT = map_root


def _rotation_translation_to_4x4(rotation, translation) -> np.ndarray:
    """Build a 4x4 pose from quaternion (wxyz) / 3x3 rotation + translation."""
    mat = np.eye(4, dtype=np.float64)
    rot = np.asarray(rotation, dtype=np.float64)
    trans = np.asarray(translation, dtype=np.float64).reshape(-1)[:3]
    if rot.shape == (3, 3):
        mat[:3, :3] = rot
    else:
        from pyquaternion import Quaternion

        q = rot.reshape(-1)
        mat[:3, :3] = Quaternion(*q).rotation_matrix
    mat[:3, 3] = trans
    return mat


def _attach_frame_calib(frame: Any, frame_data: Dict[str, Any]) -> Any:
    """Fill pose fields missing on official navsim ``Frame``.

    Packed OpenScene logs store ``ego2global_*`` / ``lidar2ego_*``; HEAL
    perception expects 4x4 ``lidar2global`` / ``ego2global`` on the frame.
    NavSim lidar frame coincides with ego, so missing lidar2ego is identity.
    """
    ego_t = np.asarray(frame_data["ego2global_translation"], dtype=np.float64)
    ego_r = np.asarray(frame_data["ego2global_rotation"], dtype=np.float64)
    if frame_data.get("lidar2ego_translation") is not None:
        lidar_t = np.asarray(
            frame_data["lidar2ego_translation"], dtype=np.float64
        )
    else:
        lidar_t = np.zeros(3, dtype=np.float64)
    if frame_data.get("lidar2ego_rotation") is not None:
        lidar_r = np.asarray(
            frame_data["lidar2ego_rotation"], dtype=np.float64
        )
    else:
        lidar_r = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    if frame_data.get("ego2global") is not None:
        ego2global = np.asarray(frame_data["ego2global"], dtype=np.float64)
    else:
        ego2global = _rotation_translation_to_4x4(ego_r, ego_t)
    if frame_data.get("lidar2ego") is not None:
        lidar2ego = np.asarray(frame_data["lidar2ego"], dtype=np.float64)
    else:
        lidar2ego = _rotation_translation_to_4x4(lidar_r, lidar_t)
    if frame_data.get("lidar2global") is not None:
        lidar2global = np.asarray(frame_data["lidar2global"], dtype=np.float64)
    else:
        lidar2global = ego2global @ lidar2ego

    frame.ego2global_translation = ego_t.astype(np.float32)
    frame.ego2global_rotation = ego_r.astype(np.float32)
    frame.lidar2ego_translation = lidar_t.astype(np.float32)
    frame.lidar2ego_rotation = np.asarray(lidar_r, dtype=np.float32)
    frame.ego2global = ego2global.astype(np.float32)
    frame.lidar2ego = lidar2ego.astype(np.float32)
    frame.lidar2global = lidar2global.astype(np.float32)
    if getattr(frame, "scene_token", None) is None:
        frame.scene_token = frame_data.get("scene_token")
    return frame


class PackedSceneLoader:
    _ALL_CAM_NAMES = (
        "cam_f0",
        "cam_l0",
        "cam_l1",
        "cam_l2",
        "cam_r0",
        "cam_r1",
        "cam_r2",
        "cam_b0",
    )

    def __init__(
        self,
        pack_file,
        meta: Dict[str, Any],
        scene_filter: SceneFilter,
        sensor_config: Optional[SensorConfig] = None,
    ):
        """Initialize the packed scene loader.

        Args:
            pack_file: Opened pack reader exposing ``read(index)``.
            meta: Decoded pack metadata with scene info and token lists.
            scene_filter: Scene filter defining history and future frame counts.
            sensor_config: Sensor selection configuration per frame iteration.
        """
        _require_navsim()
        if sensor_config is None:
            sensor_config = SensorConfig.build_no_sensors()
        self.pack_file = pack_file
        self.meta = meta
        self.scene_filter = scene_filter
        self._scene_filter = scene_filter
        self.sensor_config = sensor_config

        self.scene_info_dict = meta["scene_info_dict"]
        self.token_list = meta["token_list"]

        self.token_index: Dict[str, int] = {
            token: i for i, token in enumerate(self.token_list)
        }

        self.synth_scene_meta_dict: Dict[str, Any] = {}
        self.synth_scene_info_dict: Dict[str, Any] = {}
        self.synth_token_list: List[str] = []
        self.synth_token_index: Dict[str, int] = {}

        if "synth_scene_info_dict" in meta:
            self.synth_scene_meta_dict = meta["synth_scene_meta_dict"]
            self.synth_scene_info_dict = meta["synth_scene_info_dict"]
            self.synth_token_list = list(meta["synth_token_list"])
            base = len(self.token_list)
            self.synth_token_index = {
                token: base + j
                for j, token in enumerate(self.synth_token_list)
            }

        self._scene_items = list(self.scene_info_dict.items())
        self._synth_keys_sorted = sorted(self.synth_scene_info_dict.keys())
        self._tokens = [
            k for k, _ in self._scene_items
        ] + self._synth_keys_sorted
        # Align with `NavsimDataset` which indexes by `len(scene_loader.tokens)` (scene windows).
        self.dataset_length = len(self._tokens)
        logger.info("%s scenes loaded in total.", len(self._tokens))
        logger.info(
            "%s original scene windows loaded.", len(self.scene_info_dict)
        )
        logger.info(
            "%s synthetic scene windows loaded.",
            len(self.synth_scene_info_dict),
        )

    @property
    def tokens(self) -> List[str]:
        """Scene window tokens, original first then synthetic sorted keys."""
        return self._tokens

    def __len__(self) -> int:
        """Return the number of scene windows."""
        return self.dataset_length

    def _decode_image(self, img_bytes) -> Optional[np.ndarray]:
        """Decode packed image bytes into an RGB image.

        Args:
            img_bytes: Encoded image bytes from the pack entry.

        Returns:
            Decoded RGB image array or None when bytes are empty.
        """
        if img_bytes is None or len(img_bytes) == 0:
            return None
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")

    def _load_single_frame_data(self, global_idx: int) -> Dict:
        """Read and unpack one original frame by global pack index.

        Args:
            global_idx: Global pack index of the frame entry.

        Returns:
            Unpacked frame data dictionary.
        """
        # key = self.samples[global_idx]
        raw_data = self.pack_file.read(global_idx)
        if raw_data is None:
            raise ValueError(f"Could not read index {global_idx}")
        return msgpack.unpackb(
            raw_data, object_hook=msgpack_numpy.decode, raw=False
        )

    def _load_synth_single_frame_data(self, synth_pack_idx: int) -> Dict:
        """Read one synthetic frame from the same pack (indices after raw `token_list`)."""
        raw_data = self.pack_file.read(synth_pack_idx)
        if raw_data is None:
            raise ValueError(
                f"Could not read synthetic pack index {synth_pack_idx}"
            )
        return msgpack.unpackb(
            raw_data, object_hook=msgpack_numpy.decode, raw=False
        )

    def _build_scene_dict_list_from_chunk_tokens(
        self, chunk_tokens: List[str]
    ) -> List[Dict[str, Any]]:
        """Load and collect frame entries for one scene window token chunk.

        Args:
            chunk_tokens: Ordered frame tokens of the scene window.

        Returns:
            List of per-frame dicts with pack index and frame metadata.
        """
        scene_dict_list: List[Dict[str, Any]] = []
        append = scene_dict_list.append
        for token in chunk_tokens:
            gidx = self.token_index[token]
            frame_data = self._load_single_frame_data(gidx)
            append(
                {
                    "global_idx": gidx,
                    "token": frame_data.get("token", token),
                    "log_name": frame_data.get("log_name"),
                    "scene_token": frame_data.get("scene_token"),
                    "map_location": frame_data.get("map_location"),
                    "frame_data": frame_data,
                }
            )
        return scene_dict_list

    def _build_camera_object(
        self, camera_dict: Dict, sensor_names: List[str]
    ) -> Cameras:
        """Build the per-frame ``Cameras`` object for the selected sensors.

        Args:
            camera_dict: Frame data dict holding camera params and image bytes.
            sensor_names: Enabled camera names for the current frame iteration.

        Returns:
            ``Cameras`` with decoded images and calibrations per enabled camera.
        """
        data_dict = {}
        sensor_name_set = set(sensor_names)
        raw_cams = (
            camera_dict.get("cams", {})
            or {}
            or camera_dict.get("camera_dict", {})
        )
        cams_lower_map = {str(k).lower(): v for k, v in raw_cams.items()}
        raw_camera_data = camera_dict.get("camera_data", {}) or {}
        camera_data_lower_map = {
            str(k).lower(): v for k, v in raw_camera_data.items()
        }
        for cam_name in self._ALL_CAM_NAMES:
            if cam_name in sensor_name_set:
                # Keep only case-insensitive matching (no camera-name alias mapping).
                cam_params = raw_cams.get(
                    cam_name, cams_lower_map.get(cam_name.lower(), {})
                )
                img_bytes = raw_camera_data.get(
                    cam_name, camera_data_lower_map.get(cam_name.lower())
                )

                decoded_img = self._decode_image(img_bytes)

                data_dict[cam_name] = Camera(
                    image=decoded_img,
                    sensor2lidar_rotation=cam_params.get(
                        "sensor2lidar_rotation"
                    ),
                    sensor2lidar_translation=cam_params.get(
                        "sensor2lidar_translation"
                    ),
                    intrinsics=cam_params.get("cam_intrinsic"),
                    distortion=cam_params.get("distortion"),
                )
            else:
                data_dict[cam_name] = Camera()  # Empty

        return Cameras(**data_dict)

    def build_single_frame(self, frame_data, frame_idx):
        """Build one navsim ``Frame`` from packed frame data.

        Args:
            frame_data: Unpacked frame data dictionary.
            frame_idx: Frame iteration index used for sensor selection.

        Returns:
            Assembled ``Frame`` with calibration fields attached.
        """
        if "ego_status" in frame_data and frame_data["ego_status"] is not None:
            global_ego_status = EgoStatus(**frame_data["ego_status"])
        else:
            global_ego_status = Scene._build_ego_status(frame_data)

        if (
            "annotations" in frame_data
            and frame_data["annotations"] is not None
        ):
            annotations = Annotations(**frame_data["annotations"])
        else:
            annotations = Scene._build_annotations(frame_data)

        sensor_names = self.sensor_config.get_sensors_at_iteration(frame_idx)

        cameras = self._build_camera_object(frame_data, sensor_names)

        lidar = Lidar(
            lidar_pc=None,
        )
        frame = Frame(
            token=frame_data["token"],
            timestamp=frame_data["timestamp"],
            roadblock_ids=frame_data.get("roadblock_ids", []),
            traffic_lights=frame_data.get("traffic_lights", []),
            annotations=annotations,
            ego_status=global_ego_status,
            lidar=lidar,
            cameras=cameras,
        )
        return _attach_frame_calib(frame, frame_data)

    def load_scene_from_scene_dict_list(
        self, chunk_tokens: List[str]
    ) -> Scene:
        """Assemble an original-loaded ``Scene`` from one token chunk.

        Args:
            chunk_tokens: Ordered frame tokens covering history and future.

        Returns:
            Constructed ``Scene`` with metadata, map API and frames.
        """
        num_history_frames = self.scene_filter.num_history_frames
        num_future_frames = self.scene_filter.num_future_frames

        scene_dict_list = self._build_scene_dict_list_from_chunk_tokens(
            chunk_tokens
        )
        current = scene_dict_list[num_history_frames - 1]

        scene_metadata = SceneMetadata(
            log_name=current["log_name"],
            scene_token=current["scene_token"],
            map_name=current["map_location"],
            initial_token=current["token"],
            num_history_frames=num_history_frames,
            num_future_frames=num_future_frames,
            # corresponding_original_scene=current.get(
            #     "corresponding_original_scene", None   ),
            # corresponding_original_initial_token=current.get(
            #     "corresponding_original_initial_token", None),
        )
        map_api = Scene._build_map_api(scene_metadata.map_name)

        frames_list: List[Frame] = [
            self.build_single_frame(entry["frame_data"], frame_idx)
            for frame_idx, entry in enumerate(scene_dict_list)
        ]
        return Scene(
            scene_metadata=scene_metadata,
            map_api=map_api,
            frames=frames_list,
        )

    def load_scene_from_synthetic_scene_name(
        self, scene_name: str, extend_data: Dict[str, Any]
    ) -> Scene:
        """Assemble a synthetic ``Scene`` from its packed metadata and extension data.

        Args:
            scene_name: Synthetic scene key into ``synth_scene_meta_dict``.
            extend_data: Decoded synthetic extension payloads (traffic lights, tracks).

        Returns:
            Constructed synthetic ``Scene`` with extended annotations attached.
        """
        synth_meta = self.synth_scene_meta_dict[scene_name]
        scene_metadata = SceneMetadata(**synth_meta)
        map_api = Scene._build_map_api(scene_metadata.map_name)
        frames_tokens = self.synth_scene_info_dict[scene_name]
        frames_list: List[Frame] = []
        for frame_idx, frame_token in enumerate(frames_tokens):
            frame_global_idx = self.synth_token_index[frame_token]
            frame_data = self._load_synth_single_frame_data(frame_global_idx)
            frames_list.append(self.build_single_frame(frame_data, frame_idx))
        extended_traffic_light_data = (
            self._deserialize_extended_traffic_light_data(
                extend_data.get("extended_traffic_light_data", None)
            )
        )
        extended_detections_tracks = (
            self._deserialize_extended_detections_tracks(
                extend_data.get("extended_detections_tracks", None)
            )
        )
        return Scene(
            scene_metadata=scene_metadata,
            map_api=map_api,
            frames=frames_list,
            extended_traffic_light_data=extended_traffic_light_data,
            extended_detections_tracks=extended_detections_tracks,
        )

    def _deserialize_extended_traffic_light_data(self, value: Any) -> Any:
        """Best-effort decode for packed synthetic `extended_traffic_light_data`."""
        if value is None or not isinstance(value, list):
            return value

        try:
            from nuplan.common.maps.maps_datatypes import (
                TrafficLightStatusData,
                TrafficLightStatusType,
            )
        except Exception:
            return value
        from types import SimpleNamespace

        try:
            # navsim expects each iteration entry to expose `.traffic_lights`.
            from navsim.common.dataclasses import (
                TrafficLightStatuses as _TrafficLightStatuses,
            )
        except Exception:
            _TrafficLightStatuses = None

        def _decode_status(raw_status: Any) -> Any:
            if raw_status is None:
                return raw_status
            if isinstance(raw_status, TrafficLightStatusType):
                return raw_status
            if isinstance(raw_status, str):
                try:
                    return TrafficLightStatusType[raw_status]
                except Exception:
                    try:
                        return TrafficLightStatusType(raw_status)
                    except Exception:
                        return raw_status
            try:
                return TrafficLightStatusType(raw_status)
            except Exception:
                return raw_status

        decoded_all: List[Any] = []
        for frame_item in value:
            if isinstance(frame_item, dict):
                if frame_item.get("__type__") == "TrafficLightStatuses":
                    frame_lights = frame_item.get("traffic_lights", [])
                else:
                    frame_lights = frame_item.get("traffic_lights", frame_item)
            else:
                frame_lights = frame_item

            if not isinstance(frame_lights, list):
                decoded_all.append(frame_item)
                continue

            decoded_lights: List[Any] = []
            for light in frame_lights:
                if not isinstance(light, dict):
                    decoded_lights.append(light)
                    continue
                try:
                    decoded_lights.append(
                        TrafficLightStatusData(
                            lane_connector_id=light["lane_connector_id"],
                            status=_decode_status(light.get("status")),
                            timestamp=light["timestamp"],
                        )
                    )
                except Exception:
                    decoded_lights.append(light)
            if _TrafficLightStatuses is not None:
                try:
                    decoded_all.append(
                        _TrafficLightStatuses(traffic_lights=decoded_lights)
                    )
                    continue
                except Exception:
                    pass
            decoded_all.append(SimpleNamespace(traffic_lights=decoded_lights))

        return decoded_all

    def _deserialize_extended_detections_tracks(self, value: Any) -> Any:
        """Best-effort decode for packed synthetic `extended_detections_tracks`."""
        if value is None:
            return None
        if not isinstance(value, list):
            return value

        decoded: List[Any] = []
        for item in value:
            # Already a DetectionsTracks-like object.
            if hasattr(item, "tracked_objects"):
                decoded.append(item)
                continue

            # Preferred packed payload: list[tracked_object_dict].
            if isinstance(item, list):
                try:
                    decoded.append(detections_tracks_from_list(item))
                    continue
                except Exception:
                    logger.warning(
                        "Failed to deserialize detections track list element."
                    )
                    continue

            # Alternative payload: {"tracked_objects": list[tracked_object_dict]}.
            if isinstance(item, dict):
                tracked_list = item.get("tracked_objects")
                if isinstance(tracked_list, list):
                    try:
                        decoded.append(
                            detections_tracks_from_list(tracked_list)
                        )
                        continue
                    except Exception:
                        logger.warning(
                            "Failed to deserialize detections track dict element."
                        )
                        continue
                decoded.append(item)
                continue

            # Legacy stringified entries are not recoverable into tracked objects.
            logger.warning(
                "Unsupported extended_detections_tracks element type: %s",
                type(item).__name__,
            )

        return decoded

    def get_scene_from_idx(self, idx: int) -> Scene:
        """Resolve scene index ``idx`` to a full ``Scene``.

        ``idx`` matches ``len(tokens)`` / ``_tokens``: original windows in
        ``scene_info_dict`` insertion order, then synthetic windows in sorted
        key order (same as ``_tokens``).
        """
        n_orig = len(self.scene_info_dict)
        if idx < n_orig:
            scene_key, chunk_tokens = self._scene_items[idx]
            return scene_key, self.load_scene_from_scene_dict_list(
                chunk_tokens
            )
        else:
            j = idx - n_orig
            if j < 0 or j >= len(self._synth_keys_sorted):
                raise IndexError(f"Index out of range: {idx}")
            scene_key = self._synth_keys_sorted[j]
            extend_data = self.get_synthetic_extend_data(j)
            return scene_key, self.load_scene_from_synthetic_scene_name(
                scene_key, extend_data
            )

    def get_synthetic_extend_data(self, idx: int) -> Dict[str, Any]:
        """Load extension payload of a synthetic scene by synthetic index.

        Args:
            idx: Synthetic scene index relative to ``_synth_keys_sorted``.

        Returns:
            Decoded extension data dict, empty when no entry is packed.
        """
        extend_data_idx = (
            len(self.token_list) + len(self.synth_token_list) + idx
        )
        pack = self.pack_file
        raw_data = pack.read(extend_data_idx)
        if raw_data is None:
            logger.warning("No synthetic extend data found for index=%s", idx)
            return {}
        return msgpack.unpackb(
            raw_data, object_hook=msgpack_numpy.decode, raw=False
        )


def load_cfg(
    config_dir=None,
    config_name="navtest",
):  # 注意去掉 .yaml 后缀
    """Compose the navsim hydra config for a split.

    Args:
        config_dir: Directory holding the split yaml; defaults to navsim's split dir.
        config_name: Config file stem without ``.yaml`` suffix.

    Returns:
        Composed hydra config.
    """
    _require_navsim()
    _require_hydra()
    file_root = os.path.dirname(__file__)
    pkg_root = os.path.dirname(navsim.__file__)
    if config_dir is None:
        config_dir = os.path.join(
            pkg_root, "planning/script/config/common/train_test_split/"
        )
        if not os.path.exists(config_dir):
            assert (
                False
            ), f"Please provide split dir, default {config_dir} not exist!"

    config_dir = os.path.abspath(os.path.expanduser(config_dir))
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name=config_name)
    return cfg


def load_scene_filter(filter_config: str, config_name: Optional[str] = None):
    """
    Load scene_filter from yaml path or hydra config tuple.

    Supported forms:
    - load_scene_filter("/abs/path/to/navhard_two_stage.yaml")
    - load_scene_filter("/abs/path/to/config_dir", "navhard_two_stage")
    - load_scene_filter("navhard_two_stage")  # from default navsim config dir
    """
    config_dir = None
    cfg_name = config_name

    if config_name is not None:
        config_dir = filter_config
    else:
        path_obj = Path(filter_config)
        if path_obj.suffix == ".yaml" or path_obj.suffix == ".yml":
            config_dir = str(path_obj.parent)
            cfg_name = path_obj.stem
        else:
            cfg_name = filter_config
    print(config_dir, cfg_name)
    cfg = load_cfg(config_dir=config_dir, config_name=cfg_name or "navtest")
    scene_filter_cfg = cfg.get("scene_filter", cfg)
    return instantiate(scene_filter_cfg)


@OBJECT_REGISTRY.register
class NavsimDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str = "trainval",
        pack_type: str = "lmdb",
        pack_kwargs: Optional[dict] = None,
        scene_filter: Optional[SceneFilter] = None,
        split_cfg_root: str = None,
        sensor_config: Optional[SensorConfig] = None,
        return_scene: bool = True,
        token_sets=None,
        max_num=None,
        transforms: Optional[List] = None,
        time_horizon=4,
        interval_length=0.5,
        map_root=None,
        compute_features: bool = False,
        compute_target: bool = False,
    ):
        self.compute_features = compute_features
        self.compute_target = compute_target

        if map_root is None:
            map_root = os.environ.get("NUPLAN_MAPS_ROOT")
        if map_root is None:
            raise ValueError(
                "Must provide map_root or set NUPLAN_MAPS_ROOT env variable!"
            )
        self.map_root = map_root
        _sync_nuplan_maps_root(self.map_root)
        _require_navsim_and_nuplan()
        self.data_path = data_path
        self.max_num = max_num
        self.split = split
        if "navhard" in self.split.lower() and NAVSIM_VERSION == "v1":
            raise RuntimeError(
                f"split={self.split!r} requires navsim v2+, "
                f"but current navsim version is {_navsim_pkg_version!r}."
            )

        self.transforms = transforms

        if scene_filter is None:
            self.scene_filter = load_scene_filter(split_cfg_root, f"{split}")
        else:
            self.scene_filter = scene_filter

        if sensor_config is None:
            sensor_config = SensorConfig.build_no_sensors()
        if isinstance(sensor_config, dict):
            empty_sensor = asdict(SensorConfig.build_no_sensors())
            empty_sensor.update(sensor_config)
            sensor_config = empty_sensor
            sensor_config = SensorConfig(**sensor_config)
        # Keep for feature construction.
        self.sensor_config = sensor_config

        self.kwargs = {} if pack_kwargs is None else pack_kwargs
        try:
            self.pack_type = get_packtype_from_path(data_path)
        except NotImplementedError:
            assert pack_type is not None
            self.pack_type = PackTypeMapper[pack_type.lower()]
        self.pack_file = self.pack_type(
            self.data_path, writable=False, **self.kwargs
        )
        self.pack_file.open()
        self.meta = self._decode_meta(self.pack_file)

        self.scene_loader = PackedSceneLoader(
            pack_file=self.pack_file,
            meta=self.meta,
            scene_filter=self.scene_filter,
            sensor_config=sensor_config,
        )
        self.return_scene = return_scene

        self.trajectory_sampling = TrajectorySampling(
            time_horizon=time_horizon,
            interval_length=interval_length,
        )

    def _apply_img_transforms(self, imgs, camera_names) -> dict:
        data = {
            "imgs": imgs,
            "process_type": "image",
            "camera_names": camera_names,
        }
        data = self.transforms(data)
        return data

    def _compute_vla_features(
        self, agent_input: AgentInput
    ) -> Dict[str, torch.Tensor]:
        ego_statuses = agent_input.ego_statuses
        cameras = agent_input.cameras
        history_trajectory = torch.tensor(
            [
                [
                    float(e.ego_pose[0]),
                    float(e.ego_pose[1]),
                    float(e.ego_pose[2]),
                ]
                for e in ego_statuses[:4]
            ],
            dtype=torch.float32,
        )
        high_command_one_hot = torch.tensor(
            ego_statuses[-1].driving_command, dtype=torch.float32
        )
        status_feature = torch.cat(
            [
                high_command_one_hot.clone(),
                torch.tensor(
                    ego_statuses[-1].ego_velocity, dtype=torch.float32
                ),
                torch.tensor(
                    ego_statuses[-1].ego_acceleration, dtype=torch.float32
                ),
            ],
            dim=-1,
        )

        num_patches_dict = None
        cam_names = ["cam_f0"]

        intrinsics_list: List[torch.Tensor] = []
        rotation_list: List[torch.Tensor] = []
        translation_list: List[torch.Tensor] = []
        if self.sensor_config is not None:
            # `navsim_v2.common.dataclasses.SensorConfig` is a dataclass, not a dict.
            # Convert it to a plain dict and pick camera fields only.
            sensor_dict = asdict(self.sensor_config)
            cam_names = [
                k
                for k, v in sensor_dict.items()
                if k.startswith("cam_") and v is not False and v is not None
            ]

            all_imgs = []
            for cam_name in cam_names:
                frame_indices = sensor_dict[cam_name]
                # Handle boolean includes (rare in configs, but keep it robust).
                if isinstance(frame_indices, bool):
                    frame_indices = [-1] if frame_indices else []
                if not frame_indices:
                    continue

                frame_list = []
                for idx in frame_indices:
                    cam = getattr(agent_input.cameras[idx], cam_name)
                    frame_list.append(cam.image)
                all_imgs.append(frame_list)

                ref_cam = getattr(
                    agent_input.cameras[frame_indices[0]], cam_name
                )
                if ref_cam.intrinsics is not None:
                    intrinsics_list.append(
                        torch.tensor(ref_cam.intrinsics, dtype=torch.float32)
                    )
                    rotation_list.append(
                        torch.tensor(
                            ref_cam.sensor2lidar_rotation, dtype=torch.float32
                        )
                    )
                    translation_list.append(
                        torch.tensor(
                            ref_cam.sensor2lidar_translation,
                            dtype=torch.float32,
                        )
                    )

            if self.transforms is not None:
                transform_out = self._apply_img_transforms(all_imgs, cam_names)
                pixel_values = transform_out["imgs"]
                if "num_tiles_per_view" in transform_out:
                    num_patches_dict = dict(
                        zip(cam_names, transform_out["num_tiles_per_view"])
                    )
            else:
                pixel_values = all_imgs
        else:
            frame_indices = [-1]
            cam_image = cameras[frame_indices[0]].cam_f0.image
            if self.transforms is not None:
                transform_out = self._apply_img_transforms(
                    [[cam_image]], ["cam_f0"]
                )
                pixel_values = transform_out["imgs"]
                if "num_tiles_per_view" in transform_out:
                    num_patches_dict = dict(
                        zip(["cam_f0"], transform_out["num_tiles_per_view"])
                    )
            else:
                pixel_values = [cam_image]

        if (
            num_patches_dict is not None
            and isinstance(pixel_values, list)
            and len(pixel_values) > 0
            and isinstance(pixel_values[0], torch.Tensor)
        ):
            pixel_values = torch.stack(pixel_values)

        if (
            isinstance(pixel_values, list)
            and pixel_values
            and isinstance(pixel_values[0], list)
        ):
            pixel_values = [torch.stack(frames) for frames in pixel_values]

        result = {
            "history_trajectory": history_trajectory.cpu(),
            "high_command_one_hot": high_command_one_hot.cpu(),
            "status_feature": status_feature.cpu(),
            "pixel_values": pixel_values,
            "camera_names": cam_names,
        }
        if num_patches_dict is not None:
            result["num_patches_list"] = num_patches_dict
        if self.sensor_config is not None and intrinsics_list:
            result["cam_intrinsics"] = torch.stack(intrinsics_list)
            result["cam_rotation"] = torch.stack(rotation_list)
            result["cam_translation"] = torch.stack(translation_list)
        return result

    def _compute_trajectory_targets(
        self,
        scene: NavSimCommonScene,
    ) -> Dict[str, torch.Tensor]:
        future_trajectory = scene.get_future_trajectory(
            num_trajectory_frames=self.trajectory_sampling.num_poses
        )
        return {"trajectory": torch.tensor(future_trajectory.poses)}

    def _decode_meta(self, pack_file) -> Dict[str, Any]:
        meta = self._decode(pack_file, "__meta__")
        return meta

    def _decode(self, pack_file, key):
        sample = pack_file.read(key)
        if sample is not None:
            unpacked_data = msgpack.unpackb(
                sample, object_hook=msgpack_numpy.decode, raw=False
            )

            return unpacked_data
        return None

    def get_meta(self) -> Dict[str, Any]:
        """Get meta information of dataset."""
        return self.meta

    @property
    def tokens(self) -> List[str]:
        """Expose scene tokens for downstream packers/loaders."""
        return self.scene_loader.tokens

    def __len__(self):
        if self.max_num is None:
            return len(self.scene_loader.tokens)
        else:
            return min(len(self.scene_loader.tokens), self.max_num)

    def build_feature_target(self, scene: Scene) -> Dict[str, Any]:
        data = {}
        if self.compute_features:
            agent_input = scene.get_agent_input()
            feature = self._compute_vla_features(agent_input)
            data["features"] = feature
        if self.compute_target:
            data["targets"] = self._compute_trajectory_targets(scene)
        return data

    def __getitem__(self, idx: int):
        token, scene = self.scene_loader.get_scene_from_idx(idx)
        data = {}
        data["token"] = token
        data.update(self.build_feature_target(scene))
        if self.return_scene:
            data["scene"] = scene

        return data


class NavsimCacheStateDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        pack_type: str = "lmdb",
        pack_kwargs: Optional[dict] = None,
    ):
        self.data_path = data_path
        self.pack_kwargs = pack_kwargs or {}
        self.pack_type = PackTypeMapper[pack_type.lower()]

        meta_file = os.path.join(data_path, "cache_metas.pkl")
        if not os.path.exists(meta_file):
            raise FileNotFoundError(f"Cache meta file not found: {meta_file}")
        with open(meta_file, "rb") as f:
            self.cache_meta = pickle.load(f)
        self.token_list = list(self.cache_meta.keys())

        ranks = [r for r, _ in self.cache_meta.values()]
        lmdb_length = max(ranks) + 1 if ranks else 0

        self.pack_files = []
        for rank in range(lmdb_length):
            lmdb_file = os.path.join(data_path, f"cache_state_rank{rank}_lmdb")
            pack_file = self.pack_type(
                lmdb_file, writable=False, **self.pack_kwargs
            )
            pack_file.open()
            self.pack_files.append(pack_file)

    def _get_packer(self, rank):
        if self.pack_files[rank] is None:
            lmdb_file = os.path.join(
                self.data_path, f"cache_state_rank{rank}_lmdb"
            )
            pack = self.pack_type(
                lmdb_file,
                writable=False,
                **self.pack_kwargs,
            )
            pack.open()
            self.pack_files[rank] = pack
        return self.pack_files[rank]

    def load_cache_states(self, token: str) -> Dict[str, Any]:
        if token not in self.cache_meta:
            raise KeyError(f"Token {token} not found in cache meta.")
        rank, idx = self.cache_meta[token]
        packer = self._get_packer(rank)
        cache_bytes = packer.read(idx)
        data = pickle.loads(cache_bytes)
        return data

    def __len__(self):
        return len(self.token_list)

    def __getitem__(self, idx: int):
        token = self.token_list[idx]
        cache_state = self.load_cache_states(token)
        assert cache_state["token"] == token, "Token mismatch!"
        return cache_state
