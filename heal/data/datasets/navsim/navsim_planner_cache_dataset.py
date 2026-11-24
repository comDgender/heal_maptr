# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import os
import pickle
from typing import Dict, Optional

import msgpack
import msgpack_numpy
import numpy as np
import torch
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.pack_type.utils import get_packtype_from_path
from torch.utils.data import Dataset

from heal.data.datasets.navsim.navsim_feature_builder import (
    _NAVSIM_AGENT_TYPES,
    _local_to_current_ego,
)
from heal.utils.perception_to_planner import (
    perception_dyn_to_planner_agents,
    perception_maptr_to_planner_lanes,
    perception_static_to_planner_static,
)

msgpack_numpy.patch()


def _to_tensor(arr, dtype=torch.float32):
    if dtype == torch.long:
        numpy_type = np.int64
    elif dtype == torch.bool:
        numpy_type = np.bool_
    else:
        numpy_type = np.float32

    if isinstance(arr, torch.Tensor):
        if arr.dtype != dtype:
            if dtype == torch.long:
                arr = arr.long()
            elif dtype == torch.bool:
                arr = arr.bool()
            else:
                arr = arr.float()
        return arr

    if isinstance(arr, np.ndarray):
        if arr.dtype != numpy_type:
            arr = arr.astype(numpy_type, copy=False)
        return torch.from_numpy(arr)

    return torch.tensor(arr, dtype=dtype)


def _build_perception_dict_from_cache(cache_entry):
    """Build a perception-result dict from a cache entry.

    The cache entry has keys like:
      - dynamic_objects_det: {"bboxes": ndarray, "scores": ndarray,
        "labels": ndarray}
      - static_objects_det: same format
      - maptr: {"pts": ndarray, "scores": ndarray, "labels": ndarray,
        "boxes": ndarray (optional)}

    Returns dict with torch tensors matching what the conversion functions
    expect.
    """
    dyn = cache_entry.get("dynamic_objects_det", {})
    static = cache_entry.get("static_objects_det", {})
    maptr = cache_entry.get("maptr", {})

    dyn_dict = {
        "bboxes": _to_tensor(dyn["bboxes"], dtype=torch.float32),
        "scores": _to_tensor(dyn["scores"], dtype=torch.float32),
        "labels": _to_tensor(dyn["labels"], dtype=torch.long),
    }

    static_dict = {
        "bboxes": _to_tensor(static["bboxes"], dtype=torch.float32),
        "scores": _to_tensor(static["scores"], dtype=torch.float32),
        "labels": _to_tensor(static["labels"], dtype=torch.long),
    }

    maptr_dict = {
        "pts_3d": _to_tensor(maptr["pts_3d"], dtype=torch.float32),
        "scores_3d": _to_tensor(maptr["scores_3d"], dtype=torch.float32),
        "labels_3d": _to_tensor(maptr["labels_3d"], dtype=torch.long),
    }

    return dyn_dict, static_dict, maptr_dict


def _rotate_velocity(vx, vy, from_heading, to_heading):
    """Rotate velocity vector between two heading frames."""
    dh = from_heading - to_heading
    cos_dh, sin_dh = np.cos(dh), np.sin(dh)
    return cos_dh * vx - sin_dh * vy, sin_dh * vx + cos_dh * vy


def _extract_ego_future_gt(scene, num_future_poses=10):
    """Extract ego future GT trajectory at native 2 Hz from Scene frames.

    No upsampling — directly uses Navsim's 2 Hz future frames.

    Args:
        scene: Navsim Scene object with future frames.
        num_future_poses: Number of future frames (10 for 5s at 2Hz).

    Returns:
        np.ndarray of shape (num_future_poses, 3) — [x, y, heading].
    """
    num_history = scene.scene_metadata.num_history_frames
    num_future = scene.scene_metadata.num_future_frames

    current_frame = scene.frames[num_history - 1]
    current_ego_pose = np.array(
        current_frame.ego_status.ego_pose, dtype=np.float64
    )
    cx, cy, ch = current_ego_pose
    cos_ch, sin_ch = np.cos(ch), np.sin(ch)

    poses = []
    for f_idx in range(
        num_history, num_history + min(num_future, num_future_poses)
    ):
        frame = scene.frames[f_idx]
        gx, gy, gh = frame.ego_status.ego_pose
        dx, dy = gx - cx, gy - cy
        lx = cos_ch * dx + sin_ch * dy
        ly = -sin_ch * dx + cos_ch * dy
        lh = gh - ch
        poses.append([lx, ly, lh])

    result = np.zeros((num_future_poses, 3), dtype=np.float32)
    for i, p in enumerate(poses):
        result[i] = p
    return result


def _extract_neighbors_future_gt(
    scene,
    predicted_neighbor_num=10,
    num_future_poses=10,
):
    """Extract neighbor agents' future GT trajectories at native 2 Hz.

    No upsampling — directly uses Navsim's 2 Hz future frames.

    Args:
        scene: Navsim Scene object.
        predicted_neighbor_num: Number of neighbor agents to keep (10).
        num_future_poses: Number of future time steps (10 for 5s at 2Hz).

    Returns:
        np.ndarray of shape (predicted_neighbor_num, num_future_poses, 3).
    """
    num_history = scene.scene_metadata.num_history_frames
    num_future = scene.scene_metadata.num_future_frames

    current_frame = scene.frames[num_history - 1]
    current_ego_pose = np.array(
        current_frame.ego_status.ego_pose, dtype=np.float64
    )

    # Build track_token -> agent info mapping from current frame
    current_ann = current_frame.annotations
    current_agents = {}
    for a_idx in range(len(current_ann.track_tokens)):
        token = current_ann.track_tokens[a_idx]
        name = current_ann.names[a_idx]
        if name not in _NAVSIM_AGENT_TYPES:
            continue
        box = current_ann.boxes[a_idx]
        current_agents[token] = {
            "type": _NAVSIM_AGENT_TYPES[name],
            "x": float(box[0]),
            "y": float(box[1]),
            "heading": float(box[6]),
        }

    # Collect future positions per track_token across future frames
    agent_futures: Dict[str, dict] = {}
    for f_idx in range(num_history, num_history + num_future):
        frame = scene.frames[f_idx]
        frame_ego_pose = frame.ego_status.ego_pose
        ann = frame.annotations

        for a_idx in range(len(ann.track_tokens)):
            token = ann.track_tokens[a_idx]
            if token not in current_agents:
                continue

            box = ann.boxes[a_idx]
            lx, ly, lh = _local_to_current_ego(
                float(box[0]),
                float(box[1]),
                float(box[6]),
                frame_ego_pose,
                current_ego_pose,
            )

            if token not in agent_futures:
                agent_futures[token] = {}
            future_step = f_idx - num_history
            agent_futures[token][future_step] = np.array(
                [lx, ly, lh], dtype=np.float32
            )

    # Select top agents by distance
    unique_tokens = list(current_agents.keys())
    if not unique_tokens:
        return np.zeros(
            (predicted_neighbor_num, num_future_poses, 3),
            dtype=np.float32,
        )

    dist = np.array(
        [
            np.linalg.norm([current_agents[t]["x"], current_agents[t]["y"]])
            for t in unique_tokens
        ]
    )

    agent_types_list = [current_agents[t]["type"] for t in unique_tokens]
    sorted_idx = np.argsort(dist)

    pb = [i for i in sorted_idx if agent_types_list[i] != [1, 0, 0]]
    veh = [i for i in sorted_idx if agent_types_list[i] == [1, 0, 0]]

    max_ped_bike = predicted_neighbor_num
    if len(pb) + len(veh) <= predicted_neighbor_num:
        selected = list(sorted_idx[:predicted_neighbor_num])
    else:
        sel_pb = pb[:max_ped_bike]
        selected = sel_pb + veh
        remain = predicted_neighbor_num - len(selected)
        if remain > 0:
            selected += pb[max_ped_bike : max_ped_bike + remain]
        selected = sorted(selected, key=lambda i: dist[i])[
            :predicted_neighbor_num
        ]

    result = np.zeros(
        (predicted_neighbor_num, num_future_poses, 3),
        dtype=np.float32,
    )

    for i, idx in enumerate(selected):
        token = unique_tokens[idx]
        if token not in agent_futures:
            continue

        for step, pose in agent_futures[token].items():
            if step < num_future_poses:
                result[i, step] = pose

    return result


@OBJECT_REGISTRY.register
class NavsimPlannerCacheDataset(Dataset):
    """Dataset for planner training using cached perception results.

    Reads GT Scene data from NavsimDataset and perception cache from pkl
    files. Converts cached perception results to planner input tensors,
    replacing the GT features that DiffusionPlannerFeatureBuilder would
    normally produce for agents, static objects, and lanes.

    All temporal data is at Navsim's native 2 Hz rate (no upsampling).

    Also extracts ego_future_gt and neighbors_future_gt from the Scene's
    future frames, which are required by the training pipeline
    (StatePerturbation and NoiseScheduleSampler transforms).

    Supports two data source modes:

    * **navsim_dataset mode** (default): Reads GT Scene data from
      ``NavsimDataset`` and optionally combines with perception cache.
      All feature extraction happens at ``__getitem__`` time.
    * **lmdb mode**: When ``lmdb_data_path`` is provided, reads
      pre-computed planning features directly from an LMDB database
      (produced by ``DiffusionPlannerNavsimPacker``). This avoids
      loading the NavsimDataset entirely and is much faster.

    Args:
        use_lmdb: Whether to use lmdb data.
        navsim_dataset: Config dict for NavsimDataset (built via registry).
            Must return Scene objects with route/map data.
            Ignored when ``lmdb_data_path`` is set.
        lmdb_data_path: Path to an LMDB database containing pre-packed
            planning features (produced by DiffusionPlannerNavsimPacker).
            When set, ``navsim_dataset`` is not used and all features are
            read directly from LMDB.
        use_perception_cache: Whether to use cached perception result.
            Only used in navsim_dataset mode.
        cache_dir: Path to the perception cache directory.
            Only used in navsim_dataset mode.
        agent_num: Max number of agents (must match planner config).
        model_time_len: Temporal length for agent features at 2Hz.
        static_objects_num: Max number of static objects.
        max_ped_bike: Max pedestrian/bicycle agents.
        lane_num: Max number of lane elements.
        lane_len: Number of points per lane element.
        route_num: Max route lane elements for GT map features.
        route_len: Points per route lane element for GT map features.
        predicted_neighbor_num: Number of neighbor agents for future GT.
        num_future_poses: Number of future time steps at 2Hz.
    """

    def __init__(
        self,
        use_lmdb: bool = True,
        lmdb_data_path: Optional[str] = None,
        navsim_dataset: Optional[Dataset] = None,
        use_perception_cache: bool = True,
        cache_dir: Optional[str] = None,
        dynamic_objects_score_threshold: Optional[float] = 0.2,
        static_objects_score_threshold: Optional[float] = 0.2,
        maptr_score_threshold: Optional[float] = 0.2,
        agent_num: int = 32,
        model_time_len: int = 4,
        static_objects_num: int = 5,
        max_ped_bike: int = 10,
        lane_num: int = 70,
        lane_len: int = 20,
        route_num: int = 25,
        route_len: int = 20,
        predicted_neighbor_num: int = 10,
        num_future_poses: int = 10,
        with_agent_past_trajectory: bool = True,
        with_agent_future_trajectory: bool = True,
        use_lane_boundary: bool = False,
    ):
        self.agent_num = agent_num
        self.model_time_len = model_time_len
        self.static_objects_num = static_objects_num
        self.max_ped_bike = max_ped_bike
        self.lane_num = lane_num
        self.lane_len = lane_len
        self.predicted_neighbor_num = predicted_neighbor_num
        self.num_future_poses = num_future_poses
        self.use_perception_cache = use_perception_cache
        self.dynamic_objects_score_threshold = dynamic_objects_score_threshold
        self.static_objects_score_threshold = static_objects_score_threshold
        self.maptr_score_threshold = maptr_score_threshold
        self.use_lmdb = use_lmdb
        self.with_agent_past_trajectory = with_agent_past_trajectory
        self.with_agent_future_trajectory = with_agent_future_trajectory
        self.use_lane_boundary = use_lane_boundary

        if self.use_lmdb:
            # LMDB mode: read pre-packed planning features directly
            self.lmdb_data_path = lmdb_data_path
            self._lmdb_pack_file = None
            self._lmdb_keys = None
            self.navsim_dataset = None
            if lmdb_data_path is not None:
                self._init_lmdb_reader(lmdb_data_path)
        else:
            self.navsim_dataset = navsim_dataset
        if self.use_perception_cache:
            # Load perception cache metadata (small dict, safe for fork)
            self.cache_dir = cache_dir
            self._load_cache_metas(cache_dir)

    def _load_cache_metas(self, cache_dir: str):
        """Load cache_metas.pkl and build sample_token -> file_info lookup.

        The metas dict maps sample_token to:
          - New format (4-element): (rank, chunk_id, idx, offset)
          - Old format (3-element): (rank, chunk_id, idx)
          - Old non-chunked format (2-element): (rank, idx)
        """
        meta_path = os.path.join(cache_dir, "cache_metas.pkl")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"cache_metas.pkl not found in {cache_dir}"
            )
        with open(meta_path, "rb") as f:
            self._cache_metas: Dict = pickle.load(f)

        # Validate format from first entry
        sample_val = next(iter(self._cache_metas.values()))
        self._cache_has_offset = len(sample_val) == 4
        self._cache_chunked = len(sample_val) in (3, 4)

        print(
            f"NavsimPlannerCacheDataset: loaded {len(self._cache_metas)} "
            f"cache metadata entries from {cache_dir} "
            f"(format: {'offset' if self._cache_has_offset else 'legacy'})"
        )

    def _get_split_path(self, rank, chunk_id=None):
        if chunk_id is not None:
            return os.path.join(
                self.cache_dir,
                f"split_{rank}_chunk_{chunk_id}.pkl",
            )
        return os.path.join(self.cache_dir, f"split_{rank}.pkl")

    def _init_lmdb_reader(self, lmdb_data_path: str):
        """Open the LMDB database and read key list.

        The LMDB is produced by ``DiffusionPlannerNavsimPacker``, which
        stores each sample as msgpack-serialized bytes keyed by integer
        index.  A special ``__len__`` key stores the total count.
        """
        pack_type_cls = get_packtype_from_path(lmdb_data_path)
        self._lmdb_pack_file = pack_type_cls(
            lmdb_data_path,
            writable=False,
        )
        self._lmdb_pack_file.open()
        self._lmdb_keys = self._lmdb_pack_file.get_keys()
        print(
            f"NavsimPlannerCacheDataset: LMDB mode, loaded "
            f"{len(self._lmdb_keys)} samples from {lmdb_data_path}"
        )

    def _decode_lmdb(self, raw):
        """Decode msgpack-serialized bytes from LMDB.

        LMDB stores data as binary bytes; both dict keys and string values
        need utf-8 decoding. This mirrors NuscenesDataset._decode.
        """

        def _decode_hook(obj):
            def _decode_bytes(obj):
                if isinstance(obj, bytes):
                    obj = obj.decode("utf-8")
                return obj

            new_obj = {}
            for k, v in obj.items():
                k = _decode_bytes(k)
                v = _decode_bytes(v)
                new_obj[k] = v
            return new_obj

        return msgpack.unpackb(raw, object_hook=_decode_hook, raw=True)

    def _getitem_from_lmdb(self, idx):
        """Read a single sample from LMDB and convert to planner tensors.

        The LMDB stores data in the format produced by
        ``DiffusionPlannerNavsimParser.__getitem__`` and serialized with
        ``msgpack.packb``.  Each entry contains numpy arrays for all
        planner features.

        Returns a dict matching the output format of the navsim_dataset
        branch (torch tensors).
        """
        key = self._lmdb_keys[idx]
        raw = self._lmdb_pack_file.read_idx(key)
        info = self._decode_lmdb(raw)

        # Convert numpy arrays to torch tensors, matching the output
        # format of the navsim_dataset branch.
        result = {
            "token": info["token"],
            "neighbor_agents_past": _to_tensor(
                info["neighbor_agents_past"], dtype=torch.float32
            ),
            "ego_current_state": _to_tensor(
                info["ego_current_state"], dtype=torch.float32
            ),
            # "ego_current_state": torch.tensor(
            #     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            #     dtype=torch.float32,
            # ),
            "static_objects": _to_tensor(
                info["static_objects"], dtype=torch.float32
            ),
            "lanes": _to_tensor(info["lanes"], dtype=torch.float32),
            "lanes_speed_limit": _to_tensor(
                info["lanes_speed_limit"], dtype=torch.float32
            ),
            "lanes_has_speed_limit": _to_tensor(
                info["lanes_has_speed_limit"], dtype=torch.bool
            ),
            "ego_future_gt": _to_tensor(
                info["ego_future_gt"], dtype=torch.float32
            ),
            "neighbors_future_gt": _to_tensor(
                info["neighbors_future_gt"], dtype=torch.float32
            ),
            "route_lanes": _to_tensor(
                info["route_lanes"], dtype=torch.float32
            ),
            "route_lanes_speed_limit": _to_tensor(
                info["route_lanes_speed_limit"], dtype=torch.float32
            ),
            "route_lanes_has_speed_limit": _to_tensor(
                info["route_lanes_has_speed_limit"], dtype=torch.bool
            ),
        }
        if self.use_lane_boundary:
            result["lanes"] = _to_tensor(
                info["lane_boundaries"]["lanes"], dtype=torch.float32
            )
            result["lanes_speed_limit"] = _to_tensor(
                info["lane_boundaries"]["lanes_speed_limit"],
                dtype=torch.float32,
            )
            result["lanes_has_speed_limit"] = _to_tensor(
                info["lane_boundaries"]["lanes_has_speed_limit"],
                dtype=torch.bool,
            )
        return result

    def _read_cache_entry(self, token: str):
        """Read a single cache entry from the split pkl file.

        Uses the offset in cache_metas for random access (new format),
        or falls back to loading the entire file and indexing (legacy).

        Each call opens and closes the file independently, which is safe for
        DataLoader multi-worker (fork) mode, since each worker gets its own
        file descriptor.
        """
        meta = self._cache_metas.get(token)
        if meta is None:
            return None

        if self._cache_has_offset:
            rank, chunk_id, idx, offset = meta
            split_file = self._get_split_path(rank, chunk_id)
            with open(split_file, "rb") as fh:
                fh.seek(offset)
                return pickle.load(fh)
        else:
            # Legacy format: load entire file and index
            if self._cache_chunked:
                rank, chunk_id, idx = meta
                split_file = self._get_split_path(rank, chunk_id)
            else:
                rank, idx = meta
                split_file = self._get_split_path(rank)

            with open(split_file, "rb") as fh:
                entries = pickle.load(fh)
                return entries[idx]

    def __len__(self):
        if self.use_lmdb:
            return len(self._lmdb_keys)
        return len(self.navsim_dataset)

    def __getitem__(self, idx):
        if self.use_lmdb:
            result = self._getitem_from_lmdb(idx)
            token = result["token"]
        else:
            navsim_data = self.navsim_dataset[idx]
            scene = navsim_data.get("scene", None)
            token = navsim_data["token"]

            # Fallback: use GT features from feature_builder at 2Hz
            gt_features = navsim_data.get("features", {})
            neighbors_future_gt = gt_features.get(
                "neighbors_future_gt",
                torch.zeros(
                    self.predicted_neighbor_num,
                    self.num_future_poses,
                    3,
                    dtype=torch.float32,
                ),
            )

            if (
                "trajectory" in navsim_data["targets"]
                and navsim_data["targets"]["trajectory"] != []
            ):
                ego_future_gt = navsim_data["targets"]["trajectory"]
            else:
                ego_future_gt = _extract_ego_future_gt(
                    scene,
                    num_future_poses=self.num_future_poses,
                )

            result = {
                "neighbor_agents_past": _to_tensor(
                    gt_features["neighbor_agents_past"], dtype=torch.float32
                ),
                "ego_current_state": _to_tensor(
                    gt_features["ego_current_state"], dtype=torch.float32
                ),
                "static_objects": _to_tensor(
                    gt_features["static_objects"], dtype=torch.float32
                ),
                "lanes": _to_tensor(gt_features["lanes"], dtype=torch.float32),
                "lanes_speed_limit": _to_tensor(
                    gt_features["lanes_speed_limit"], dtype=torch.float32
                ),
                "lanes_has_speed_limit": _to_tensor(
                    gt_features["lanes_has_speed_limit"], dtype=torch.bool
                ),
                "ego_future_gt": _to_tensor(
                    ego_future_gt, dtype=torch.float32
                ),
                "neighbors_future_gt": _to_tensor(
                    neighbors_future_gt, dtype=torch.float32
                ),
                "route_lanes": _to_tensor(
                    gt_features["route_lanes"], dtype=torch.float32
                ),
                "route_lanes_speed_limit": _to_tensor(
                    gt_features["route_lanes_speed_limit"], dtype=torch.float32
                ),
                "route_lanes_has_speed_limit": _to_tensor(
                    gt_features["route_lanes_has_speed_limit"],
                    dtype=torch.bool,
                ),
            }
        if self.use_perception_cache:
            cache_entry = self._read_cache_entry(token)
            if "perception_postprocess" in cache_entry:
                result["neighbor_agents_past"] = _to_tensor(
                    cache_entry["perception_postprocess"][
                        "neighbor_agents_past"
                    ],
                    dtype=torch.float32,
                )
                result["static_objects"] = _to_tensor(
                    cache_entry["perception_postprocess"]["static_objects"],
                    dtype=torch.float32,
                )
                result["lanes"] = _to_tensor(
                    cache_entry["perception_postprocess"]["lanes"],
                    dtype=torch.float32,
                )
                result["lanes_speed_limit"] = _to_tensor(
                    cache_entry["perception_postprocess"]["lanes_speed_limit"],
                    dtype=torch.float32,
                )
                result["lanes_has_speed_limit"] = _to_tensor(
                    cache_entry["perception_postprocess"][
                        "lanes_has_speed_limit"
                    ],
                    dtype=torch.bool,
                )
            else:
                (
                    dyn_dict,
                    static_dict,
                    maptr_dict,
                ) = _build_perception_dict_from_cache(cache_entry)
                if self.dynamic_objects_score_threshold is not None:
                    mask = (
                        dyn_dict["scores"]
                        >= self.dynamic_objects_score_threshold
                    )
                    dyn_dict["scores"] = dyn_dict["scores"][mask]
                    dyn_dict["bboxes"] = dyn_dict["bboxes"][mask]
                    dyn_dict["labels"] = dyn_dict["labels"][mask]
                if self.static_objects_score_threshold is not None:
                    mask = (
                        static_dict["scores"]
                        >= self.static_objects_score_threshold
                    )
                    static_dict["scores"] = static_dict["scores"][mask]
                    static_dict["bboxes"] = static_dict["bboxes"][mask]
                    static_dict["labels"] = static_dict["labels"][mask]
                if self.maptr_score_threshold is not None:
                    thresh_mask = (
                        maptr_dict["scores_3d"] > self.maptr_score_threshold
                    )
                    tmp_score = self.maptr_score_threshold
                    while thresh_mask.sum() == 0:
                        tmp_score *= 0.9
                        if tmp_score < 0.01:
                            thresh_mask = maptr_dict["scores_3d"] > -1
                            break
                        thresh_mask = maptr_dict["scores_3d"] >= tmp_score
                    maptr_dict["scores_3d"] = maptr_dict["scores_3d"][
                        thresh_mask
                    ]
                    maptr_dict["labels_3d"] = maptr_dict["labels_3d"][
                        thresh_mask
                    ]
                    maptr_dict["pts_3d"] = maptr_dict["pts_3d"][thresh_mask]

                # Convert dynamic objects -> neighbor_agents_past
                # perception_dyn_to_planner_agents only fills the last
                # timestep, so model_time_len controls the temporal dim.
                neighbor_agents_past = perception_dyn_to_planner_agents(
                    [dyn_dict],
                    agent_num=self.agent_num,
                    model_time_len=self.model_time_len,
                    max_ped_bike=self.max_ped_bike,
                )[
                    0
                ]  # (agent_num, model_time_len, 11)

                # Convert static objects -> static_objects
                static_objects = perception_static_to_planner_static(
                    [static_dict],
                    static_objects_num=self.static_objects_num,
                )[
                    0
                ]  # (static_objects_num, 10)

                # maptr -> lanes, lanes_speed_limit, lanes_has_speed_limit
                (
                    lanes,
                    lanes_sl,
                    lanes_hsl,
                ) = perception_maptr_to_planner_lanes(
                    [maptr_dict],
                    lane_num=self.lane_num,
                    lane_len=self.lane_len,
                )
                lanes = lanes[0]  # (lane_num, lane_len, 12)
                lanes_sl = lanes_sl[0]  # (lane_num, 1)
                lanes_hsl = lanes_hsl[0].float()  # (lane_num, 1)

                result["neighbor_agents_past"] = _to_tensor(
                    neighbor_agents_past, dtype=torch.float32
                )
                result["static_objects"] = _to_tensor(
                    static_objects, dtype=torch.float32
                )
                result["lanes"] = _to_tensor(lanes, dtype=torch.float32)
                result["lanes_speed_limit"] = _to_tensor(
                    lanes_sl, dtype=torch.float32
                )
                result["lanes_has_speed_limit"] = _to_tensor(
                    lanes_hsl, dtype=torch.bool
                )

        if not self.with_agent_future_trajectory:
            result["neighbors_future_gt"] = torch.zeros(
                self.predicted_neighbor_num,
                self.num_future_poses,
                3,
                dtype=torch.float32,
            )
        if not self.with_agent_past_trajectory:
            result["neighbor_agents_past"][:, :-1, :] = 0.0

        # Token for metric matching
        # result["token"] = token

        # Target trajectories (for training loss)
        # targets = navsim_data.get("targets", {})
        # if targets:
        #     result["targets"] = targets

        return result

    def __getstate__(self):
        """Support pickle for DataLoader multi-worker fork.

        Close LMDB file handle before pickling; it will be reopened in
        the child process via __setstate__.
        """
        state = self.__dict__.copy()
        if state.get("_lmdb_pack_file") is not None:
            state["_lmdb_pack_file"].close()
            state["_lmdb_pack_file"] = None
        return state

    def __setstate__(self, state):
        """Restore state after unpickling in DataLoader worker."""
        self.__dict__ = state
        if self.use_lmdb:
            pack_type_cls = get_packtype_from_path(self.lmdb_data_path)
            self._lmdb_pack_file = pack_type_cls(
                self.lmdb_data_path,
                writable=False,
            )
            self._lmdb_pack_file.open()
            self._lmdb_keys = self._lmdb_pack_file.get_keys()


@OBJECT_REGISTRY.register
class NavsimPerceptionFeatureCacheDataset(Dataset):
    """Dataset that merges cached perception BEV features with planning GT.

    This dataset is designed for the **second stage** of E2E training where
    the perception model is frozen and only the planner is trained.  Instead
    of running the full perception model online, we read pre-cached BEV
    feature maps (and optionally detection results) from the perception
    inference cache produced by ``PerceptionInferCacheProcess``.

    Two data sources are combined:

    1. **Planning GT** from an LMDB database (produced by
       ``DiffusionPlannerNavsimPacker``) — contains ``ego_current_state``,
       ``route_lanes``, ``ego_future_gt``, ``neighbors_future_gt``, etc.
    2. **Perception feature cache** from pkl files (produced by
       ``PerceptionInferCacheProcess``) — contains ``bev_embed``,
       ``dynamic_bev_embed``, ``maptr_bev_embed``, detection results, and
       optionally ``perception_postprocess``.

    The output dict is compatible with ``DiffusionPlannerEncoder`` /
    ``Diffusion_Planner`` and the training transforms
    (``StatePerturbation``, ``NoiseScheduleSampler``).

    Key design decisions
    --------------------
    * When the perception cache contains ``perception_postprocess`` (i.e.
      detection→planner conversion was done during caching), we use it to
      fill ``neighbor_agents_past``, ``static_objects``, ``lanes``,
      ``lanes_speed_limit``, ``lanes_has_speed_limit``, overriding the GT
      values from LMDB.
    * When ``perception_postprocess`` is not present, the GT values from
      LMDB are kept and only BEV features are injected from cache.
    * BEV features are loaded from cache and mapped to the keys expected by
      the encoder: ``base_bev``, ``dynamic_bev``, ``static_bev`` depending
      on which are available in the cache entry.
    * The LMDB provides a ``token_to_idx.pkl`` for O(1) token lookup; if
      it is missing we fall back to scanning the LMDB on init (same as
      ``NavsimPlannerCacheDataset``).

    Args:
        perception_feature_cache_dir: Path to the perception cache directory
            containing ``cache_metas.pkl`` and split pkl chunk files.
        palnner_lmdb_data_path: Path to the LMDB database \
            with planning GT features.
        with_agent_future_trajectory: Whether to keep agent future GT.
    """

    def __init__(
        self,
        perception_feature_cache_dir: Optional[str] = None,
        palnner_lmdb_data_path: Optional[str] = None,
        with_agent_future_trajectory: bool = False,
        predicted_neighbor_num: int = 10,
        num_future_poses: int = 8,
        need_bev_feature: bool = False,
        need_dynamic_feature: bool = False,
        need_static_feature: bool = False,
        need_occ_feature: bool = False,
    ):
        self.with_agent_future_trajectory = with_agent_future_trajectory
        self.predicted_neighbor_num = predicted_neighbor_num
        self.num_future_poses = num_future_poses
        self.need_bev_feature = need_bev_feature
        self.need_dynamic_feature = need_dynamic_feature
        self.need_static_feature = need_static_feature
        self.need_occ_feature = need_occ_feature

        # --- Initialise LMDB reader (planning GT) ---
        self.lmdb_data_path = palnner_lmdb_data_path
        self._lmdb_pack_file = None
        self._lmdb_keys = None
        if self.lmdb_data_path is not None:
            self._init_lmdb_reader(self.lmdb_data_path)
        else:
            raise ValueError(
                "lmdb_data_path is required for "
                "NavsimPerceptionFeatureCacheDataset"
            )

        # --- Initialise perception cache reader ---
        self.cache_dir = perception_feature_cache_dir
        self._cache_metas = None
        self._cache_has_offset = False
        self._cache_chunked = False
        self._load_cache_metas(perception_feature_cache_dir)

    # ------------------------------------------------------------------
    # LMDB helpers (same pattern as NavsimPlannerCacheDataset)
    # ------------------------------------------------------------------
    def _init_lmdb_reader(self, lmdb_data_path: str):
        """Open the LMDB database and read key list."""
        pack_type_cls = get_packtype_from_path(lmdb_data_path)
        self._lmdb_pack_file = pack_type_cls(
            lmdb_data_path,
            writable=False,
        )
        self._lmdb_pack_file.open()
        self._lmdb_keys = self._lmdb_pack_file.get_keys()
        print(
            f"NavsimPerceptionFeatureCacheDataset: LMDB mode, loaded "
            f"{len(self._lmdb_keys)} samples from {lmdb_data_path}"
        )

    def _decode_lmdb(self, raw):
        """Decode msgpack-serialized bytes from LMDB."""

        def _decode_hook(obj):
            def _decode_bytes(obj):
                if isinstance(obj, bytes):
                    obj = obj.decode("utf-8")
                return obj

            new_obj = {}
            for k, v in obj.items():
                k = _decode_bytes(k)
                v = _decode_bytes(v)
                new_obj[k] = v
            return new_obj

        return msgpack.unpackb(raw, object_hook=_decode_hook, raw=True)

    # ------------------------------------------------------------------
    # Perception cache helpers (same pattern as NavsimPlannerCacheDataset)
    # ------------------------------------------------------------------

    def _load_cache_metas(self, cache_dir: str):
        """Load cache_metas.pkl and build sample_token → file_info lookup."""
        meta_path = os.path.join(cache_dir, "cache_metas.pkl")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"cache_metas.pkl not found in {cache_dir}"
            )
        with open(meta_path, "rb") as f:
            self._cache_metas: Dict = pickle.load(f)

        # Validate format from first entry
        sample_val = next(iter(self._cache_metas.values()))
        self._cache_has_offset = len(sample_val) == 4
        self._cache_chunked = len(sample_val) in (3, 4)

        print(
            f"NavsimPerceptionFeatureCacheDataset: loaded "
            f"{len(self._cache_metas)} cache metadata entries from "
            f"{cache_dir} "
            f"(format: {'offset' if self._cache_has_offset else 'legacy'})"
        )

    def _get_split_path(self, rank, chunk_id=None):
        if chunk_id is not None:
            return os.path.join(
                self.cache_dir,
                f"split_{rank}_chunk_{chunk_id}.pkl",
            )
        return os.path.join(self.cache_dir, f"split_{rank}.pkl")

    def _read_cache_entry(self, token: str):
        """Read a single cache entry from the split pkl file."""
        meta = self._cache_metas.get(token)
        if meta is None:
            return None

        if self._cache_has_offset:
            rank, chunk_id, idx, offset = meta
            split_file = self._get_split_path(rank, chunk_id)
            with open(split_file, "rb") as fh:
                fh.seek(offset)
                return pickle.load(fh)
        else:
            # Legacy format: load entire file and index
            if self._cache_chunked:
                rank, chunk_id, idx = meta
                split_file = self._get_split_path(rank, chunk_id)
            else:
                rank, idx = meta
                split_file = self._get_split_path(rank)

            with open(split_file, "rb") as fh:
                entries = pickle.load(fh)
                return entries[idx]

    def __len__(self):
        return len(self._lmdb_keys)

    def __getitem__(self, idx):
        """Get a sample combining planning GT from LMDB + cache features.

        The planning GT is always read from LMDB.  Then, if
        ``use_perception_cache`` is True, the perception cache entry for
        this sample is loaded and:

        1. BEV feature maps (``bev_embed``, ``dynamic_bev_embed``,
           ``maptr_bev_embed``) are injected as ``base_bev``,
           ``dynamic_bev``, ``static_bev`` for the encoder.
        2. If ``perception_postprocess`` exists in the cache entry, the
           detection-derived features override the LMDB GT values for
           ``neighbor_agents_past``, ``static_objects``, ``lanes``,
           ``lanes_speed_limit``, ``lanes_has_speed_limit``.
        3. If ``perception_postprocess`` is absent but raw detection
           results are available, they are converted on-the-fly using the
           same conversion functions as
           ``NavsimPlannerCacheDataset.__getitem__``.
        """
        # --- 1. Read planning GT from LMDB ---
        key = self._lmdb_keys[idx]
        raw = self._lmdb_pack_file.read_idx(key)
        info = self._decode_lmdb(raw)

        result = {
            "token": info["token"],
            "ego_current_state": _to_tensor(
                info["ego_current_state"], dtype=torch.float32
            ),
            "ego_future_gt": _to_tensor(
                info["ego_future_gt"], dtype=torch.float32
            ),
            "neighbors_future_gt": _to_tensor(
                info["neighbors_future_gt"], dtype=torch.float32
            ),
            "route_lanes": _to_tensor(
                info["route_lanes"], dtype=torch.float32
            ),
            "route_lanes_speed_limit": _to_tensor(
                info["route_lanes_speed_limit"], dtype=torch.float32
            ),
            "route_lanes_has_speed_limit": _to_tensor(
                info["route_lanes_has_speed_limit"], dtype=torch.bool
            ),
        }
        # Neighbor/lane GT required by NoiseScheduleSampler when
        # use_neighbors_future=True. lanes/lanes_speed_limit/
        # lanes_has_speed_limit may be overridden by cache postprocess.
        for k in (
            "neighbor_agents_past",
            "static_objects",
            "lanes",
            "lanes_speed_limit",
        ):
            if k in info:
                result[k] = _to_tensor(info[k], dtype=torch.float32)
        if "lanes_has_speed_limit" in info:
            result["lanes_has_speed_limit"] = _to_tensor(
                info["lanes_has_speed_limit"], dtype=torch.bool
            )

        # --- 2. Inject perception cache features ---
        token = info["token"]
        cache_entry = self._read_cache_entry(token)

        if cache_entry is not None:
            self._inject_bev_features(result, cache_entry)

        # --- 3. Post-processing flags ---
        if not self.with_agent_future_trajectory:
            result["neighbors_future_gt"] = torch.zeros(
                self.predicted_neighbor_num,
                self.num_future_poses,
                3,
                dtype=torch.float32,
            )

        return result

    # ------------------------------------------------------------------
    # BEV feature injection
    # ------------------------------------------------------------------
    def _inject_bev_features(self, result: dict, cache_entry: dict):
        """Inject BEV feature maps from perception cache into result.

        The perception cache may contain one or more BEV feature maps:

        - ``bev_embed``: unified BEV feature → mapped to ``base_bev``
        - ``dynamic_bev_embed``: dynamic object BEV → ``dynamic_bev``
        - ``maptr_bev_embed``: MapTR lane BEV → ``static_bev``
        - ``occ_bev_embed``: occupancy BEV → ``static_bev`` (fallback)

        ``DiffusionPlannerEncoder`` expects ``base_bev`` for
        ``BEVSingleEncoder``, or ``base_bev + dynamic_bev + static_bev``
        for ``BEVFusionEncoder``.
        """
        # Unified BEV → base_bev (required by BEVSingleEncoder)
        bev_embed = cache_entry.get("bev_embed")
        if bev_embed is not None and self.need_bev_feature:
            result["base_bev"] = _to_tensor(bev_embed, dtype=torch.float32)

        # Separate BEV maps for BEVFusionEncoder
        dynamic_bev = cache_entry.get("dynamic_bev_embed")
        if dynamic_bev is not None and self.need_dynamic_feature:
            result["dynamic_bev"] = _to_tensor(
                dynamic_bev, dtype=torch.float32
            )

        maptr_bev = cache_entry.get("maptr_bev_embed")
        if maptr_bev is not None and self.need_static_feature:
            result["static_bev"] = _to_tensor(maptr_bev, dtype=torch.float32)

        occ_bev = cache_entry.get("occ_bev_embed")
        if occ_bev is not None and self.need_occ_feature:
            result["occ_bev"] = _to_tensor(occ_bev, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Pickle support for DataLoader multi-worker fork
    # ------------------------------------------------------------------

    def __getstate__(self):
        """Close LMDB handle before pickling; reopen in child process."""
        state = self.__dict__.copy()
        if state.get("_lmdb_pack_file") is not None:
            state["_lmdb_pack_file"].close()
            state["_lmdb_pack_file"] = None
        return state

    def __setstate__(self, state):
        """Restore state after unpickling in DataLoader worker."""
        self.__dict__ = state
        if self.lmdb_data_path is not None:
            pack_type_cls = get_packtype_from_path(self.lmdb_data_path)
            self._lmdb_pack_file = pack_type_cls(
                self.lmdb_data_path,
                writable=False,
            )
            self._lmdb_pack_file.open()
            self._lmdb_keys = self._lmdb_pack_file.get_keys()
