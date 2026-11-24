# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from __future__ import annotations
import logging
import os
import pickle
from typing import Dict, Optional, Sequence

import msgpack
import msgpack_numpy
import numpy as np
import torch
import torch.utils.data as data
from hat_engine.data.datasets.data_packer import Packer

from heal.data.datasets.navsim.navsim_dataset import NavsimDataset
from heal.data.datasets.navsim.navsim_feature_builder import (
    DiffusionPlannerFeatureBuilder,
    DiffusionPlannerLaneBoundaryFeatureBuilder,
)

msgpack_numpy.patch()

logger = logging.getLogger(__name__)

try:
    from nuplan.planning.simulation.trajectory.trajectory_sampling import (
        TrajectorySampling,
    )

    from heal.data.datasets.navsim.trajectory_target_builder import (
        TrajectoryTargetBuilder,
    )
except ImportError:  # nuplan / navsim not installed in this environment.
    TrajectorySampling = None
    TrajectoryTargetBuilder = None


class DiffusionPlannerNavsimParser(data.Dataset):
    """Parse packed navsim scenes into diffusion planner format at 2 Hz.

    Wraps :class:`NavsimDataset` to load ``Scene`` objects from the packed
    LMDB, then extracts planner features with the same logic as
    ``NavsimPlannerCacheDataset``'s ``use_perception_cache=False`` branch.

    Args:
        data_path: Path to the packed (lmdb) navsim scenes.
        split: Scene-filter split name, e.g. ``navtrain`` / ``navtest``.
        split_cfg_root: Directory holding the hydra split configs.
        map_root: nuplan maps root. Falls back to ``NUPLAN_MAPS_ROOT``.
        add_lane_boundary: Also pack lane-boundary features.
        mode: ``train`` or ``test``, forwarded to the feature builder.
    """

    def __init__(
        self,
        data_path: str,
        split: str = "navtrain",
        split_cfg_root: Optional[str] = None,
        map_root: Optional[str] = None,
        agent_num: int = 32,
        model_time_len: int = 4,
        static_objects_num: int = 5,
        max_ped_bike: int = 10,
        lane_num: int = 70,
        lane_len: int = 20,
        route_num: int = 25,
        route_len: int = 20,
        predicted_neighbor_num: int = 10,
        num_future_poses: int = 8,
        raw_frequency: float = 0.5,
        pad_frequency: float = 0.5,
        radius: float = 100.0,
        mode: str = "test",
        agents_future_frames: int = 8,
        add_lane_boundary: bool = False,
        map_min_length: float = 5.0,
        pc_range: Optional[Sequence[float]] = None,
        map_pc_range: Optional[Sequence[float]] = None,
        map_version: str = "nuplan-maps-v1.0",
    ):
        if map_root is not None:
            os.environ["NUPLAN_MAPS_ROOT"] = map_root

        self.navsim_dataset = NavsimDataset(
            data_path=data_path,
            split=split,
            pack_type="lmdb",
            split_cfg_root=split_cfg_root,
            map_root=map_root,
            sensor_config={},
            return_scene=True,
        )

        self.feature_builder = DiffusionPlannerFeatureBuilder(
            lane_num=lane_num,
            route_num=route_num,
            lane_len=lane_len,
            route_len=route_len,
            static_objects_num=static_objects_num,
            agent_num=agent_num,
            model_time_len=model_time_len,
            max_ped_bike=max_ped_bike,
            radius=radius,
            raw_frequency=raw_frequency,
            pad_frequency=pad_frequency,
            mode=mode,
            agents_future_frames=agents_future_frames,
            predicted_neighbor_num=predicted_neighbor_num,
            pc_range=pc_range,
            map_pc_range=map_pc_range,
            map_root=map_root,
            map_version=map_version,
        )

        self.lane_boundary_builder = None
        if add_lane_boundary:
            self.lane_boundary_builder = (
                DiffusionPlannerLaneBoundaryFeatureBuilder(
                    lane_num=lane_num,
                    lane_len=lane_len,
                    map_min_length=map_min_length,
                    radius=radius,
                    map_pc_range=map_pc_range,
                    map_root=map_root,
                    map_version=map_version,
                )
            )

        self.agent_num = agent_num
        self.model_time_len = model_time_len
        self.static_objects_num = static_objects_num
        self.max_ped_bike = max_ped_bike
        self.lane_num = lane_num
        self.lane_len = lane_len
        self.route_num = route_num
        self.route_len = route_len
        self.predicted_neighbor_num = predicted_neighbor_num
        self.agents_future_frames = agents_future_frames
        self.num_future_poses = num_future_poses
        self.add_lane_boundary = add_lane_boundary

        logger.info(
            "DiffusionPlannerNavsimParser: %d samples",
            len(self.navsim_dataset),
        )

    def __len__(self):
        return len(self.navsim_dataset)

    def __getitem__(self, idx):
        navsim_data = self.navsim_dataset[idx]
        scene = navsim_data.get("scene", None)
        token = navsim_data["token"]

        if scene is None:
            raise ValueError("Please set `return_scene` to True")

        gt_features = navsim_data.get("features", {})
        if not gt_features:
            gt_features = self.feature_builder.compute_features(scene)

        # Ego future GT at 2 Hz.
        targets = navsim_data.get("targets", {})
        if "trajectory" in targets and targets["trajectory"] != []:
            ego_agent_future = targets["trajectory"]
            if isinstance(ego_agent_future, torch.Tensor):
                ego_agent_future = ego_agent_future.numpy()
        else:
            ego_agent_future = _extract_ego_future_gt(
                scene,
                num_future_poses=self.num_future_poses,
            )

        info = {
            "token": token,
            "ego_current_state": np.asarray(gt_features["ego_current_state"]),
            "ego_future_gt": ego_agent_future,
            "neighbor_agents_past": np.asarray(
                gt_features["neighbor_agents_past"]
            ),
            "neighbors_future_gt": np.asarray(
                gt_features.get(
                    "neighbors_future_gt",
                    np.zeros(
                        (
                            self.predicted_neighbor_num,
                            self.agents_future_frames,
                            3,
                        ),
                        dtype=np.float32,
                    ),
                )
            ),
            "static_objects": np.asarray(gt_features["static_objects"]),
            "lanes": np.asarray(gt_features["lanes"]),
            "lanes_speed_limit": np.asarray(gt_features["lanes_speed_limit"]),
            "lanes_has_speed_limit": np.asarray(
                gt_features["lanes_has_speed_limit"]
            ),
            "route_lanes": np.asarray(gt_features["route_lanes"]),
            "route_lanes_speed_limit": np.asarray(
                gt_features["route_lanes_speed_limit"]
            ),
            "route_lanes_has_speed_limit": np.asarray(
                gt_features["route_lanes_has_speed_limit"]
            ),
        }

        if self.lane_boundary_builder is not None:
            lane_boundary_features = (
                self.lane_boundary_builder.compute_features(scene)
            )
            lane_boundaries = lane_boundary_features["lane_boundaries"]
            for key in lane_boundaries:
                lane_boundaries[key] = np.asarray(lane_boundaries[key])
            info["lane_boundaries"] = lane_boundaries

        return info


class DiffusionPlannerNavsimPacker(Packer):
    """Packer for navsim dataset in diffusion planner format.

    Wraps :class:`DiffusionPlannerNavsimParser` and serializes each sample
    with msgpack.
    """

    def __init__(
        self,
        data_path: str,
        split: str = "navtrain",
        split_cfg_root: Optional[str] = None,
        map_root: Optional[str] = None,
        num_workers: int = 20,
        pack_type: str = "lmdb",
        target_data_dir: str = "",
        agent_num: int = 32,
        model_time_len: int = 4,
        static_objects_num: int = 5,
        max_ped_bike: int = 10,
        lane_num: int = 70,
        lane_len: int = 20,
        route_num: int = 25,
        route_len: int = 20,
        predicted_neighbor_num: int = 10,
        num_future_poses: int = 8,
        raw_frequency: float = 0.5,
        pad_frequency: float = 0.5,
        mode: str = "test",
        agents_future_frames: int = 8,
        radius: float = 100.0,
        num_samples: Optional[int] = None,
        add_lane_boundary: bool = False,
        map_min_length: float = 5.0,
        pc_range: Optional[Sequence[float]] = None,
        map_pc_range: Optional[Sequence[float]] = None,
        map_version: str = "nuplan-maps-v1.0",
        **kwargs,
    ):
        self.dataset = DiffusionPlannerNavsimParser(
            data_path=data_path,
            split=split,
            split_cfg_root=split_cfg_root,
            map_root=map_root,
            agent_num=agent_num,
            model_time_len=model_time_len,
            static_objects_num=static_objects_num,
            max_ped_bike=max_ped_bike,
            lane_num=lane_num,
            lane_len=lane_len,
            route_num=route_num,
            route_len=route_len,
            predicted_neighbor_num=predicted_neighbor_num,
            num_future_poses=num_future_poses,
            raw_frequency=raw_frequency,
            pad_frequency=pad_frequency,
            mode=mode,
            radius=radius,
            agents_future_frames=agents_future_frames,
            add_lane_boundary=add_lane_boundary,
            map_min_length=map_min_length,
            pc_range=pc_range,
            map_pc_range=map_pc_range,
            map_version=map_version,
        )
        if num_samples is None:
            num_samples = len(self.dataset)
        super().__init__(
            target_data_dir, num_samples, pack_type, num_workers, **kwargs
        )

    def pack_data(self, idx):
        info = self.dataset[idx]
        return msgpack.packb(info)

    def __call__(self):
        """Pack data and then build a token-to-index PKL file."""
        super().__call__()
        self._save_token_index()

    def _save_token_index(self):
        """Build token->LMDB-index mapping and save as a PKL file."""
        import lmdb as lmdb_lib

        self.pack_file.close()
        lmdb_path = self.pack_file.uri
        pkl_path = os.path.join(lmdb_path, "token_to_idx.pkl")

        logger.info("Building token->index mapping from LMDB ...")

        env = lmdb_lib.open(lmdb_path, readonly=True, lock=False)
        token_to_idx: Dict[str, int] = {}
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                try:
                    idx = int(key)
                except ValueError:
                    continue
                try:
                    packed = msgpack.unpackb(value, raw=False)
                    token = packed.get("token", None)
                    if token is not None:
                        token_to_idx[token] = idx
                except Exception:
                    logger.warning("Failed to decode entry %s", key)

        env.close()

        with open(pkl_path, "wb") as f:
            pickle.dump(token_to_idx, f)

        logger.info(
            "Saved token->index mapping (%d entries) to %s",
            len(token_to_idx),
            pkl_path,
        )


def _extract_ego_future_gt(scene, num_future_poses=8):
    """Extract ego future GT trajectory at native 2 Hz from Scene frames."""
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
