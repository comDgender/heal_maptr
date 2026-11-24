# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from hat_engine.registry import OBJECT_REGISTRY

from heal.data.datasets.navsim.nuplan_utils.ego_process import (
    calculate_additional_ego_states,
)
from heal.data.datasets.navsim.nuplan_utils.map_process import (
    get_neighbor_vector_set_map,
    map_process,
)
from heal.data.datasets.navsim.nuplan_utils.roadblock_utils import (
    route_roadblock_correction,
)

try:
    from nuplan.common.actor_state.state_representation import (
        Point2D,
        TimePoint,
    )
    from nuplan.common.actor_state.vehicle_parameters import (
        get_pacifica_parameters,
    )
    from nuplan.common.maps.maps_datatypes import (
        TrafficLightStatusData,
        TrafficLightStatusType,
    )
except ImportError:
    Point2D = None
    TrafficLightStatusData = None
    TrafficLightStatusType = None
    get_pacifica_parameters = None
    TimePoint = None

try:
    from navsim.common.enums import BoundingBoxIndex
    from navsim.planning.scenario_builder.navsim_scenario_utils import (
        ego_status_to_ego_state,
    )
except ImportError:
    BoundingBoxIndex = None
    ego_status_to_ego_state = None


_NAVSIM_AGENT_TYPES = {
    "vehicle": [1, 0, 0],
    "pedestrian": [0, 1, 0],
    "bicycle": [0, 0, 1],
}

_NAVSIM_STATIC_TYPES = {
    "czone_sign": [1, 0, 0, 0],
    "barrier": [0, 1, 0, 0],
    "traffic_cone": [0, 0, 1, 0],
    "generic_object": [0, 0, 0, 1],
}


def _local_to_current_ego(x, y, heading, frame_ego_pose, current_ego_pose):
    """Transform position from a history frame's local coords \
        to current ego's local coords.

    Navsim annotations.boxes are in each frame's ego/lidar local frame.
    """
    fx, fy, fh = frame_ego_pose
    cx, cy, ch = current_ego_pose

    cos_fh, sin_fh = np.cos(fh), np.sin(fh)
    gx = cos_fh * x - sin_fh * y + fx
    gy = sin_fh * x + cos_fh * y + fy
    gh = heading + fh

    dx, dy = gx - cx, gy - cy
    cos_ch, sin_ch = np.cos(ch), np.sin(ch)
    lx = cos_ch * dx + sin_ch * dy
    ly = -sin_ch * dx + cos_ch * dy
    lh = gh - ch
    return lx, ly, lh


def _rotate_velocity(vx, vy, from_heading, to_heading):
    """Rotate velocity vector between two heading frames."""
    dh = from_heading - to_heading
    cos_dh, sin_dh = np.cos(dh), np.sin(dh)
    return cos_dh * vx - sin_dh * vy, sin_dh * vx + cos_dh * vy


@OBJECT_REGISTRY.register
class DiffusionPlannerFeatureBuilder:
    """Converts a navsim Scene to the input dict expected by Diffusion_Planner.

    Set ``needs_scene = True`` so that NavsimDataset.build_feature_target
    passes the full Scene object rather than an AgentInput.
    """

    needs_scene = True

    def __init__(
        self,
        agent_num: int = 32,
        lane_num: int = 70,
        route_num: int = 25,
        lane_len: int = 20,
        route_len: int = 20,
        static_objects_num: int = 5,
        model_time_len: int = 21,
        max_ped_bike: int = 10,
        radius: float = 100.0,
        raw_frequency: float = 0.5,
        pad_frequency: float = 0.1,
        mode: str = "test",
        agents_future_frames: int = 8,
        predicted_neighbor_num: int = 10,
        return_bbox3d: bool = False,
        pc_range: Optional[Sequence[float]] = None,
        map_pc_range: Optional[Sequence[float]] = None,
        map_root: Optional[str] = None,
        map_version: str = "nuplan-maps-v1.0",
    ):
        self.agent_num = agent_num
        self.lane_num = lane_num
        self.route_num = route_num
        self.lane_len = lane_len
        self.route_len = route_len
        self.static_objects_num = static_objects_num
        self.model_time_len = model_time_len
        self.max_ped_bike = max_ped_bike
        self._radius = radius
        self.raw_frequency = raw_frequency
        self.pad_frequency = pad_frequency
        self.mode = mode
        self.agents_future_frames = agents_future_frames
        self.predicted_neighbor_num = predicted_neighbor_num
        self.return_bbox3d = return_bbox3d
        self.pc_range = pc_range
        self.map_pc_range = map_pc_range

        self._map_features = [
            "LANE",
            "LEFT_BOUNDARY",
            "RIGHT_BOUNDARY",
            "ROUTE_LANES",
        ]
        self._max_elements = {
            "LANE": lane_num,
            "LEFT_BOUNDARY": lane_num,
            "RIGHT_BOUNDARY": lane_num,
            "ROUTE_LANES": route_num,
        }
        self._max_points = {
            "LANE": lane_len,
            "LEFT_BOUNDARY": lane_len,
            "RIGHT_BOUNDARY": lane_len,
            "ROUTE_LANES": route_len,
        }

        if self.map_pc_range is not None:
            from heal.data.datasets.navsim.navsim_perception_dataset import (  # noqa
                get_maps_api,
            )

            self.MAPS = [
                "sg-one-north",
                "us-ma-boston",
                "us-nv-las-vegas-strip",
                "us-pa-pittsburgh-hazelwood",
            ]
            self.map_api_info = {}
            for loc in self.MAPS:
                self.map_api_info[loc] = get_maps_api(
                    map_root, map_version, loc
                )

    def get_unique_name(self) -> str:
        return "diffusion_planner_feature"

    def compute_features(self, scene) -> Dict[str, Any]:
        num_history = scene.scene_metadata.num_history_frames
        current_idx = num_history - 1
        current_frame = scene.frames[current_idx]

        ego_pose = np.array(
            current_frame.ego_status.ego_pose, dtype=np.float64
        )

        neighbor_agents_past, selected_tokens, boxes_3d = self._extract_agents(
            scene, ego_pose
        )  # (32, 21, 11), (32, 7)
        static_objects, static_boxes_3d = self._extract_static_objects(
            scene, ego_pose
        )  # (5, 10), (5, 7)
        # Match nuplan DataProcessor.observation_adapter() for inference:
        # ego centric x/y and heading only; other entries set to 0.

        map_data = self._extract_map(scene, ego_pose)

        data = {
            "neighbor_agents_past": torch.tensor(
                neighbor_agents_past, dtype=torch.float32
            ),
            "static_objects": torch.tensor(
                static_objects, dtype=torch.float32
            ),
        }
        if self.return_bbox3d:
            data["agent_boxes_3d"] = torch.tensor(
                boxes_3d, dtype=torch.float32
            )
            data["static_boxes_3d"] = torch.tensor(
                static_boxes_3d, dtype=torch.float32
            )
        if self.mode == "train":
            neighbor_agents_future = self._extract_neighbors_future_gt(
                scene=scene,
                selected_tokens=selected_tokens,
                predicted_neighbor_num=self.predicted_neighbor_num,
                num_future_poses=self.agents_future_frames,
                max_ped_bike=self.max_ped_bike,
            )
            data["neighbors_future_gt"] = torch.tensor(
                neighbor_agents_future, dtype=torch.float32
            )
            ego_current_state = self._extract_ego_current_state(
                scene, ego_pose
            )
            data["ego_current_state"] = torch.tensor(
                ego_current_state, dtype=torch.float32
            )
        else:
            ego_current_state = np.array(
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                dtype=np.float32,
            )  # (10, )
            data["ego_current_state"] = torch.tensor(
                ego_current_state, dtype=torch.float32
            )
        for k, v in map_data.items():
            arr = np.asarray(v)
            if arr.dtype == np.bool_:
                data[k] = torch.tensor(arr, dtype=torch.bool)
            else:
                data[k] = torch.tensor(arr, dtype=torch.float32)
        # "lanes": (70, 20, 12)
        # "lanes_speed_limit": (70, 1)
        # "lanes_has_speed_limit": (70, 1)
        # "route_lanes": (1, 20, 12)
        # "route_lanes_speed_limit": (1, 1)
        # "route_lanes_has_speed_limit": (1, 1)
        return data

    # ------------------------------------------------------------------
    # Ego current state
    # ------------------------------------------------------------------
    def _extract_ego_current_state(self, scene, current_ego_pose):
        """Build ego_agent_past and time_stamps, then compute ego_current_state.

        Follows the same logic as DataProcessor.process_scenario() which calls
        calculate_additional_ego_states(ego_agent_past, time_stamps_past).

        Only the last two frames are needed because
        calculate_additional_ego_states uses ego_agent_past[-1] and [-2].

        Args:
            scene: Navsim Scene object.
            current_ego_pose: np.ndarray [x, y, heading] of current ego.

        Returns:
            np.ndarray of shape (10,) with
            [x, y, cos_h, sin_h, vx, vy, ax, ay, steering_angle, yaw_rate].
        """
        num_history = scene.scene_metadata.num_history_frames
        ego_agent_past = np.zeros((2, 7), dtype=np.float64)
        time_stamps = np.zeros(2, dtype=np.int64)

        for i, f_idx in enumerate(range(num_history - 2, num_history)):
            frame = scene.frames[f_idx]
            frame_ego_pose = frame.ego_status.ego_pose
            es = frame.ego_status

            # The ego vehicle is at the origin of its own local frame (0, 0, 0)
            # _local_to_current_ego expects local coords, not global coords.
            lx, ly, lh = _local_to_current_ego(
                0.0,
                0.0,
                0.0,
                frame_ego_pose,
                current_ego_pose,
            )
            # Transform velocity to current-ego local frame
            lvx, lvy = _rotate_velocity(
                es.ego_velocity[0],
                es.ego_velocity[1],
                frame_ego_pose[2],
                current_ego_pose[2],
            )
            # Transform acceleration to current-ego local frame
            lax, lay = _rotate_velocity(
                es.ego_acceleration[0],
                es.ego_acceleration[1],
                frame_ego_pose[2],
                current_ego_pose[2],
            )

            ego_agent_past[i] = [lx, ly, lh, lvx, lvy, lax, lay]
            time_stamps[i] = frame.timestamp

        return calculate_additional_ego_states(ego_agent_past, time_stamps)

    # ------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------
    def _extract_agents(self, scene, current_ego_pose):
        num_history = scene.scene_metadata.num_history_frames
        current_fidx = num_history - 1

        # Collect all agent observations across frames
        agent_frames: Dict[str, dict] = {}
        current_frame_types: Dict[str, List[int]] = {}
        current_frame_3d_boxes: Dict[str, np.ndarray] = {}

        for f_idx in range(num_history):
            frame = scene.frames[f_idx]
            frame_ego = frame.ego_status.ego_pose
            ann = frame.annotations

            for a_idx in range(len(ann.track_tokens)):
                token = ann.track_tokens[a_idx]
                name = ann.names[a_idx]
                if name not in _NAVSIM_AGENT_TYPES:
                    continue

                box = ann.boxes[a_idx]
                vel = ann.velocity_3d[a_idx]

                if self.pc_range is not None:
                    x_min, y_min, _, x_max, y_max, _ = self.pc_range
                    if (
                        box[0] < x_min
                        or box[0] > x_max
                        or box[1] < y_min
                        or box[1] > y_max
                    ):
                        continue

                lx, ly, lh = _local_to_current_ego(
                    float(box[0]),
                    float(box[1]),
                    float(box[6]),
                    frame_ego,
                    current_ego_pose,
                )
                lvx, lvy = _rotate_velocity(
                    float(vel[0]),
                    float(vel[1]),
                    frame_ego[2],
                    current_ego_pose[2],
                )

                if token not in agent_frames:
                    agent_frames[token] = {}

                agent_frames[token][f_idx] = np.array(
                    [
                        lx,
                        ly,
                        np.cos(lh),
                        np.sin(lh),
                        lvx,
                        lvy,
                        float(box[4]),
                        float(box[3]),
                    ],
                    dtype=np.float32,
                )  # box[3]:length(->y), box[4]:width(->x)

                # Agent type and 3D box from the current frame only
                if f_idx == current_fidx:
                    current_frame_types[token] = _NAVSIM_AGENT_TYPES[name]
                    # 3D box from current frame only, so z is already in
                    # current-ego local frame (no transform needed).
                    # Format: [lx, ly, z, w, l, h, lh]
                    current_frame_3d_boxes[token] = np.array(
                        [
                            lx,
                            ly,
                            float(box[2]),
                            float(box[4]),
                            float(box[3]),
                            float(box[5]),
                            lh,
                        ],
                        dtype=np.float32,
                    )

        # Filter: only keep agents that appear in the current frame
        current_tokens = list(current_frame_types.keys())
        N = len(current_tokens)

        if N == 0:
            return (
                np.zeros(
                    (self.agent_num, self.model_time_len, 11),
                    dtype=np.float32,
                ),
                [],
                np.zeros((self.agent_num, 7), dtype=np.float32),
            )

        # Backward-fill: pad missing frames with the nearest future state
        agents_navsim = np.zeros((N, num_history, 8), dtype=np.float32)
        for a, tok in enumerate(current_tokens):
            cur_state = agent_frames[tok][current_fidx]
            agents_navsim[a, current_fidx] = cur_state
            for f_idx in range(current_fidx - 1, -1, -1):
                if f_idx in agent_frames[tok]:
                    cur_state = agent_frames[tok][f_idx]
                agents_navsim[a, f_idx] = cur_state

        # Nearest-neighbour upsample from navsim 2 Hz to model 10 Hz
        navsim_times = np.array(
            [
                -(num_history - 1 - i) * self.raw_frequency
                for i in range(num_history)
            ]
        )
        model_times = np.array(
            [
                -(self.model_time_len - 1 - j) * self.pad_frequency
                for j in range(self.model_time_len)
            ]
        )
        nn_map = np.argmin(
            np.abs(model_times[:, None] - navsim_times[None, :]), axis=1
        )

        upsampled = agents_navsim[:, nn_map, :]  # (N, model_time_len, 8)

        # Append one-hot type → 11-dim
        type_arr = np.zeros((N, self.model_time_len, 11), dtype=np.float32)
        type_arr[:, :, :8] = upsampled
        for a, tok in enumerate(current_tokens):
            type_arr[a, :, 8:11] = current_frame_types[tok]

        # Select top agents by distance, balancing ped/bike vs vehicle
        dist = np.linalg.norm(type_arr[:, -1, :2], axis=-1)
        sorted_idx = np.argsort(dist)

        pb = [
            i
            for i in sorted_idx
            if current_frame_types[current_tokens[i]] != [1, 0, 0]
        ]
        veh = [
            i
            for i in sorted_idx
            if current_frame_types[current_tokens[i]] == [1, 0, 0]
        ]

        if len(pb) + len(veh) <= self.agent_num:
            selected = list(sorted_idx[: self.agent_num])
        else:
            sel_pb = pb[: self.max_ped_bike]
            selected = sel_pb + veh
            remain = self.agent_num - len(selected)
            if remain > 0:
                selected += pb[self.max_ped_bike : self.max_ped_bike + remain]
            selected = sorted(selected, key=lambda i: dist[i])[
                : self.agent_num
            ]

        result = np.zeros(
            (self.agent_num, self.model_time_len, 11), dtype=np.float32
        )
        boxes_3d = np.zeros((self.agent_num, 7), dtype=np.float32)
        selected_tokens = []
        for i, idx in enumerate(selected):
            result[i] = type_arr[idx]
            boxes_3d[i] = current_frame_3d_boxes[current_tokens[idx]]
            selected_tokens.append(current_tokens[idx])
        return result, selected_tokens, boxes_3d

    # ------------------------------------------------------------------
    # Static objects
    # ------------------------------------------------------------------
    def _extract_static_objects(self, scene, current_ego_pose):
        current_idx = scene.scene_metadata.num_history_frames - 1
        ann = scene.frames[current_idx].annotations

        objs = []
        objs_3d = []
        for a_idx in range(len(ann.names)):
            name = ann.names[a_idx]
            if name not in _NAVSIM_STATIC_TYPES:
                continue
            box = ann.boxes[a_idx]
            if self.pc_range is not None:
                x_min, y_min, _, x_max, y_max, _ = self.pc_range
                if (
                    box[0] < x_min
                    or box[0] > x_max
                    or box[1] < y_min
                    or box[1] > y_max
                ):
                    continue
            bx, by = float(box[0]), float(box[1])
            bh = float(box[6])
            objs.append(
                np.array(
                    [
                        bx,
                        by,
                        np.cos(bh),
                        np.sin(bh),
                        float(box[4]),
                        float(box[3]),
                        *_NAVSIM_STATIC_TYPES[name],
                    ],
                    dtype=np.float32,
                )
            )  # box[3]:length(->x), box[4]:width(->y)
            # 3D box from current frame only, so z is already in
            # current-ego local frame (no transform needed).
            # Format: [bx, by, z, w, l, h, bh]
            objs_3d.append(
                np.array(
                    [
                        bx,
                        by,
                        float(box[2]),
                        float(box[4]),
                        float(box[3]),
                        float(box[5]),
                        bh,
                    ],
                    dtype=np.float32,
                )
            )

        result = np.zeros((self.static_objects_num, 10), dtype=np.float32)
        boxes_3d = np.zeros((self.static_objects_num, 7), dtype=np.float32)
        if objs:
            objs_arr = np.stack(objs)
            objs_3d_arr = np.stack(objs_3d)
            dists = np.linalg.norm(objs_arr[:, :2], axis=-1)
            top_idx = np.argsort(dists)[: self.static_objects_num]
            for i, idx in enumerate(top_idx):
                result[i] = objs_arr[idx]
                boxes_3d[i] = objs_3d_arr[idx]
        return result, boxes_3d

    # ------------------------------------------------------------------
    # Map
    # ------------------------------------------------------------------
    def _extract_map(self, scene, anchor_ego_state):
        current_idx = scene.scene_metadata.num_history_frames - 1
        current_frame = scene.frames[current_idx]
        ego_pose = current_frame.ego_status.ego_pose
        ego_coords = Point2D(float(ego_pose[0]), float(ego_pose[1]))
        route_ids = current_frame.roadblock_ids
        map_api = scene.map_api

        # Match nuplan DataProcessor.observation_adapter() behavior:
        # correct disconnected / off-route segments for vector map extraction.
        if (
            route_roadblock_correction is not None
            and ego_status_to_ego_state is not None
            and get_pacifica_parameters is not None
            and TimePoint is not None
        ):
            ego_state_for_route = ego_status_to_ego_state(
                ego_status=current_frame.ego_status,
                vehicle_parameters=get_pacifica_parameters(),
                time_point=TimePoint(int(current_frame.timestamp)),
            )
            route_ids = route_roadblock_correction(
                ego_state=ego_state_for_route,
                map_api=map_api,
                route_roadblock_ids=route_ids,
            )

        tl_data = []
        if TrafficLightStatusData is not None:
            # navsim Frame.traffic_lights stores (lane_connector_id, is_red)
            tl_ts = int(current_frame.timestamp)
            for lc_id, is_red in current_frame.traffic_lights:
                status = (
                    TrafficLightStatusType.RED
                    if is_red
                    else TrafficLightStatusType.GREEN
                )
                tl_data.append(
                    TrafficLightStatusData(
                        status=status,
                        lane_connector_id=int(lc_id),
                        timestamp=tl_ts,
                    )
                )

        if self.map_pc_range is not None:
            map_api_custom = self.map_api_info[scene.scene_metadata.map_name]
            (
                coords,
                tl_out,
                speed_limit,
                lane_route,
            ) = get_neighbor_vector_set_map(
                map_api_custom,
                self._map_features,
                ego_coords,
                self.map_pc_range,
                tl_data,
            )
        else:
            (
                coords,
                tl_out,
                speed_limit,
                lane_route,
            ) = get_neighbor_vector_set_map(
                map_api,
                self._map_features,
                ego_coords,
                self._radius,
                tl_data,
            )
        vector_map = map_process(
            route_ids,
            anchor_ego_state,
            coords,
            tl_out,
            speed_limit,
            lane_route,
            self._map_features,
            self._max_elements,
            self._max_points,
        )
        return vector_map

    # ------------------------------------------------------------------
    # Agents Future
    # ------------------------------------------------------------------
    def _extract_neighbors_future_gt(
        self,
        scene,
        selected_tokens,
        predicted_neighbor_num=10,
        num_future_poses=8,
        max_ped_bike=10,
    ):
        """Extract future GT trajectories for agents selected by _extract_agents.

        Follows the same pattern as nuplan's agent_future_process: only the
        agents that appear in the current frame and were selected for the past
        feature are kept.  Future poses are in the current ego's local frame.

        Args:
            scene: Navsim Scene object with future frames.
            selected_tokens: List of track_tokens returned by _extract_agents,
                ordered to match the rows of neighbor_agents_past.
            predicted_neighbor_num: Number of neighbor slots (must match
                agent_num or the planner's expected dimension).
            num_future_poses: Number of future time steps to output.
            max_ped_bike: Unused (kept for interface compatibility).

        Returns:
            np.ndarray of shape (predicted_neighbor_num, num_future_poses, 3)
            with [x, y, heading] in current-ego local coords.
        """
        num_history = scene.scene_metadata.num_history_frames
        num_future = scene.scene_metadata.num_future_frames

        current_frame = scene.frames[num_history - 1]
        current_ego_pose = np.array(
            current_frame.ego_status.ego_pose, dtype=np.float64
        )

        # Collect future positions per track_token across future frames
        agent_futures: Dict[str, dict] = {}
        for f_idx in range(num_history, num_history + num_future):
            frame = scene.frames[f_idx]
            frame_ego_pose = frame.ego_status.ego_pose
            ann = frame.annotations

            for a_idx in range(len(ann.track_tokens)):
                token = ann.track_tokens[a_idx]
                # Only keep tokens that were selected in _extract_agents
                if token not in selected_tokens:
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

        # Build output aligned with selected_tokens order
        result = np.zeros(
            (predicted_neighbor_num, num_future_poses, 3),
            dtype=np.float32,
        )
        for i, token in enumerate(selected_tokens):
            if i >= predicted_neighbor_num:
                break
            if token not in agent_futures:
                continue
            for step, pose in agent_futures[token].items():
                if step < num_future_poses:
                    result[i, step] = pose

        return result


@OBJECT_REGISTRY.register
class DiffusionPlannerLaneBoundaryFeatureBuilder:
    """Lane-boundary features for diffusion planner packing.

    Migrated from heal_e2e ``DiffusionPlannerLaneBoundaryFeatureBuilder``.
    Set ``needs_scene = True`` so NavsimDataset passes the full Scene.
    """

    needs_scene = True

    def __init__(
        self,
        lane_num: int = 70,
        lane_len: int = 20,
        map_min_length: float = 5.0,
        radius: float = 100.0,
        map_pc_range: Optional[Sequence[float]] = None,
        map_root: Optional[str] = None,
        map_version: str = "nuplan-maps-v1.0",
    ):
        self.lane_num = lane_num
        self.lane_len = lane_len
        self.map_min_length = map_min_length
        self.radius = radius
        self.map_pc_range = map_pc_range

        self._map_features = [
            "LANE",
            "LEFT_BOUNDARY",
            "RIGHT_BOUNDARY",
            "STOP_LINE",
            "CROSSWALK",
        ]
        self._max_elements = {
            "LANE": -1,
            "LEFT_BOUNDARY": -1,
            "RIGHT_BOUNDARY": -1,
            "STOP_LINE": -1,
            "CROSSWALK": -1,
        }
        self._max_points = {
            "LANE": lane_len,
            "LEFT_BOUNDARY": lane_len,
            "RIGHT_BOUNDARY": lane_len,
            "STOP_LINE": lane_len,
            "CROSSWALK": lane_len,
        }

        if self.map_pc_range is not None:
            from heal.data.datasets.navsim.navsim_perception_dataset_utils import (  # noqa: E501
                get_maps_api,
            )

            self.MAPS = [
                "sg-one-north",
                "us-ma-boston",
                "us-nv-las-vegas-strip",
                "us-pa-pittsburgh-hazelwood",
            ]
            self.map_api_info = {}
            for loc in self.MAPS:
                self.map_api_info[loc] = get_maps_api(
                    map_root, map_version, loc
                )

    def get_unique_name(self) -> str:
        return "diffusion_planner_line_boundary_feature"

    def compute_features(self, scene) -> Dict[str, Any]:
        num_history = scene.scene_metadata.num_history_frames
        current_idx = num_history - 1
        current_frame = scene.frames[current_idx]

        ego_pose = np.array(
            current_frame.ego_status.ego_pose, dtype=np.float64
        )
        map_data = self._extract_map_boundary(scene, ego_pose)

        data = {"lane_boundaries": {}}
        for k, v in map_data.items():
            arr = np.asarray(v)
            if arr.dtype == np.bool_:
                data["lane_boundaries"][k] = torch.tensor(
                    arr, dtype=torch.bool
                )
            else:
                data["lane_boundaries"][k] = torch.tensor(
                    arr, dtype=torch.float32
                )
        return data

    def _extract_map_boundary(self, scene, anchor_ego_state):
        from heal.data.datasets.navsim.navsim_perception_dataset_utils import (
            get_lane_gt_by_nms,
        )
        from heal.data.datasets.navsim.navsim_perception_dataset_utils import (
            get_neighbor_vector_set_map as gn,
        )
        from heal.data.datasets.navsim.navsim_perception_dataset_utils import (
            map_process as mp,
        )

        current_idx = scene.scene_metadata.num_history_frames - 1
        current_frame = scene.frames[current_idx]
        ego_pose = current_frame.ego_status.ego_pose
        ego_coords = Point2D(float(ego_pose[0]), float(ego_pose[1]))
        route_roadblock_ids = []

        if self.map_pc_range is not None:
            map_api_custom = self.map_api_info[scene.scene_metadata.map_name]
            coords, speed_limit, lane_route = gn(
                map_api_custom,
                self._map_features,
                ego_coords,
                self.map_pc_range,
            )
        else:
            map_api = scene.map_api
            coords, speed_limit, lane_route = gn(
                map_api, self._map_features, ego_coords, self.radius
            )

        vector_map = mp(
            route_roadblock_ids,
            anchor_ego_state,
            coords,
            speed_limit,
            lane_route,
            self._map_features,
            self._max_elements,
            self._max_points,
            False,
        )

        if "filtered_lane_data_thr20" in vector_map:
            lanes_pecetion = np.array(
                vector_map["filtered_lane_data_thr20"], dtype=np.float32
            )
        else:
            lanes_pecetion = vector_map["lanes_pecetion"]
            lanes_pecetion = self.fliter_lane_gt_by_length(
                lanes_pecetion, min_length=self.map_min_length
            )
            lanes_pecetion = get_lane_gt_by_nms(lanes_pecetion)

        lanes_out = np.zeros(
            (self.lane_num, self.lane_len, 12), dtype=np.float64
        )
        speed_limit_out = np.zeros((self.lane_num, 1), dtype=np.float32)
        has_speed_limit_out = np.zeros((self.lane_num, 1), dtype=np.bool_)
        mapping = {}
        for i, e in enumerate(lanes_pecetion):
            dist = np.linalg.norm(e, axis=-1).min()
            mapping[i] = dist
        mapping = sorted(mapping.items(), key=lambda item: item[1])
        sorted_elements = mapping[: self.lane_num]
        sorted_idx = np.array([i[0] for i in sorted_elements])

        vectors = np.zeros(
            (lanes_pecetion.shape[0], self.lane_len, 2), dtype=np.float32
        )
        vectors[:, :-1, :] = (
            lanes_pecetion[:, 1:, :] - lanes_pecetion[:, :-1, :]
        )
        lanes_out[: len(sorted_idx), :, 0:2] = lanes_pecetion[sorted_idx][
            :, :, 0:2
        ]
        lanes_out[: len(sorted_idx), :, 2:4] = vectors[sorted_idx][:, :, 0:2]
        return {
            "lanes": lanes_out,
            "lanes_speed_limit": speed_limit_out,
            "lanes_has_speed_limit": has_speed_limit_out,
        }

    def fliter_lane_gt_by_length(self, lane_data, min_length=5.0):
        lane_length = np.sum(
            np.linalg.norm(lane_data[:, 1:] - lane_data[:, :-1], axis=2),
            axis=1,
        )
        return lane_data[lane_length > min_length]
