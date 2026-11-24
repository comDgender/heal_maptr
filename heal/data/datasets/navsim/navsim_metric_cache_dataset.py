# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from __future__ import annotations
import inspect
import logging
import os
import pickle
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, Optional

import msgpack
import msgpack_numpy
import numpy as np
import shapely.wkb
from hat_engine.data.datasets.data_packer import Packer
from hat_engine.utils.pack_type import Lmdb

_NUPLAN_NMC_IMPORT_ERROR: Optional[BaseException] = None
_NAVSIM_NMC_IMPORT_ERROR: Optional[BaseException] = None
_HAS_NUPLAN_NMC = False
_HAS_NAVSIM_NMC = False

Agent = Any
EgoState = Any
OrientedBox = Any
SceneObjectMetadata = Any
StateSE2 = Any
StateVector2D = Any
TimePoint = Any
StaticObject = Any
DetectionsTracks = Any
AGENT_TYPES: Any = ()
TrackedObjectType = Any
TrackedObjects = Any
SemanticMapLayer = Any
InterpolatedTrajectory = Any
NavSimScenario = Any
TrajectorySampling = Any
get_pacifica_parameters = None  # type: ignore[assignment]

_VEHICLE_PARAMS = None

try:
    from nuplan.common.actor_state.agent import Agent as _Agent
    from nuplan.common.actor_state.ego_state import EgoState as _EgoState
    from nuplan.common.actor_state.oriented_box import (
        OrientedBox as _OrientedBox,
    )
    from nuplan.common.actor_state.scene_object import (
        SceneObjectMetadata as _SceneObjectMetadata,
    )
    from nuplan.common.actor_state.state_representation import (
        StateSE2 as _StateSE2,
    )
    from nuplan.common.actor_state.state_representation import (
        StateVector2D as _StateVector2D,
    )
    from nuplan.common.actor_state.state_representation import (
        TimePoint as _TimePoint,
    )
    from nuplan.common.actor_state.static_object import (
        StaticObject as _StaticObject,
    )
    from nuplan.common.actor_state.tracked_objects import (
        TrackedObjects as _TrackedObjects,
    )
    from nuplan.common.actor_state.tracked_objects_types import (
        AGENT_TYPES as _AGENT_TYPES,
    )
    from nuplan.common.actor_state.tracked_objects_types import (
        TrackedObjectType as _TrackedObjectType,
    )
    from nuplan.common.actor_state.vehicle_parameters import (
        get_pacifica_parameters as _gpp,
    )
    from nuplan.common.maps.maps_datatypes import (
        SemanticMapLayer as _SemanticMapLayer,
    )
    from nuplan.planning.simulation.observation.observation_type import (
        DetectionsTracks as _DetectionsTracks,
    )
    from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
        InterpolatedTrajectory as _InterpolatedTrajectory,
    )
    from nuplan.planning.simulation.trajectory.trajectory_sampling import (
        TrajectorySampling as _TrajectorySampling,
    )

    Agent = _Agent
    EgoState = _EgoState
    OrientedBox = _OrientedBox
    SceneObjectMetadata = _SceneObjectMetadata
    StateSE2 = _StateSE2
    StateVector2D = _StateVector2D
    TimePoint = _TimePoint
    StaticObject = _StaticObject
    DetectionsTracks = _DetectionsTracks
    AGENT_TYPES = _AGENT_TYPES
    TrackedObjectType = _TrackedObjectType
    TrackedObjects = _TrackedObjects
    SemanticMapLayer = _SemanticMapLayer
    get_pacifica_parameters = _gpp
    InterpolatedTrajectory = _InterpolatedTrajectory
    TrajectorySampling = _TrajectorySampling
    _VEHICLE_PARAMS = get_pacifica_parameters()
    _HAS_NUPLAN_NMC = True
except ImportError as exc:
    _NUPLAN_NMC_IMPORT_ERROR = exc
    _VEHICLE_PARAMS = None
    Agent = Any
    EgoState = Any
    OrientedBox = Any
    SceneObjectMetadata = Any
    StateSE2 = Any
    StateVector2D = Any
    TimePoint = Any
    StaticObject = Any
    DetectionsTracks = Any
    AGENT_TYPES = ()
    TrackedObjectType = Any
    TrackedObjects = Any
    SemanticMapLayer = Any
    InterpolatedTrajectory = Any
    TrajectorySampling = Any
    get_pacifica_parameters = None  # type: ignore[assignment]

try:
    from navsim.planning.scenario_builder.navsim_scenario import (
        NavSimScenario as _NavSimScenario,
    )

    NavSimScenario = _NavSimScenario
    _HAS_NAVSIM_NMC = True
except ImportError as exc:
    _NAVSIM_NMC_IMPORT_ERROR = exc
    NavSimScenario = Any

_HAS_NMC_DEPS = _HAS_NUPLAN_NMC and _HAS_NAVSIM_NMC

PACKABLE_FORMAT = "navsim_metric_cache_packable"
logger = logging.getLogger(__name__)


def _require_nmc_deps() -> None:
    """Raise if nuplan/navsim stack for this module is unavailable."""
    if (
        _HAS_NUPLAN_NMC
        and _HAS_NAVSIM_NMC
        and _VEHICLE_PARAMS is not None
        and get_pacifica_parameters is not None
    ):
        return
    chain: Optional[BaseException] = (
        _NAVSIM_NMC_IMPORT_ERROR or _NUPLAN_NMC_IMPORT_ERROR
    )
    msg = (
        "heal.data.datasets.navsim.navsim_metric_cache_dataset: "
        "this operation requires `nuplan` and `navsim` "
        "(with planning/scenario modules). "
        "Install them in this environment."
    )
    raise RuntimeError(msg) from chain


def ego_state_to_array(es: EgoState) -> np.ndarray:
    """Encode an ego state into a flat float64 array.

    Args:
        es: Ego state to serialize.

    Returns:
        Array of ego state fields ordered as time, pose, velocity,
        acceleration and steering angle.
    """
    _require_nmc_deps()
    return np.array(list(es), dtype=np.float64)


def ego_state_from_array(a: np.ndarray) -> EgoState:
    """Rebuild an ego state from its flat array encoding.

    Args:
        a: Array produced by :func:`ego_state_to_array`.

    Returns:
        Ego state built from the rear axle parametrization.
    """
    _require_nmc_deps()
    time_us, x, y, heading, vx, vy, ax, ay, steering = a
    return EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(float(x), float(y), float(heading)),
        rear_axle_velocity_2d=StateVector2D(float(vx), float(vy)),
        rear_axle_acceleration_2d=StateVector2D(float(ax), float(ay)),
        tire_steering_angle=float(steering),
        time_point=TimePoint(int(time_us)),
        vehicle_parameters=_VEHICLE_PARAMS,
    )


def interpolated_trajectory_to_array(
    traj: InterpolatedTrajectory,
) -> np.ndarray:
    """Encode a sampled trajectory into a 2D float64 array.

    Args:
        traj: Trajectory whose sampled ego states are serialized.

    Returns:
        Array of ego state rows, one per sampled pose.
    """
    _require_nmc_deps()
    return np.array(
        [list(es) for es in traj.get_sampled_trajectory()], dtype=np.float64
    )


def interpolated_trajectory_from_array(
    a: np.ndarray,
) -> InterpolatedTrajectory:
    """Rebuild a trajectory from its 2D array encoding.

    Args:
        a: Array produced by :func:`interpolated_trajectory_to_array`.

    Returns:
        Trajectory interpolating the decoded ego states.
    """
    _require_nmc_deps()
    return InterpolatedTrajectory([ego_state_from_array(row) for row in a])


def _geometries_to_wkb(geoms: np.ndarray) -> list:
    """Serialize shapely geometries to a list of WKB byte strings."""
    return [shapely.wkb.dumps(g) for g in geoms]


def _geometries_from_wkb(wkb_list: list) -> np.ndarray:
    """Deserialize an object array of shapely geometries from WKB bytes."""
    geoms = np.empty(len(wkb_list), dtype=object)
    for i, b in enumerate(wkb_list):
        geoms[i] = shapely.wkb.loads(b)
    return geoms


def occupancy_map_to_dict(omap) -> dict:
    """Encode an occupancy map into a msgpack-safe dict.

    Args:
        omap: PDM occupancy map instance to serialize.

    Returns:
        Dict with tokens, WKB-encoded geometries and node capacity.
    """
    return {
        "tokens": omap._tokens,
        "geometries_wkb": _geometries_to_wkb(omap._geometries),
        "node_capacity": omap._node_capacity,
    }


def occupancy_map_from_dict(d: dict, cls):
    """Rebuild an occupancy map from its dict encoding.

    Args:
        d: Dict produced by :func:`occupancy_map_to_dict`.
        cls: Occupancy map class to instantiate.

    Returns:
        Occupancy map instance populated with decoded entries.
    """
    return cls(
        tokens=d["tokens"],
        geometries=_geometries_from_wkb(d["geometries_wkb"]),
        node_capacity=d["node_capacity"],
    )


def drivable_map_to_dict(dmap) -> dict:
    """Encode a drivable-area map into a msgpack-safe dict.

    Args:
        dmap: PDM drivable map instance to serialize.

    Returns:
        Occupancy-map dict extended with integer map types.
    """
    d = occupancy_map_to_dict(dmap)
    d["map_types"] = [int(mt) for mt in dmap._map_types]
    return d


def drivable_map_from_dict(d: dict, cls):
    """Rebuild a drivable-area map from its dict encoding.

    Args:
        d: Dict produced by :func:`drivable_map_to_dict`.
        cls: Drivable map class to instantiate.

    Returns:
        Drivable map instance populated with decoded entries.
    """
    _require_nmc_deps()
    return cls(
        tokens=d["tokens"],
        map_types=[SemanticMapLayer(v) for v in d["map_types"]],
        geometries=_geometries_from_wkb(d["geometries_wkb"]),
        node_capacity=d["node_capacity"],
    )


def _tracked_object_to_dict(obj) -> dict:
    """Encode a tracked object into a msgpack-safe dict.

    Args:
        obj: Agent or static object to serialize.

    Returns:
        Dict of box, metadata and, for agents, velocity fields.
    """
    _require_nmc_deps()
    d = {
        "type": int(obj.tracked_object_type),
        "x": obj.center.x,
        "y": obj.center.y,
        "heading": obj.center.heading,
        "length": obj.box.length,
        "width": obj.box.width,
        "height": obj.box.height,
        "token": obj.metadata.token,
        "track_token": obj.metadata.track_token,
        "track_id": obj.metadata.track_id,
        "timestamp_us": obj.metadata.timestamp_us,
    }
    if obj.tracked_object_type in AGENT_TYPES:
        d["vx"] = obj.velocity.x
        d["vy"] = obj.velocity.y
    return d


def _tracked_object_from_dict(d: dict):
    """Rebuild a tracked object from its dict encoding.

    Args:
        d: Dict produced by :func:`_tracked_object_to_dict`.

    Returns:
        :class:`Agent` for agent types, otherwise :class:`StaticObject`.
    """
    _require_nmc_deps()
    obj_type = TrackedObjectType(d["type"])
    box = OrientedBox(
        StateSE2(d["x"], d["y"], d["heading"]),
        d["length"],
        d["width"],
        d["height"],
    )
    metadata = SceneObjectMetadata(
        timestamp_us=d["timestamp_us"],
        token=d["token"],
        track_id=d.get("track_id"),
        track_token=d.get("track_token"),
    )
    if obj_type in AGENT_TYPES:
        return Agent(
            tracked_object_type=obj_type,
            oriented_box=box,
            velocity=StateVector2D(d["vx"], d["vy"]),
            metadata=metadata,
        )
    return StaticObject(
        tracked_object_type=obj_type, oriented_box=box, metadata=metadata
    )


def pdm_observation_to_dict(obs) -> dict:
    """Encode a PDM observation into a msgpack-safe dict.

    Args:
        obs: PDM observation instance to serialize.

    Returns:
        Dict of sampling parameters, occupancy maps and unique objects.
    """
    d = {
        "sample_interval": obs._sample_interval,
        "observation_samples": obs._observation_samples,
        "map_radius": obs._map_radius,
        "observation_sample_res": obs._observation_sample_res,
        "collided_track_ids": obs._collided_track_ids,
        "occupancy_maps": [
            occupancy_map_to_dict(m) for m in obs._occupancy_maps
        ],
    }
    occupancy_maps_tl = getattr(obs, "_occupancy_maps_tl", None)
    d["occupancy_maps_tl"] = (
        [
            [tokens, np.asarray(polygons, dtype=np.object_)]
            for tokens, polygons in occupancy_maps_tl
        ]
        if occupancy_maps_tl is not None
        else None
    )
    unique = getattr(obs, "_unique_objects", None)
    d["unique_objects"] = (
        {tok: _tracked_object_to_dict(obj) for tok, obj in unique.items()}
        if unique is not None
        else None
    )
    return d


def pdm_observation_from_dict(d: dict, obs_cls, omap_cls):
    """Rebuild a PDM observation from its dict encoding.

    Args:
        d: Dict produced by :func:`pdm_observation_to_dict`.
        obs_cls: PDM observation class to instantiate.
        omap_cls: Occupancy map class used to rebuild occupancy maps.

    Returns:
        Initialized PDM observation instance.
    """
    obs = object.__new__(obs_cls)
    obs._sample_interval = d["sample_interval"]
    obs._observation_samples = d["observation_samples"]
    obs._map_radius = d["map_radius"]
    obs._observation_sample_res = d["observation_sample_res"]
    obs._collided_track_ids = d["collided_track_ids"]
    obs._red_light_token = "red_light"
    res = obs._observation_sample_res
    obs._global_to_local_idcs = [
        idx // res for idx in range(obs._observation_samples + res)
    ]
    obs._occupancy_maps = [
        occupancy_map_from_dict(m, omap_cls) for m in d["occupancy_maps"]
    ]
    occupancy_maps_tl = d.get("occupancy_maps_tl")
    obs._occupancy_maps_tl = (
        [
            (tokens, np.asarray(polygons, dtype=np.object_))
            for tokens, polygons in occupancy_maps_tl
        ]
        if occupancy_maps_tl is not None
        else None
    )
    unique = d.get("unique_objects")
    obs._unique_objects = (
        {
            tok: _tracked_object_from_dict(obj_d)
            for tok, obj_d in unique.items()
        }
        if unique is not None
        else None
    )
    obs._detections_tracks = []
    obs._initialized = True
    return obs


def pdm_path_to_array(path) -> np.ndarray:
    """Encode a discrete PDM path into an Nx3 pose array.

    Args:
        path: PDM path whose discrete poses are serialized.

    Returns:
        Array of (x, y, heading) rows for each path state.
    """
    return np.array(
        [[s.x, s.y, s.heading] for s in path.discrete_path], dtype=np.float64
    )


def pdm_path_from_array(a: np.ndarray, pdm_path_cls):
    """Rebuild a PDM path from its Nx3 pose array.

    Args:
        a: Array produced by :func:`pdm_path_to_array`.
        pdm_path_cls: PDM path class to instantiate.

    Returns:
        PDM path instance built from the decoded poses.
    """
    _require_nmc_deps()
    return pdm_path_cls(
        [StateSE2(float(x), float(y), float(h)) for x, y, h in a]
    )


def detections_tracks_to_list(dt: DetectionsTracks) -> list:
    """Encode tracked detections into a list of msgpack-safe dicts.

    Args:
        dt: Detections-tracks container to serialize.

    Returns:
        List of per-object dicts from :func:`_tracked_object_to_dict`.
    """
    _require_nmc_deps()
    return [_tracked_object_to_dict(obj) for obj in dt.tracked_objects]


def detections_tracks_from_list(lst: list) -> DetectionsTracks:
    """Rebuild tracked detections from a list of object dicts.

    Args:
        lst: List produced by :func:`detections_tracks_to_list`.

    Returns:
        Detections-tracks container wrapping the decoded objects.
    """
    _require_nmc_deps()
    return DetectionsTracks(
        TrackedObjects([_tracked_object_from_dict(d) for d in lst])
    )


def navsim_trajectory_to_dict(traj) -> dict:
    """Encode a NAVSIM trajectory into a msgpack-safe dict.

    Args:
        traj: NAVSIM trajectory to serialize.

    Returns:
        Dict of poses and trajectory sampling parameters.
    """
    _require_nmc_deps()
    return {
        "poses": traj.poses,
        "time_horizon": traj.trajectory_sampling.time_horizon,
        "interval_length": traj.trajectory_sampling.interval_length,
    }


def navsim_trajectory_from_dict(d: dict, traj_cls):
    """Rebuild a NAVSIM trajectory from its dict encoding.

    Args:
        d: Dict produced by :func:`navsim_trajectory_to_dict`.
        traj_cls: NAVSIM trajectory class to instantiate.

    Returns:
        NAVSIM trajectory instance with decoded sampling parameters.
    """
    _require_nmc_deps()
    return traj_cls(
        poses=d["poses"],
        trajectory_sampling=TrajectorySampling(
            time_horizon=d["time_horizon"],
            interval_length=d["interval_length"],
        ),
    )


def metric_cache_to_packable(
    cache, navsim_version: str = "v1"
) -> Dict[str, Any]:
    """Encode a NAVSIM metric cache into a msgpack-safe payload.

    Args:
        cache: MetricCache instance to serialize.
        navsim_version: NAVSIM version tag selecting the field set, one of
            ``"v1"`` or ``"v2"``.

    Returns:
        Dict payload with arrays and dicts for all metric cache fields.
    """
    _require_nmc_deps()
    payload = {
        "__format__": PACKABLE_FORMAT,
        "trajectory": interpolated_trajectory_to_array(cache.trajectory),
        "ego_state": ego_state_to_array(cache.ego_state),
        "observation": pdm_observation_to_dict(cache.observation),
        "centerline": pdm_path_to_array(cache.centerline),
        "route_lane_ids": list(cache.route_lane_ids),
        "drivable_area_map": drivable_map_to_dict(cache.drivable_area_map),
    }
    if navsim_version == "v2":
        payload["file_path"] = str(cache.file_path)
        payload["log_name"] = cache.log_name
        payload["timepoint_us"] = int(cache.timepoint.time_us)
        payload["scene_type"] = int(cache.scene_type)
        payload["map_parameters"] = {
            "map_root": cache.map_parameters.map_root,
            "map_version": cache.map_parameters.map_version,
            "map_name": cache.map_parameters.map_name,
        }

        # Explicit serialization for complex v2-only fields.
        payload["human_trajectory"] = (
            navsim_trajectory_to_dict(cache.human_trajectory)
            if cache.human_trajectory is not None
            else None
        )
        payload["past_human_trajectory"] = (
            interpolated_trajectory_to_array(cache.past_human_trajectory)
            if cache.past_human_trajectory is not None
            else None
        )
        payload["past_detections_tracks"] = (
            [
                detections_tracks_to_list(dt)
                for dt in cache.past_detections_tracks
            ]
            if cache.past_detections_tracks is not None
            else None
        )
        payload["current_tracked_objects"] = (
            [
                detections_tracks_to_list(dt)
                for dt in cache.current_tracked_objects
            ]
            if cache.current_tracked_objects is not None
            else None
        )
        payload["future_tracked_objects"] = (
            [
                detections_tracks_to_list(dt)
                for dt in cache.future_tracked_objects
            ]
            if cache.future_tracked_objects is not None
            else None
        )
    return payload


def metric_cache_from_packable(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild metric cache fields from a decoded msgpack payload.

    Args:
        payload: Payload produced by :func:`metric_cache_to_packable`.

    Returns:
        Dict of reconstructed metric cache fields, including all v2-only
        fields when present in the payload.

    Raises:
        ValueError: If the payload format tag is not recognized.
    """
    _require_nmc_deps()
    from navsim.planning.simulation.planner.pdm_planner.observation import (
        pdm_observation,
        pdm_occupancy_map,
    )

    PDMObservation = pdm_observation.PDMObservation
    PDMDrivableMap = pdm_occupancy_map.PDMDrivableMap
    PDMOccupancyMap = pdm_occupancy_map.PDMOccupancyMap
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import (
        PDMPath,
    )

    if payload.get("__format__") != PACKABLE_FORMAT:
        raise ValueError(
            f"Unsupported payload format: {payload.get('__format__')}"
        )
    cache_dict = {
        "trajectory": interpolated_trajectory_from_array(
            payload["trajectory"]
        ),
        "ego_state": ego_state_from_array(payload["ego_state"]),
        "observation": pdm_observation_from_dict(
            payload["observation"], PDMObservation, PDMOccupancyMap
        ),
        "centerline": pdm_path_from_array(payload["centerline"], PDMPath),
        "route_lane_ids": payload["route_lane_ids"],
        "drivable_area_map": drivable_map_from_dict(
            payload["drivable_area_map"], PDMDrivableMap
        ),
    }
    if "file_path" in payload:
        cache_dict["file_path"] = payload["file_path"]
    if "log_name" in payload:
        cache_dict["log_name"] = payload["log_name"]
    if payload.get("timepoint_us") is not None:
        from nuplan.common.actor_state.state_representation import TimePoint

        cache_dict["timepoint"] = TimePoint(int(payload["timepoint_us"]))
    if payload.get("scene_type") is not None:
        # SceneFrameType is navsim v2-only; keep a local IntEnum fallback
        # for v1 envs that still load v2-packed caches (v1 ignores it).
        try:
            from navsim.common.enums import SceneFrameType
        except ImportError:
            from enum import IntEnum

            class SceneFrameType(IntEnum):
                ORIGINAL = 0
                SYNTHETIC = 1

        cache_dict["scene_type"] = SceneFrameType(payload["scene_type"])
    if payload.get("map_parameters") is not None:
        # MapParameters is also v2-only on MetricCache.
        try:
            from navsim.planning.metric_caching.metric_cache import (
                MapParameters,
            )
        except ImportError:
            from dataclasses import dataclass

            @dataclass
            class MapParameters:
                map_root: str
                map_version: str
                map_name: str

        cache_dict["map_parameters"] = MapParameters(
            **payload["map_parameters"]
        )

    if payload.get("human_trajectory") is not None:
        from navsim.common.dataclasses import Trajectory as NavsimTrajectory

        cache_dict["human_trajectory"] = navsim_trajectory_from_dict(
            payload["human_trajectory"], NavsimTrajectory
        )
    else:
        cache_dict["human_trajectory"] = None

    if payload.get("past_human_trajectory") is not None:
        cache_dict["past_human_trajectory"] = (
            interpolated_trajectory_from_array(
                payload["past_human_trajectory"]
            )
        )
    else:
        cache_dict["past_human_trajectory"] = None

    for k in [
        "past_detections_tracks",
        "current_tracked_objects",
        "future_tracked_objects",
    ]:
        if payload.get(k) is None:
            cache_dict[k] = None
        else:
            cache_dict[k] = [
                detections_tracks_from_list(dt) for dt in payload[k]
            ]
    return cache_dict


class NavsimMetricCachePacker(Packer):
    """Packer that precomputes NAVSIM metric caches for dataset tokens.

    It computes the PDM metric cache for every scene of a NAVSIM split and
    writes the serialized results into a pack file.

    Args:
        data_path: Path to the NAVSIM dataset root.
        target_data_dir: Output directory for the packed metric caches.
        split_name: Name of the NAVSIM split to pack.
        num_workers: Number of worker processes for packing.
        pack_type: Pack file format, default ``"raw"``.
        map_root: Optional nuPlan maps root path.
        split_cfg_root: Optional override path to split config root.
        max_samples: Optional cap on number of scenes to pack.
        **kwargs: Extra keyword arguments forwarded to :class:`Packer`.
    """

    def __init__(
        self,
        data_path: str,
        target_data_dir: str,
        split_name: str,
        num_workers: int = 20,
        pack_type: str = "raw",
        map_root: str = None,
        split_cfg_root: str = None,
        max_samples: Optional[int] = None,
        **kwargs: Any,
    ):
        _require_nmc_deps()
        # Use importlib (not ``from ... import``) so the circular-import
        # checker does not record an edge back to navsim_dataset / package.
        # navsim_dataset already imports this module at top level.
        import importlib

        _ns = importlib.import_module(
            "heal.data.datasets.navsim.navsim_dataset"
        )

        try:
            self.navsim_pkg_version = version("navsim")
        except Exception:
            self.navsim_pkg_version = "0"
        major_version = self.navsim_pkg_version.split(".")[0]
        self.navsim_version = (
            "v2"
            if major_version.isdigit() and int(major_version) >= 2
            else "v1"
        )
        if map_root is None:
            map_root = os.environ.get("NUPLAN_MAPS_ROOT")
        if map_root is None:
            raise ValueError(
                "Must provide map_root or set NUPLAN_MAPS_ROOT env variable!"
            )
        self.map_root = map_root
        _ns._sync_nuplan_maps_root(self.map_root)

        self.dataset = _ns.NavsimDataset(
            data_path=data_path,
            split=split_name,
            pack_type="lmdb",
            sensor_config=None,
            split_cfg_root=split_cfg_root,
        )
        self.tokens = self.dataset.tokens
        if max_samples is not None:
            self.dataset = self.dataset[:max_samples]
            self.tokens = self.tokens[:max_samples]

        num_samples = len(self.dataset)
        logger.info(f"Total {num_samples} scenes to pack metric cache.")
        self.dummy_cache_path = Path("dummy_cache")
        self.packer_meta = {
            "token_to_idx": {
                token: idx for idx, token in enumerate(self.tokens)
            }
        }
        proposal_sampling = TrajectorySampling(
            num_poses=40, interval_length=0.1
        )
        from navsim.planning.metric_caching.metric_cache_processor import (
            MetricCacheProcessor,
        )

        processor_kwargs = {}
        if (
            "proposal_sampling"
            in inspect.signature(MetricCacheProcessor.__init__).parameters
        ):
            processor_kwargs["proposal_sampling"] = proposal_sampling
        self.processor = MetricCacheProcessor(
            cache_path=self.dummy_cache_path,
            force_feature_computation=True,
            **processor_kwargs,
        )
        super().__init__(
            target_data_dir, num_samples, pack_type, num_workers, **kwargs
        )

    def _write_length(self, num):
        """Write pack metadata and sample count into the pack file.

        Args:
            num: Number of packed samples to record.
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
                    "%s is not supported for write length "
                    "(string key limitation).",
                    self.target_pack_type,
                )

    def pack_data(self, idx):
        """Compute and serialize the metric cache for one dataset index.

        Args:
            idx: Index of the scene to pack.

        Returns:
            Msgpack-serialized metric cache payload.
        """
        _require_nmc_deps()
        data = self.dataset[idx]
        scene = data["scene"]
        scenario = NavSimScenario(
            scene, map_root=self.map_root, map_version="nuplan-maps-v1.0"
        )
        metric_cache = self.processor.compute_metric_cache(scenario)
        if hasattr(metric_cache, "file_name"):
            import lzma

            with lzma.open(metric_cache.file_name, "rb") as f:
                metric_cache = pickle.loads(f.read())
        cache_data = metric_cache_to_packable(
            metric_cache, navsim_version=self.navsim_version
        )
        serialized_data = msgpack.packb(
            cache_data, default=msgpack_numpy.encode
        )
        return serialized_data


class NavsimMetricCacheLoader:
    """Loader that reads precomputed NAVSIM metric caches from a pack file.

    Args:
        pack_path: Path to the lmdb pack file produced by
            :class:`NavsimMetricCachePacker`.
        navsim_version: NAVSIM version tag of the packed caches, one of
            ``"v1"`` or ``"v2"``.
    """

    def __init__(self, pack_path: str, navsim_version: str = "v1"):
        self.pack_path = pack_path
        self.pack_file = Lmdb(pack_path, writable=False)
        self.pack_file.open()
        meta_bytes = self.pack_file.read("__meta__")
        self.meta = msgpack.unpackb(meta_bytes, raw=False)
        self.token_to_idx = self.meta["token_to_idx"]
        self.version = navsim_version

    def __call__(self, token: str) -> Dict[str, Any]:
        """Load and decode the metric cache payload for one scene token.

        Args:
            token: Scene token identifying the metric cache entry.

        Returns:
            Decoded metric cache payload dict.

        Raises:
            ValueError: If the cached bytes fail msgpack decoding.
        """
        idx = self.token_to_idx[token]
        cache_bytes = self.pack_file.read(idx)
        try:
            metric_cache = msgpack.unpackb(
                cache_bytes, object_hook=msgpack_numpy.decode, raw=False
            )
        except Exception as e:
            raise ValueError(
                "Failed to decode metric cache with msgpack for "
                f"token={token}, idx={idx}, "
                f"pack_path={self.pack_path}: {e}"
            ) from e
        return metric_cache
