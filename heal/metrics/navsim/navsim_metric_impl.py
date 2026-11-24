# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from __future__ import annotations
import logging
import os
import pickle
from collections.abc import Sequence
from dataclasses import fields
from datetime import datetime
from importlib.metadata import version
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from heal.data.datasets.navsim.navsim_metric_cache_dataset import (
    PACKABLE_FORMAT,
    NavsimMetricCacheLoader,
    metric_cache_from_packable,
)

_HAS_NUPLAN_NMI = False
_HAS_NAVSIM_NMI = False
_NUPLAN_NMI_ERR: Optional[BaseException] = None
_NAVSIM_NMI_ERR: Optional[BaseException] = None

StateSE2 = Any
relative_to_absolute_poses = None  # type: ignore[assignment]
TrajectorySampling = Any
PDMResults = Any
Trajectory = Any
PDMScorer = Any
PDMSimulator = Any

try:
    from nuplan.common.actor_state.state_representation import (
        StateSE2 as _StateSE2,
    )
    from nuplan.common.geometry.convert import (
        relative_to_absolute_poses as _rta,
    )
    from nuplan.planning.simulation.trajectory.trajectory_sampling import (
        TrajectorySampling as _TrajectorySampling,
    )

    StateSE2 = _StateSE2
    relative_to_absolute_poses = _rta
    TrajectorySampling = _TrajectorySampling
    _HAS_NUPLAN_NMI = True
except ImportError as exc:
    _NUPLAN_NMI_ERR = exc
    StateSE2 = Any
    relative_to_absolute_poses = None  # type: ignore[assignment]
    TrajectorySampling = Any

try:
    from navsim.common.dataclasses import PDMResults as _PDMResults
    from navsim.common.dataclasses import Trajectory as _Trajectory
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (  # noqa: E501
        PDMScorer as _PDMScorer,
    )
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (  # noqa: E501
        PDMSimulator as _PDMSimulator,
    )

    PDMResults = _PDMResults
    Trajectory = _Trajectory
    PDMScorer = _PDMScorer
    PDMSimulator = _PDMSimulator
    _HAS_NAVSIM_NMI = True
except ImportError as exc:
    _NAVSIM_NMI_ERR = exc
    PDMResults = Any
    Trajectory = Any
    PDMScorer = Any
    PDMSimulator = Any

_navsim_pkg_version = "0"
try:
    _navsim_pkg_version = version("navsim")
except Exception:
    pass
_navsim_major_version = _navsim_pkg_version.split(".")[0]
NAVSIM_VERSION = (
    "v2"
    if _navsim_major_version.isdigit() and int(_navsim_major_version) >= 2
    else "v1"
)

logger = logging.getLogger(__name__)

_EXTENDED_PDM_AVAILABLE = False
SceneFrameType = Any
PDMScorerConfig = Any
NavsimIDMAgents = Any
NavsimIDMTrafficAgents = Any

if NAVSIM_VERSION == "v2" and _HAS_NAVSIM_NMI:
    try:
        from navsim.common.enums import SceneFrameType as _SceneFrameType
        from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (  # noqa: E501
            PDMScorerConfig as _PDMScorerConfig,
        )
        from navsim.traffic_agents_policies.navsim_IDM_traffic_agents import (
            NavsimIDMAgents as _NavsimIDMAgents,
        )
        from navsim.traffic_agents_policies.navsim_IDM_traffic_agents import (
            NavsimIDMTrafficAgents as _NavsimIDMTrafficAgents,
        )

        SceneFrameType = _SceneFrameType
        PDMScorerConfig = _PDMScorerConfig
        NavsimIDMAgents = _NavsimIDMAgents
        NavsimIDMTrafficAgents = _NavsimIDMTrafficAgents
        _EXTENDED_PDM_AVAILABLE = True
    except ImportError as exc:
        logger.warning(
            "Optional NavSim v2 PDM extension imports failed (%s); "
            "v2-only helpers will raise if used.",
            exc,
        )


def _require_metric_impl_core_deps() -> None:
    """Raise at runtime if nuplan + navsim (core metric stack) are unavailable."""  # noqa: E501
    if (
        _HAS_NUPLAN_NMI
        and _HAS_NAVSIM_NMI
        and relative_to_absolute_poses is not None
    ):
        return
    msg = (
        "heal.metrics.navsim.navsim_metric_impl: this operation requires `nuplan` "  # noqa: E501
        "and `navsim`."
    )
    raise RuntimeError(msg) from (_NAVSIM_NMI_ERR or _NUPLAN_NMI_ERR)


def _load_metric_cache_dict(payload: Any) -> Dict[str, Any]:
    _require_metric_impl_core_deps()
    if (
        isinstance(payload, dict)
        and payload.get("__format__") == PACKABLE_FORMAT
    ):
        return metric_cache_from_packable(payload)

    raise ValueError(
        "Unsupported metric cache payload format: expected packable payload "
        f"with __format__={PACKABLE_FORMAT!r}."
    )


def _require_extended_pdm(feature: str) -> None:
    _require_metric_impl_core_deps()
    if not _EXTENDED_PDM_AVAILABLE:
        raise RuntimeError(
            f"{feature} requires navsim major version >= 2 (extended PDM / MetricCache). "  # noqa: E501
            f"Detected NAVSIM_VERSION={NAVSIM_VERSION!r} ({_navsim_pkg_version!r})."  # noqa: E501
        )


def transform_trajectory(
    pred_trajectory: Trajectory, initial_ego_state: Any
) -> Any:
    _require_metric_impl_core_deps()
    from nuplan.planning.simulation.planner.ml_planner.transform_utils import (
        _get_fixed_timesteps,
        _se2_vel_acc_to_ego_state,
    )
    from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
        InterpolatedTrajectory,
    )

    future_sampling = pred_trajectory.trajectory_sampling
    timesteps = _get_fixed_timesteps(
        initial_ego_state,
        future_sampling.time_horizon,
        future_sampling.interval_length,
    )

    relative_poses = np.array(pred_trajectory.poses, dtype=np.float64)
    relative_states = [StateSE2.deserialize(pose) for pose in relative_poses]
    absolute_states = relative_to_absolute_poses(
        initial_ego_state.rear_axle, relative_states
    )

    agent_states = [
        _se2_vel_acc_to_ego_state(
            state,
            [0.0, 0.0],
            [0.0, 0.0],
            timestep,
            initial_ego_state.car_footprint.vehicle_parameters,
        )
        for state, timestep in zip(absolute_states, timesteps)
    ]

    return InterpolatedTrajectory([initial_ego_state] + agent_states)


def get_trajectory_as_array(
    trajectory: Any,
    future_sampling: TrajectorySampling,
    start_time: Any,
) -> np.ndarray:
    _require_metric_impl_core_deps()
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (  # noqa: E501
        ego_states_to_state_array,
    )
    from nuplan.common.actor_state.state_representation import TimePoint

    times_s = np.arange(
        0.0,
        future_sampling.time_horizon + future_sampling.interval_length,
        future_sampling.interval_length,
    )
    times_s += start_time.time_s
    times_us = [int(time_s * 1e6) for time_s in times_s]
    times_us = np.clip(
        times_us, trajectory.start_time.time_us, trajectory.end_time.time_us
    )
    time_points = [TimePoint(time_us) for time_us in times_us]

    trajectory_ego_states = trajectory.get_state_at_times(time_points)
    return ego_states_to_state_array(trajectory_ego_states)


def pdm_score_v1(
    metric_cache: Dict[str, Any],
    model_trajectory: Trajectory,
    future_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    scorer: PDMScorer,
) -> Dict[str, Any]:
    _require_metric_impl_core_deps()
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (  # noqa: E501
        MultiMetricIndex,
        WeightedMetricIndex,
    )

    initial_ego_state = metric_cache["ego_state"]
    pdm_trajectory = metric_cache["trajectory"]
    pred_trajectory = transform_trajectory(model_trajectory, initial_ego_state)
    pdm_states, pred_states = (
        get_trajectory_as_array(
            pdm_trajectory, future_sampling, initial_ego_state.time_point
        ),
        get_trajectory_as_array(
            pred_trajectory, future_sampling, initial_ego_state.time_point
        ),
    )
    trajectory_states = np.concatenate(
        [pdm_states[None, ...], pred_states[None, ...]], axis=0
    )
    simulated_states = simulator.simulate_proposals(
        trajectory_states, initial_ego_state
    )

    scores = scorer.score_proposals(
        simulated_states,
        metric_cache["observation"],
        metric_cache["centerline"],
        metric_cache["route_lane_ids"],
        metric_cache["drivable_area_map"],
    )
    pred_idx = 1
    score = float(scores[pred_idx])

    no_at_fault_collisions = scorer._multi_metrics[
        MultiMetricIndex.NO_COLLISION, pred_idx
    ]
    drivable_area_compliance = scorer._multi_metrics[
        MultiMetricIndex.DRIVABLE_AREA, pred_idx
    ]
    ego_progress = scorer._weighted_metrics[
        WeightedMetricIndex.PROGRESS, pred_idx
    ]
    time_to_collision_within_bound = scorer._weighted_metrics[
        WeightedMetricIndex.TTC, pred_idx
    ]
    comfort = scorer._weighted_metrics[
        WeightedMetricIndex.COMFORTABLE, pred_idx
    ]
    driving_direction_compliance = scorer._weighted_metrics[
        WeightedMetricIndex.DRIVING_DIRECTION, pred_idx
    ]
    return {
        "no_at_fault_collisions": no_at_fault_collisions,
        "drivable_area_compliance": drivable_area_compliance,
        "ego_progress": ego_progress,
        "time_to_collision_within_bound": time_to_collision_within_bound,
        "comfort": comfort,
        "driving_direction_compliance": driving_direction_compliance,
        "pdm_score": score,
    }


def transform_trajectory_pdm(
    pred_trajectory: Trajectory, initial_ego_state: Any
) -> Any:
    """Convert a relative predicted trajectory to an absolute ego trajectory.

    Args:
        pred_trajectory: Predicted trajectory with ego-relative poses.
        initial_ego_state: Initial ego state used as the reference frame.

    Returns:
        Interpolated trajectory in absolute coordinates including the initial state.
    """
    from nuplan.planning.simulation.planner.ml_planner.transform_utils import (
        _get_fixed_timesteps,
        _se2_vel_acc_to_ego_state,
    )
    from nuplan.planning.simulation.trajectory.interpolated_trajectory import (
        InterpolatedTrajectory,
    )

    _require_extended_pdm("transform_trajectory_pdm")
    future_sampling = pred_trajectory.trajectory_sampling
    timesteps = _get_fixed_timesteps(
        initial_ego_state,
        future_sampling.time_horizon,
        future_sampling.interval_length,
    )

    relative_poses = np.array(pred_trajectory.poses, dtype=np.float64)
    relative_states = [StateSE2.deserialize(pose) for pose in relative_poses]
    absolute_states = relative_to_absolute_poses(
        initial_ego_state.rear_axle, relative_states
    )

    agent_states = [
        _se2_vel_acc_to_ego_state(
            state,
            [0.0, 0.0],
            [0.0, 0.0],
            timestep,
            initial_ego_state.car_footprint.vehicle_parameters,
        )
        for state, timestep in zip(absolute_states, timesteps)
    ]

    return InterpolatedTrajectory([initial_ego_state] + agent_states)


def get_trajectory_as_array_pdm(
    trajectory: Any,
    future_sampling: TrajectorySampling,
    start_time: Any,
) -> np.ndarray:
    """Sample a trajectory at fixed timesteps and return it as a state array.

    Args:
        trajectory: Interpolated trajectory to sample.
        future_sampling: Sampling spec with time horizon and interval.
        start_time: Time point at which sampling starts.

    Returns:
        State array with one row per sampled timestep.
    """
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (  # noqa: E501
        ego_states_to_state_array,
    )
    from nuplan.common.actor_state.state_representation import TimePoint

    _require_extended_pdm("get_trajectory_as_array_pdm")
    times_s = np.arange(
        0.0,
        future_sampling.time_horizon + future_sampling.interval_length,
        future_sampling.interval_length,
    )
    times_s += start_time.time_s
    times_us = [int(time_s * 1e6) for time_s in times_s]
    times_us = np.clip(
        times_us, trajectory.start_time.time_us, trajectory.end_time.time_us
    )
    time_points = [TimePoint(time_us) for time_us in times_us]

    trajectory_ego_states = trajectory.get_state_at_times(time_points)
    return ego_states_to_state_array(trajectory_ego_states)


def pdm_score_v2(
    metric_cache: Dict[str, Any],
    model_trajectory: Trajectory,
    future_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    scorer: PDMScorer,
    traffic_agents_policy: Any,
    visualize: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Compute the navsim v2 extended PDM score for one predicted trajectory.

    Args:
        metric_cache: Metric cache dict or ``MetricCache`` for the scene token.
        model_trajectory: Predicted trajectory with ego-relative poses.
        future_sampling: Sampling spec with time horizon and interval.
        simulator: PDM simulator instance.
        scorer: PDM scorer instance.
        traffic_agents_policy: Reactive traffic agents policy.
        visualize: Whether to dump a visualization gif of the simulation.

    Returns:
        Tuple of the per-proposal score dataframe and the simulated ego states.
    """
    from navsim.planning.metric_caching.metric_cache import MetricCache

    _require_extended_pdm("pdm_score_v2")
    if (
        isinstance(metric_cache, dict)
        and metric_cache.get("__format__") == PACKABLE_FORMAT
    ):
        metric_cache = metric_cache_from_packable(metric_cache)
    metric_cache_obj = (
        MetricCache(**metric_cache)
        if isinstance(metric_cache, dict)
        else metric_cache
    )
    pred_trajectory = transform_trajectory_pdm(
        model_trajectory, metric_cache_obj.ego_state
    )
    return pdm_score_from_interpolated_trajectory(
        metric_cache=metric_cache_obj,
        pred_trajectory=pred_trajectory,
        future_sampling=future_sampling,
        simulator=simulator,
        scorer=scorer,
        traffic_agents_policy=traffic_agents_policy,
        visualize=visualize,
    )


def pdm_score_from_interpolated_trajectory(
    metric_cache: Any,
    pred_trajectory: Any,
    future_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    scorer: PDMScorer,
    traffic_agents_policy: Any,
    visualize: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Score an already-interpolated predicted trajectory with the v2 PDM stack.

    Args:
        metric_cache: ``MetricCache`` for the scene token.
        pred_trajectory: Interpolated predicted ego trajectory.
        future_sampling: Sampling spec with time horizon and interval.
        simulator: PDM simulator instance.
        scorer: PDM scorer instance.
        traffic_agents_policy: Reactive traffic agents policy.
        visualize: Whether to dump a visualization gif of the simulation.

    Returns:
        Tuple of the per-proposal score dataframe and the simulated ego states.
    """
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (  # noqa: E501
        WeightedMetricIndex,
    )

    _require_extended_pdm("pdm_score_from_interpolated_trajectory")
    initial_ego_state = metric_cache.ego_state
    pdm_trajectory = metric_cache.trajectory

    pdm_states, pred_states = (
        get_trajectory_as_array_pdm(
            pdm_trajectory, future_sampling, initial_ego_state.time_point
        ),
        get_trajectory_as_array_pdm(
            pred_trajectory, future_sampling, initial_ego_state.time_point
        ),
    )
    trajectory_states = np.concatenate(
        [pdm_states[None, ...], pred_states[None, ...]], axis=0
    )
    simulated_states = simulator.simulate_proposals(
        trajectory_states, initial_ego_state
    )

    simulated_agent_detections_tracks = (
        traffic_agents_policy.simulate_environment(
            simulated_states[1], metric_cache
        )
    )
    assert (
        len(simulated_agent_detections_tracks) == trajectory_states.shape[1]
    ), f"""
            Traffic agents policy returned trajectories of invalid length:
Traffic agents trajectories must be of length ego_trajectory_length =
            {trajectory_states.shape[1]},
            but got {len(simulated_agent_detections_tracks)}
        """

    pred_idx = 1
    pdm_result = scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
        metric_cache.map_parameters,
        simulated_agent_detections_tracks,
        metric_cache.past_human_trajectory,
    )[pred_idx]

    if (
        scorer._config.human_penalty_filter
        and metric_cache.scene_type == SceneFrameType.ORIGINAL
    ):
        human_trajectory = transform_trajectory_pdm(
            metric_cache.human_trajectory, initial_ego_state
        )
        human_states = get_trajectory_as_array_pdm(
            human_trajectory, future_sampling, initial_ego_state.time_point
        )
        human_simulated_states = simulator.simulate_proposals(
            human_states[None, ...], initial_ego_state
        )
        human_simulated_agent_detections_tracks = (
            traffic_agents_policy.simulate_environment(
                human_simulated_states[0], metric_cache
            )
        )
        human_pdm_result = scorer.score_proposals(
            human_simulated_states,
            metric_cache.observation,
            metric_cache.centerline,
            metric_cache.route_lane_ids,
            metric_cache.drivable_area_map,
            metric_cache.map_parameters,
            human_simulated_agent_detections_tracks,
        )[0]

        skip_columns = {
            "multiplicative_metrics_prod",
            "weighted_metrics",
            "weighted_metrics_array",
            "pdm_score",
        }

        modified_any = False
        for column in human_pdm_result.columns:
            if (
                column not in skip_columns
                and human_pdm_result[column].iloc[0] == 0
            ):
                pdm_result.at[0, column] = 1
                modified_any = True

        if modified_any:
            pdm_result.at[0, "multiplicative_metrics_prod"] = (
                pdm_result.at[0, "no_at_fault_collisions"]
                * pdm_result.at[0, "drivable_area_compliance"]
                * pdm_result.at[0, "driving_direction_compliance"]
                * pdm_result.at[0, "traffic_light_compliance"]
            )

            weighted_metrics = pdm_result.at[0, "weighted_metrics"].copy()
            weighted_metrics[WeightedMetricIndex.PROGRESS] = pdm_result.at[
                0, "ego_progress"
            ]
            weighted_metrics[WeightedMetricIndex.TTC] = pdm_result.at[
                0, "time_to_collision_within_bound"
            ]
            weighted_metrics[WeightedMetricIndex.LANE_KEEPING] = pdm_result.at[
                0, "lane_keeping"
            ]
            weighted_metrics[WeightedMetricIndex.HISTORY_COMFORT] = (
                pdm_result.at[0, "history_comfort"]
            )
            pdm_result.at[0, "weighted_metrics"] = weighted_metrics

    if visualize:
        from nuplan.common.maps.nuplan_map.map_factory import get_maps_api

        from heal.visualize.navsim_vis.vis_help import visualize_stage2_gif

        map_api_obj = get_maps_api(
            os.getenv("NUPLAN_MAPS_ROOT", None),
            metric_cache.map_parameters.map_version,
            metric_cache.map_parameters.map_name,
        )
        ego_poses_array = simulated_states[1]
        relative_poses = [
            StateSE2(float(pose[0]), float(pose[1]), float(pose[2]))
            for pose in ego_poses_array
        ]
        absolute_states = relative_to_absolute_poses(
            initial_ego_state.rear_axle,
            relative_poses,
        )
        simulated_states_global = np.array(
            [[s.x, s.y, s.heading] for s in absolute_states]
        )

        visualize_stage2_gif(
            token=metric_cache.log_name,
            map_api=map_api_obj,
            ego_traj_global=simulated_states_global,
            agent_tracks=simulated_agent_detections_tracks,
            save_dir="./vis_debug_output",
        )

    return pdm_result, simulated_states[pred_idx]


def compute_final_scores_two_stage(pdm_score_df: pd.DataFrame) -> pd.DataFrame:
    """Combine per-frame sub-metrics into the final two-stage score column.

    Args:
        pdm_score_df: Per-frame score dataframe with aggregated comfort values.

    Returns:
        Dataframe with a ``score`` column and the raw metric columns dropped.
    """
    _require_metric_impl_core_deps()
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (  # noqa: E501
        WeightedMetricIndex,
    )

    df = pdm_score_df.reset_index()
    assert (
        not df["two_frame_extended_comfort"].isna().any()
    ), "Found NaN in 'two_frame_extended_comfort'. Please check aggregator completeness."  # noqa: E501

    two_frame_scores = df["two_frame_extended_comfort"].to_numpy()
    weighted_metrics = np.stack(df["weighted_metrics"].to_numpy())
    weighted_metrics_array = np.stack(df["weighted_metrics_array"].to_numpy())

    two_frame_idx = WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT
    weighted_metrics[:, two_frame_idx] = two_frame_scores

    weighted_sum = (weighted_metrics * weighted_metrics_array).sum(axis=1)
    total_weight = weighted_metrics_array.sum(axis=1)
    assert np.all(
        total_weight > 0
    ), "Found total_weight == 0 during score computation."

    weighted_metric_scores = weighted_sum / total_weight
    df["score"] = (
        df["multiplicative_metrics_prod"].to_numpy() * weighted_metric_scores
    )
    df.drop(
        columns=[
            "weighted_metrics",
            "weighted_metrics_array",
            "multiplicative_metrics_prod",
        ],
        inplace=True,
    )
    return df


def compute_final_scores_one_stage(pdm_score_df: pd.DataFrame) -> pd.DataFrame:
    """Combine per-frame sub-metrics into the final one-stage score column.

    Args:
        pdm_score_df: Per-frame score dataframe with aggregated comfort values.

    Returns:
        Dataframe with a ``score`` column and the raw metric columns dropped.
    """
    _require_metric_impl_core_deps()
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (  # noqa: E501
        WeightedMetricIndex,
    )

    df = pdm_score_df.copy()
    two_frame_scores = df["two_frame_extended_comfort"].to_numpy()
    weighted_metrics = np.stack(df["weighted_metrics"].to_numpy())
    weighted_metrics_array = np.stack(df["weighted_metrics_array"].to_numpy())

    mask = np.isnan(two_frame_scores)
    two_frame_idx = WeightedMetricIndex.TWO_FRAME_EXTENDED_COMFORT

    weighted_metrics[mask, two_frame_idx] = 0.0
    weighted_metrics_array[mask, two_frame_idx] = 0.0
    non_mask = ~mask
    weighted_metrics[non_mask, two_frame_idx] = two_frame_scores[non_mask]

    weighted_sum = (weighted_metrics * weighted_metrics_array).sum(axis=1)
    total_weight = weighted_metrics_array.sum(axis=1)
    total_weight[total_weight == 0.0] = np.nan
    weighted_metric_scores = weighted_sum / total_weight

    df["score"] = (
        df["multiplicative_metrics_prod"].to_numpy() * weighted_metric_scores
    )
    df.drop(
        columns=[
            "weighted_metrics",
            "weighted_metrics_array",
            "multiplicative_metrics_prod",
        ],
        inplace=True,
    )
    return df


def create_scene_aggregators_one_stage(
    all_mappings: Dict[str, str],
    full_score_df: pd.DataFrame,
    proposal_sampling: TrajectorySampling,
) -> pd.DataFrame:
    """Aggregate per-frame scores across adjacent frames for one-stage eval.

    Args:
        all_mappings: Mapping of current frame token to previous frame token.
        full_score_df: Per-frame score dataframe indexed by ``token``.
        proposal_sampling: Sampling spec used to evaluate comfort windows.

    Returns:
        Score dataframe updated with two-frame extended comfort values.
    """
    _require_metric_impl_core_deps()
    from navsim.planning.simulation.planner.pdm_planner.scoring.scene_aggregator import (  # noqa: E501
        SceneAggregator,
    )

    full_score_df["two_frame_extended_comfort"] = np.nan
    full_score_df = full_score_df.set_index("token")

    all_updates = []
    for now_frame, previous_frame in all_mappings.items():
        aggregator = SceneAggregator(
            now_frame=now_frame,
            previous_frame=previous_frame,
            score_df=full_score_df,
            proposal_sampling=proposal_sampling,
        )
        updated_rows = aggregator.aggregate_scores(one_stage_only=True)
        all_updates.append(updated_rows)
    all_updates_df = pd.concat(all_updates, ignore_index=True).set_index(
        "token"
    )
    full_score_df.update(all_updates_df)
    full_score_df.reset_index(inplace=True)
    full_score_df = full_score_df.drop(columns=["ego_simulated_states"])
    return full_score_df


def infer_start_adjacent_mapping(
    score_df: pd.DataFrame, time_gap_threshold: float = 0.55
) -> Dict[str, str]:
    """Infer current-to-previous frame token pairs from per-log start times.

    Args:
        score_df: Per-frame score dataframe with ``frame_type`` and ``start_time``.
        time_gap_threshold: Maximum start-time gap in seconds treating frames as adjacent.

    Returns:
        Mapping of current frame token to the previous frame token.
    """
    _require_extended_pdm("infer_start_adjacent_mapping")
    adjacent_mapping: Dict[str, str] = {}
    for _, group_df in score_df[
        score_df["frame_type"] == SceneFrameType.ORIGINAL
    ].groupby("log_name"):
        group_df = group_df.sort_values(by="start_time").reset_index(drop=True)
        for i in range(1, len(group_df)):
            prev_row = group_df.iloc[i - 1]
            current_row = group_df.iloc[i]
            prev_token = prev_row["token"]
            current_token = current_row["token"]
            time_diff = current_row["start_time"] - prev_row["start_time"]
            if abs(time_diff) <= time_gap_threshold:
                adjacent_mapping[current_token] = prev_token
    return adjacent_mapping


def calculate_weighted_average_score(df: pd.DataFrame) -> pd.Series:
    """Compute per-column weighted averages over a group of score rows.

    Args:
        df: Score rows with a ``weight`` column plus metric columns.

    Returns:
        Series of weighted-average scores indexed by metric column name.
    """
    score_cols = [c for c in df.columns if c not in {"weight", "token"}]
    if df.empty:
        return pd.Series([np.nan] * len(score_cols), index=score_cols)

    weights = df["weight"].to_numpy()
    scores = df[score_cols].to_numpy()
    total_weight = weights.sum()
    if total_weight == 0:
        return pd.Series([np.nan] * len(score_cols), index=score_cols)

    weighted_avg = (scores * weights[:, None]).sum(axis=0) / total_weight
    return pd.Series(weighted_avg, index=score_cols)


def calculate_individual_mapping_scores(
    pdm_score_df: pd.DataFrame,
    all_mappings: Dict[Tuple[str, str], List[Tuple[str, str]]],
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Aggregate two-stage group scores over all frame-mapping groups.

    Args:
        pdm_score_df: Per-frame score dataframe with ``token`` and ``weight``.
        all_mappings: Mapping of (orig, prev) token pairs to second-stage pairs.

    Returns:
        Tuple of (combined, stage-one, stage-two) per-metric mean scores.
    """
    all_group_scores = []
    stage1_group_scores = []
    stage2_group_scores = []

    for (orig_token, prev_token), second_stage_pairs in all_mappings.items():
        first_tokens = [
            pair[0] for pair in second_stage_pairs if len(pair) > 0
        ]
        second_tokens = [
            pair[1] for pair in second_stage_pairs if len(pair) > 1
        ]

        group1_stage1_df = pdm_score_df[pdm_score_df["token"] == orig_token]
        group1_stage2_df = pdm_score_df[
            pdm_score_df["token"].isin(first_tokens)
        ]
        group2_stage1_df = pdm_score_df[pdm_score_df["token"] == prev_token]
        group2_stage2_df = pdm_score_df[
            pdm_score_df["token"].isin(second_tokens)
        ]

        group1_stage1_scores = calculate_weighted_average_score(
            group1_stage1_df
        )
        group1_stage2_scores = calculate_weighted_average_score(
            group1_stage2_df
        )
        group2_stage1_scores = calculate_weighted_average_score(
            group2_stage1_df
        )
        group2_stage2_scores = calculate_weighted_average_score(
            group2_stage2_df
        )

        stage1_group_scores.append(group1_stage1_scores)
        stage1_group_scores.append(group2_stage1_scores)
        stage2_group_scores.append(group1_stage2_scores)
        stage2_group_scores.append(group2_stage2_scores)

        group1_scores = group1_stage1_scores * group1_stage2_scores
        group2_scores = group2_stage1_scores * group2_stage2_scores
        all_group_scores.append((group1_scores + group2_scores) / 2)

    return (
        pd.DataFrame(all_group_scores).mean(),
        pd.DataFrame(stage1_group_scores).mean(),
        pd.DataFrame(stage2_group_scores).mean(),
    )


def create_scene_aggregators_two_stage(
    all_mappings: Dict[Tuple[str, str], List[Tuple[str, str]]],
    full_score_df: pd.DataFrame,
    proposal_sampling: TrajectorySampling,
) -> pd.DataFrame:
    """Aggregate per-frame scores across two-stage frame-mapping groups.

    Args:
        all_mappings: Mapping of (orig, prev) token pairs to second-stage pairs.
        full_score_df: Per-frame score dataframe indexed by ``token``.
        proposal_sampling: Sampling spec used to evaluate comfort windows.

    Returns:
        Score dataframe updated with two-frame extended comfort and weights.
    """
    _require_extended_pdm("create_scene_aggregators_two_stage")
    from navsim.planning.simulation.planner.pdm_planner.scoring.scene_aggregator import (  # noqa: E501
        SceneAggregator,
    )

    full_score_df["two_frame_extended_comfort"] = np.nan
    full_score_df["weight"] = np.nan
    full_score_df = full_score_df.set_index("token")

    all_updates = []
    for (now_frame, previous_frame), second_stage in all_mappings.items():
        aggregator = SceneAggregator(
            now_frame=now_frame,
            previous_frame=previous_frame,
            second_stage=second_stage,
            score_df=full_score_df,
            proposal_sampling=proposal_sampling,
        )
        updated_rows = aggregator.aggregate_scores()
        all_updates.append(updated_rows)

    all_updates_df = pd.concat(all_updates, ignore_index=True).set_index(
        "token"
    )
    full_score_df.update(all_updates_df)
    full_score_df.reset_index(inplace=True)
    full_score_df = full_score_df.drop(columns=["ego_simulated_states"])
    return full_score_df


class NavsimMetricV1Impl:
    """v1 navsim scoring implementation extracted from ``NavsimMetric``."""

    def __init__(
        self,
        metric_cache_path: str,
        future_seconds: float,
        interval_seconds: float,
    ):
        _require_metric_impl_core_deps()
        self.proposal_sampling = TrajectorySampling(
            num_poses=int(future_seconds / interval_seconds),
            interval_length=interval_seconds,
        )
        self.simulator = PDMSimulator(self.proposal_sampling)
        self.metric_cache_loader = NavsimMetricCacheLoader(
            pack_path=metric_cache_path
        )
        self.scorer = PDMScorer(self.proposal_sampling)

    @classmethod
    def from_metric_config(
        cls,
        metric_cache_path: str,
        future_seconds: float,
        interval_seconds: float,
    ) -> "NavsimMetricV1Impl":
        logger.info(
            f"Initializing NavsimMetricV1Impl with metric_cache_path={metric_cache_path}, "  # noqa: E501
            f"future_seconds={future_seconds}, interval_seconds={interval_seconds}"  # noqa: E501
        )
        print("cls:", cls)
        return cls(
            metric_cache_path=metric_cache_path,
            future_seconds=future_seconds,
            interval_seconds=interval_seconds,
        )

    @classmethod
    def from_metric(cls, metric: Any) -> "NavsimMetricV1Impl":
        return cls.from_metric_config(
            metric_cache_path=metric.metric_cache_path,
            future_seconds=metric.future_seconds,
            interval_seconds=metric.interval_seconds,
        )

    def _load_metric_cache(self, token: str) -> Dict[str, Any]:
        payload = self.metric_cache_loader(token)
        return _load_metric_cache_dict(payload)

    def score_single(
        self, token: str, model_traj: Trajectory
    ) -> Dict[str, Any]:
        metric_cache = self._load_metric_cache(token)
        return pdm_score_v1(
            metric_cache=metric_cache,
            model_trajectory=model_traj,
            future_sampling=self.proposal_sampling,
            simulator=self.simulator,
            scorer=self.scorer,
        )

    def update(
        self,
        data: Dict[str, Any],
        outputs: Dict[str, Any],
        results: List[Dict[str, Any]],
    ) -> None:
        poses = outputs["pred_traj"].detach().float().cpu()
        if isinstance(data, list):
            tokens = [scene.scene_metadata.initial_token for scene in data]
        elif "token" in data:
            tokens = data["token"]
        else:
            raise KeyError(
                "Missing token information in data for Navsim v1 update."
            )

        traj_poses = [Trajectory(p) for p in poses]
        for i, token in enumerate(tokens):
            score_row: Dict[str, Any] = {"token": token, "valid": True}
            pdm_score = self.score_single(token, traj_poses[i])
            score_row.update(pdm_score)
            results.append(score_row)

    def summarize(self, all_scores: List[Any]) -> Dict[str, float]:
        metrics_summary: Dict[str, float] = {}
        keys = [
            "pdm_score",
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "ego_progress",
            "time_to_collision_within_bound",
            "comfort",
            "driving_direction_compliance",
        ]
        for key in keys:
            values = [
                row[key]
                for row in all_scores
                if key in row and row.get("valid", True) is not False
            ]
            metrics_summary[key] = sum(values) / len(values) if values else 0.0
        return metrics_summary


class NavsimMetricV2Impl(NavsimMetricV1Impl):
    """v2 navsim scoring implementation extracted from ``NavsimMetric``."""

    def __init__(
        self,
        metric_cache_path: str,
        future_seconds: float,
        interval_seconds: float,
        traffic_agents_policy: Any,
        save_path: str,
        eval_stage: str,
        verbose: bool,
        visualize: bool,
        reactive_all_mapping: Optional[
            Dict[Tuple[str, str], List[List[str]]]
        ] = None,
    ):
        _require_extended_pdm("NavsimMetricV2Impl.__init__")
        super().__init__(
            metric_cache_path=metric_cache_path,
            future_seconds=future_seconds,
            interval_seconds=interval_seconds,
        )
        self.scorer = PDMScorer(
            self.proposal_sampling,
            config=PDMScorerConfig(
                progress_weight=5.0,
                ttc_weight=5.0,
                lane_keeping_weight=2.0,
                history_comfort_weight=2.0,
                two_frame_extended_comfort_weight=2.0,
                driving_direction_horizon=1.0,
                driving_direction_compliance_threshold=2.0,
                driving_direction_violation_threshold=6.0,
                stopped_speed_threshold=5e-03,
                future_collision_horizon_window=1.0,
                progress_distance_threshold=5.0,
                lane_keeping_deviation_limit=0.5,
                lane_keeping_horizon_window=2.0,
                human_penalty_filter=True,
            ),
        )
        self.traffic_agents_policy = traffic_agents_policy
        self.save_path = save_path
        self.eval_stage = eval_stage
        self.verbose = verbose
        self.visualize = visualize
        self.reactive_all_mapping = reactive_all_mapping

    @classmethod
    def from_metric_config(
        cls,
        metric_cache_path: str,
        future_seconds: float,
        interval_seconds: float,
        save_path: str,
        eval_stage: str,
        verbose: bool,
        visualize: bool,
        reactive_all_mapping: Optional[
            Dict[Tuple[str, str], List[List[str]]]
        ] = None,
    ) -> "NavsimMetricV2Impl":
        _require_extended_pdm("NavsimMetricV2Impl.from_metric_config")
        logger.info(
            f"Creating NavsimMetricV2Impl with metric_cache_path={metric_cache_path}, "  # noqa: E501
            f"future_seconds={future_seconds}, interval_seconds={interval_seconds}, "  # noqa: E501
            f"save_path={save_path}, eval_stage={eval_stage}, verbose={verbose}, "  # noqa: E501
            f"visualize={visualize}, reactive_all_mapping_provided={reactive_all_mapping is not None}"  # noqa: E501
        )

        proposal_sampling = TrajectorySampling(
            num_poses=int(future_seconds / interval_seconds),
            interval_length=interval_seconds,
        )
        traffic_agents_policy = NavsimIDMTrafficAgents(
            idm_agents_observation=NavsimIDMAgents(
                target_velocity=10,
                min_gap_to_lead_agent=1.0,
                headway_time=1.5,
                accel_max=1.0,
                decel_max=2.0,
                open_loop_detections_types=[],
                minimum_path_length=20,
                planned_trajectory_samples=None,
                planned_trajectory_sample_interval=None,
                radius=100,
                add_open_loop_parked_vehicles=True,
                idm_snap_threshold=3.0,
            ),
            future_trajectory_sampling=proposal_sampling,
            map_root_override=os.getenv("NUPLAN_MAPS_ROOT", None),
        )
        return cls(
            metric_cache_path=metric_cache_path,
            future_seconds=future_seconds,
            interval_seconds=interval_seconds,
            traffic_agents_policy=traffic_agents_policy,
            save_path=save_path,
            eval_stage=eval_stage,
            verbose=verbose,
            visualize=visualize,
            reactive_all_mapping=reactive_all_mapping,
        )

    @classmethod
    def from_metric(cls, metric: Any) -> "NavsimMetricV2Impl":
        return cls.from_metric_config(
            metric_cache_path=metric.metric_cache_path,
            future_seconds=metric.future_seconds,
            interval_seconds=metric.interval_seconds,
            save_path=metric.save_path,
            eval_stage=metric.eval_stage,
            verbose=metric.verbose,
            visualize=metric.visualize,
            reactive_all_mapping=metric.reactive_all_mapping,
        )

    def score_single(self, row: Dict[str, Any]) -> Dict[str, Any]:
        token = row["token"]
        trajectory_np = row["trajectory"]
        visualize = row.get("visualize", self.visualize)

        metric_cache = self._load_metric_cache(token)
        model_traj = Trajectory(trajectory_np)
        score_df, ego_simulated_states = pdm_score_v2(
            metric_cache=metric_cache,
            model_trajectory=model_traj,
            future_sampling=self.proposal_sampling,
            simulator=self.simulator,
            scorer=self.scorer,
            traffic_agents_policy=self.traffic_agents_policy,
            visualize=visualize,
        )

        result = score_df.iloc[0].to_dict()
        result["valid"] = True
        result["token"] = token
        result["log_name"] = metric_cache["log_name"]
        result["frame_type"] = metric_cache["scene_type"]
        result["start_time"] = metric_cache["timepoint"].time_s

        end_pose = StateSE2(
            x=model_traj.poses[-1, 0],
            y=model_traj.poses[-1, 1],
            heading=model_traj.poses[-1, 2],
        )
        absolute_endpoint = relative_to_absolute_poses(
            metric_cache["ego_state"].rear_axle, [end_pose]
        )[0]
        result["endpoint_x"] = absolute_endpoint.x
        result["endpoint_y"] = absolute_endpoint.y
        result["start_point_x"] = metric_cache["ego_state"].rear_axle.x
        result["start_point_y"] = metric_cache["ego_state"].rear_axle.y
        result["ego_simulated_states"] = ego_simulated_states
        return result

    def update(
        self,
        data: Dict[str, Any],
        outputs: Dict[str, Any],
        results: List[Dict[str, Any]],
    ) -> None:
        poses = outputs["pred_traj"].detach().float().cpu()
        if isinstance(data, list):
            tokens = [scene.scene_metadata.initial_token for scene in data]
        elif "token" in data:
            tokens = data["token"]
        else:
            raise KeyError(
                "Missing token information in data for Navsim v2 update."
            )

        for i, token in enumerate(tokens):
            row: Dict[str, Any] = {
                "token": token,
                "valid": True,
                "visualize": self.visualize,
                "trajectory": poses[i].numpy(),
            }
            results.append(self.score_single(row))

    def _aggregate_two_stage_scores(self, all_scores: List[Any]) -> float:
        all_scores = [
            pd.DataFrame([x]) if isinstance(x, dict) else x for x in all_scores
        ]
        pdm_score_df = pd.concat(all_scores, ignore_index=True)
        pdm_score_df = pdm_score_df.drop_duplicates(subset=["token"])
        logger.info("Computing v2 extend PDM Score...")

        available_tokens = set(pdm_score_df["token"].tolist())
        all_mappings: Dict[Tuple[str, str], List[List[str]]] = {}
        reactive_all_mapping = self.reactive_all_mapping

        def _is_non_string_sequence(obj: Any) -> bool:
            return isinstance(obj, Sequence) and not isinstance(
                obj, (str, bytes)
            )

        parsed_mapping_entries: List[Tuple[str, str, Any]] = []
        if reactive_all_mapping is not None:
            # Dict/DictConfig style: {(orig, prev): [[s1, s2], ...], ...}
            try:
                reactive_items = reactive_all_mapping.items()
            except Exception:
                reactive_items = None
            if reactive_items is not None:
                for key, second_stage_pairs in reactive_items:
                    if _is_non_string_sequence(key) and len(key) >= 2:
                        parsed_mapping_entries.append(
                            (key[0], key[1], second_stage_pairs)
                        )
                    else:
                        logger.warning(
                            "Skip malformed reactive mapping key: %s", key
                        )
            else:
                # List/ListConfig style: [[orig, prev, [[s1, s2], ...]], ...]
                for entry in reactive_all_mapping:
                    if _is_non_string_sequence(entry) and len(entry) >= 3:
                        parsed_mapping_entries.append(
                            (entry[0], entry[1], entry[2])
                        )
                    else:
                        logger.warning(
                            "Skip malformed reactive mapping entry: %s", entry
                        )

        for (
            orig_token,
            prev_token,
            second_stage_pairs,
        ) in parsed_mapping_entries:
            if (
                orig_token not in available_tokens
                or prev_token not in available_tokens
                or len(second_stage_pairs) == 0
            ):
                continue

            valid_second_stage = []
            for pair in second_stage_pairs:
                if len(pair) < 2:
                    continue
                if pair[0] in available_tokens and pair[1] in available_tokens:
                    valid_second_stage.append(pair)

            if valid_second_stage:
                all_mappings[(orig_token, prev_token)] = valid_second_stage
            elif len(valid_second_stage) == 0:
                logger.warning(
                    "No complete second-stage pairs available for (%s, %s), skipping group.",  # noqa: E501
                    orig_token,
                    prev_token,
                )

        if len(all_mappings) > 0:
            for (orig_token, prev_token), _ in all_mappings.items():
                if orig_token not in available_tokens:
                    logger.warning(
                        f"Current frame token {orig_token} missing in results."
                    )
                if prev_token not in available_tokens:
                    logger.warning(
                        f"Previous frame token {prev_token} missing in results."  # noqa: E501
                    )

        pdm_score_df = create_scene_aggregators_two_stage(
            all_mappings,
            pdm_score_df,
            self.proposal_sampling,
        )
        pdm_score_df = compute_final_scores_two_stage(pdm_score_df)
        pseudo_closed_loop_valid = True

        num_sucessful_scenarios = pdm_score_df["valid"].sum()
        num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
        failed_tokens = (
            pdm_score_df[~pdm_score_df["valid"]]["token"].to_list()
            if num_failed_scenarios > 0
            else []
        )

        score_cols = [
            c
            for c in pdm_score_df.columns
            if (
                (
                    any(score.name in c for score in fields(PDMResults))
                    or c == "two_frame_extended_comfort"
                    or c == "score"
                )
                and c != "pdm_score"
            )
        ]

        pcl_group_score, pcl_stage1_score, pcl_stage2_score = (
            calculate_individual_mapping_scores(
                pdm_score_df[score_cols + ["token", "weight"]], all_mappings
            )
        )

        for col in score_cols:
            stage_one_mask = (
                pdm_score_df["frame_type"] == SceneFrameType.ORIGINAL
            )
            stage_two_mask = (
                pdm_score_df["frame_type"] == SceneFrameType.SYNTHETIC
            )
            pdm_score_df.loc[stage_one_mask, f"{col}_stage_one"] = (
                pdm_score_df.loc[stage_one_mask, col]
            )
            pdm_score_df.loc[stage_two_mask, f"{col}_stage_two"] = (
                pdm_score_df.loc[stage_two_mask, col]
            )

        pdm_score_df.drop(columns=score_cols, inplace=True)
        pdm_score_df["score"] = pdm_score_df["score_stage_one"].combine_first(
            pdm_score_df["score_stage_two"]
        )
        pdm_score_df.drop(
            columns=["score_stage_one", "score_stage_two"], inplace=True
        )

        stage1_cols = [
            f"{col}_stage_one" for col in score_cols if col != "score"
        ]
        stage2_cols = [
            f"{col}_stage_two" for col in score_cols if col != "score"
        ]
        score_cols = stage1_cols + stage2_cols + ["score"]
        pdm_score_df = pdm_score_df[["token", "valid"] + score_cols]

        summary_rows = []
        stage1_row = pd.Series(index=pdm_score_df.columns, dtype=object)
        stage1_row["token"] = "extended_pdm_score_stage_one"
        stage1_row["valid"] = pseudo_closed_loop_valid
        stage1_row["score"] = pcl_stage1_score.get("score", np.nan)
        for col in pcl_stage1_score.index:
            if col not in ["token", "valid", "score"]:
                stage1_row[f"{col}_stage_one"] = pcl_stage1_score[col]
        summary_rows.append(stage1_row)

        stage2_row = pd.Series(index=pdm_score_df.columns, dtype=object)
        stage2_row["token"] = "extended_pdm_score_stage_two"
        stage2_row["valid"] = pseudo_closed_loop_valid
        stage2_row["score"] = pcl_stage2_score.get("score", np.nan)
        for col in pcl_stage2_score.index:
            if col not in ["token", "valid", "score"]:
                stage2_row[f"{col}_stage_two"] = pcl_stage2_score[col]
        summary_rows.append(stage2_row)

        combined_row = pd.Series(index=pdm_score_df.columns, dtype=object)
        combined_row["token"] = "extended_pdm_score_combined"
        combined_row["valid"] = pseudo_closed_loop_valid
        combined_row["score"] = pcl_group_score["score"]
        for col in pcl_stage1_score.index:
            if col not in ["token", "valid", "score"]:
                combined_row[f"{col}_stage_one"] = pcl_stage1_score[col]
        for col in pcl_stage2_score.index:
            if col not in ["token", "valid", "score"]:
                combined_row[f"{col}_stage_two"] = pcl_stage2_score[col]
        summary_rows.append(combined_row)

        pdm_score_df = pd.concat(
            [pdm_score_df, pd.DataFrame(summary_rows)], ignore_index=True
        )

        timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        pdm_result_file = os.path.join(
            self.save_path, f"pdm_v2_results_{timestamp}.csv"
        )
        pdm_score_df.to_csv(pdm_result_file)

        logger.info(f"""
            Finished running evaluation.
                Number of successful scenarios: {num_sucessful_scenarios}.
                Number of failed scenarios: {num_failed_scenarios}.
Final extended pdm score of valid results: {pdm_score_df[pdm_score_df["token"]
                == "extended_pdm_score_combined"]["score"].iloc[0]}.
                Results are stored in: {pdm_result_file}.
            """)
        if self.verbose:
            logger.info(f"""
                Detailed results:
                {pdm_score_df.iloc[-3:].T}
                """)
        if num_failed_scenarios > 0:
            logger.info(f"""
                List of failed tokens:
                {failed_tokens}
                """)
        return pdm_score_df[
            pdm_score_df["token"] == "extended_pdm_score_combined"
        ]["score"].iloc[0]

    def _aggregate_one_stage_scores(self, all_scores: List[Any]) -> float:
        all_scores = [
            pd.DataFrame([x]) if isinstance(x, dict) else x for x in all_scores
        ]
        pdm_score_df = pd.concat(all_scores, ignore_index=True)
        pdm_score_df = pdm_score_df.drop_duplicates(subset=["token"])
        logger.info("Computing v2 extend PDM Score...")

        start_adjacent_mapping = infer_start_adjacent_mapping(pdm_score_df)
        pdm_score_df = create_scene_aggregators_one_stage(
            start_adjacent_mapping, pdm_score_df, self.proposal_sampling
        )
        pdm_score_df = compute_final_scores_one_stage(pdm_score_df)

        num_sucessful_scenarios = pdm_score_df["valid"].sum()
        num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
        failed_tokens = (
            pdm_score_df[~pdm_score_df["valid"]]["token"].to_list()
            if num_failed_scenarios > 0
            else []
        )

        score_cols = [
            c
            for c in pdm_score_df.columns
            if (
                (
                    any(score.name in c for score in fields(PDMResults))
                    or c == "two_frame_extended_comfort"
                    or c == "score"
                )
                and c != "pdm_score"
            )
        ]

        average_row = pdm_score_df[score_cols].mean(skipna=True)
        average_row["token"] = "average_all_frames"
        average_row["valid"] = pdm_score_df["valid"].all()

        pdm_score_df = pdm_score_df[["token", "valid"] + score_cols]
        pdm_score_df.loc[len(pdm_score_df)] = average_row

        timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        pdm_result_file = os.path.join(
            self.save_path, f"pdm_v2_results_one_stage_{timestamp}.csv"
        )
        pdm_score_df.to_csv(pdm_result_file)

        logger.info(f"""
            Finished running evaluation.
                Number of successful scenarios: {num_sucessful_scenarios}.
                Number of failed scenarios: {num_failed_scenarios}.
Final average score of valid results: {pdm_score_df['score'].mean()}.
                Results are stored in: {pdm_result_file}.
            """)

        if self.verbose:
            logger.info(f"""
                Detailed results:
                {pdm_score_df.iloc[-3:].T}
                """)
        if num_failed_scenarios > 0:
            logger.info(f"""
                List of failed tokens:
                {failed_tokens}
                """)
        return pdm_score_df["score"].mean()

    def summarize(self, all_scores: List[Any]) -> Tuple[str, float]:
        if self.eval_stage == "one_stage":
            timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
            dump_path = os.path.join(
                self.save_path, f"pdm_v2_all_scores_one_stage_{timestamp}.pkl"
            )
            with open(dump_path, "wb") as f:
                pickle.dump(all_scores, f)
            logger.info("Dumped one-stage all_scores to %s", dump_path)
            return "epdm_score", self._aggregate_one_stage_scores(all_scores)

        return "epdm_score", self._aggregate_two_stage_scores(all_scores)
