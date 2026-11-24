# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

import msgpack
import numpy as np
import pandas as pd
import torch

# Argoverse2 dataset handling unconditionally requires av2. We let the
# ``ImportError`` raise here so the failure is loud and pointing at the
# real missing dependency, rather than silently degrading to a stub and
# crashing later with an opaque ``'object' has no attribute …`` message.
# ``hat_engine.registry.register_default_config`` catches per-module
# exceptions during its registration walk, so a missing av2 only skips
# registering Argoverse2 types — ``import heal`` itself still succeeds.
from av2.geometry.interpolate import compute_midpoint_line
from av2.map.map_api import ArgoverseStaticMap
from av2.map.map_primitives import Polyline
from av2.utils.io import read_json_file
from hat_engine.data.datasets.data_packer import Packer
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.pack_type import PackTypeMapper
from hat_engine.utils.pack_type.utils import get_packtype_from_path
from hat_engine.utils.package_helper import require_packages
from torch.utils.data import Dataset

__all__ = [
    "Argoverse2Base",
    "Argoverse2PackedDataset",
    "Argoverse2Packer",
]


def safe_list_index(ls: List[Any], elem: Any) -> Optional[int]:
    try:
        return ls.index(elem)
    except ValueError:
        return None


def side_to_directed_lineseg(
    query_point: torch.Tensor,
    start_point: torch.Tensor,
    end_point: torch.Tensor,
) -> str:
    cond = (end_point[0] - start_point[0]) * (
        query_point[1] - start_point[1]
    ) - (end_point[1] - start_point[1]) * (query_point[0] - start_point[0])
    if cond > 0:
        return "LEFT"
    elif cond < 0:
        return "RIGHT"
    else:
        return "CENTER"


@OBJECT_REGISTRY.register
class Argoverse2Base:
    """
    Argoverse2 dataset handler.

    Args:
        num_historical_steps: Number of historical time steps.
        num_future_step: Number of future time steps.
        split: Dataset split name, e.g., 'train', 'val', 'test'.
    """

    def __init__(
        self,
        num_historical_steps: int = 60,
        num_future_steps: int = 50,
        split: str = "val",
    ):
        self.dim = 3
        self.predict_unseen_agents = False
        self.vector_repr = True
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_steps = self.num_historical_steps + self.num_future_steps
        self.split = split
        assert split in ["train", "test", "val"]

        self._agent_types = [
            "vehicle",
            "pedestrian",
            "motorcyclist",
            "cyclist",
            "bus",
            "static",
            "background",
            "construction",
            "riderless_bicycle",
            "unknown",
        ]
        self._agent_categories = [
            "TRACK_FRAGMENT",
            "UNSCORED_TRACK",
            "SCORED_TRACK",
            "FOCAL_TRACK",
        ]
        self._polygon_types = ["VEHICLE", "BIKE", "BUS", "PEDESTRIAN"]
        self._polygon_is_intersections = [True, False, None]
        self._point_types = [
            "DASH_SOLID_YELLOW",
            "DASH_SOLID_WHITE",
            "DASHED_WHITE",
            "DASHED_YELLOW",
            "DOUBLE_SOLID_YELLOW",
            "DOUBLE_SOLID_WHITE",
            "DOUBLE_DASH_YELLOW",
            "DOUBLE_DASH_WHITE",
            "SOLID_YELLOW",
            "SOLID_WHITE",
            "SOLID_DASH_WHITE",
            "SOLID_DASH_YELLOW",
            "SOLID_BLUE",
            "NONE",
            "UNKNOWN",
            "CROSSWALK",
            "CENTERLINE",
        ]
        self._point_sides = ["LEFT", "RIGHT", "CENTER"]
        self._polygon_to_polygon_types = [
            "NONE",
            "PRED",
            "SUCC",
            "LEFT",
            "RIGHT",
        ]

    @require_packages("av2")
    def process(self, input_dir: str):
        """
        Process data from the specified input directory.

        Args:
            input_dir: Path to the directory containing input data.

        Returns:
            pd.DataFrame: Processed data as a Pandas DataFrame.
        """
        assert (
            ArgoverseStaticMap is not None
        ), "av2 should be installed for argoverse2 data process."
        raw_file_name = os.path.basename(input_dir)
        map_dir = Path(input_dir)
        map_path = sorted(map_dir.glob("log_map_archive_*.json"))[0]
        map_data = read_json_file(map_path)

        df = pd.read_parquet(
            os.path.join(
                input_dir,
                f"scenario_{raw_file_name}.parquet",
            )
        )
        centerlines = {
            lane_segment["id"]: Polyline.from_json_data(
                lane_segment["centerline"]
            )
            for lane_segment in map_data["lane_segments"].values()
        }
        map_api = ArgoverseStaticMap.from_json(map_path)
        data = {}
        data["agent"] = self.get_agent_features(df)
        data.update(self.get_map_features(map_api, centerlines))

        return data, map_path

    def _decode(self, sample: dict, input_dim: int):

        for k in [
            "valid_mask",
            "predict_mask",
            "type",
            "category",
            "position",
            "heading",
            "velocity",
        ]:
            sample["agent"][k] = torch.tensor(sample["agent"][k]).clone()

        for k in [
            "position",
            "orientation",
            "height",
            "type",
            "is_intersection",
        ]:
            sample["map_polygon"][k] = torch.tensor(
                sample["map_polygon"][k]
            ).clone()
        for k in [
            "position",
            "orientation",
            "magnitude",
            "height",
            "type",
            "side",
        ]:
            sample["map_point"][k] = [
                torch.tensor(p) for p in sample["map_point"][k]
            ]
        sample["map_point_to_map_polygon"]["edge_index"] = (
            torch.tensor(sample["map_point_to_map_polygon"]["edge_index"])
            .long()
            .clone()
        )
        sample["map_polygon_to_map_polygon"]["edge_index"] = torch.tensor(
            sample["map_polygon_to_map_polygon"]["edge_index"]
        ).clone()
        sample["map_polygon_to_map_polygon"]["type"] = torch.tensor(
            sample["map_polygon_to_map_polygon"]["type"]
        ).clone()

        sample["map_polygon"]["position"] = sample["map_polygon"]["position"][
            :, :input_dim
        ]
        pl_num = sample["map_polygon"]["num_nodes"]
        pl2pl_type_mat = torch.zeros([pl_num, pl_num])
        for i in range(
            sample["map_polygon_to_map_polygon"]["edge_index"].shape[1]
        ):
            src, dst = sample["map_polygon_to_map_polygon"]["edge_index"][:, i]
            pl2pl_type_mat[src][dst] = sample["map_polygon_to_map_polygon"][
                "type"
            ][i]
        sample["map_polygon_to_map_polygon"]["type_mat"] = pl2pl_type_mat
        return sample

    @require_packages("av2")
    def get_agent_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        # Optionally restrict to agents that appear within the history window.
        if self.predict_unseen_agents:
            agent_ids = list(df["track_id"].unique())
        else:
            in_history = df[df["timestep"] < self.num_historical_steps]
            agent_ids = list(in_history["track_id"].unique())
            df = df[df["track_id"].isin(agent_ids)]

        num_agents = len(agent_ids)
        av_idx = agent_ids.index("AV")

        # Pre-allocate the per-agent feature buffers.
        steps_shape = (num_agents, self.num_steps)
        valid_mask = torch.zeros(*steps_shape, dtype=torch.bool)
        predict_mask = torch.zeros(*steps_shape, dtype=torch.bool)
        current_valid_mask = torch.zeros(num_agents, dtype=torch.bool)
        agent_id: List[Optional[str]] = [None] * num_agents
        agent_type = torch.zeros(num_agents, dtype=torch.uint8)
        agent_category = torch.zeros(num_agents, dtype=torch.uint8)
        position = torch.zeros(*steps_shape, self.dim, dtype=torch.float)
        velocity = torch.zeros(*steps_shape, self.dim, dtype=torch.float)
        heading = torch.zeros(*steps_shape, dtype=torch.float)

        hist = self.num_historical_steps

        def _stack_xy(track, col_x, col_y):
            xy = np.stack([track[col_x].values, track[col_y].values], axis=-1)
            return torch.from_numpy(xy).float()

        for track_id, track_df in df.groupby("track_id"):
            agent_idx = agent_ids.index(track_id)
            steps = track_df["timestep"].values

            valid_mask[agent_idx, steps] = True
            current_valid_mask[agent_idx] = valid_mask[agent_idx, hist - 1]
            predict_mask[agent_idx, steps] = True
            if self.vector_repr:
                # Under vector representation a step is valid only when the
                # step and its predecessor are both observed.
                valid_mask[agent_idx, 1:hist] = (
                    valid_mask[agent_idx, : hist - 1]
                    & valid_mask[agent_idx, 1:hist]
                )
                valid_mask[agent_idx, 0] = False
            predict_mask[agent_idx, :hist] = False
            if not current_valid_mask[agent_idx]:
                predict_mask[agent_idx, hist:] = False

            agent_id[agent_idx] = track_id
            agent_type[agent_idx] = self._agent_types.index(
                track_df["object_type"].values[0]
            )
            agent_category[agent_idx] = track_df["object_category"].values[0]
            position[agent_idx, steps, :2] = _stack_xy(
                track_df, "position_x", "position_y"
            )
            velocity[agent_idx, steps, :2] = _stack_xy(
                track_df, "velocity_x", "velocity_y"
            )
            heading[agent_idx, steps] = torch.from_numpy(
                track_df["heading"].values
            ).float()

        if self.split == "test":
            predict_mask[
                current_valid_mask
                | (agent_category == 2)
                | (agent_category == 3),
                self.num_historical_steps :,
            ] = True

        return {
            "num_nodes": num_agents,
            "av_index": av_idx,
            "valid_mask": valid_mask.numpy().tolist(),
            "predict_mask": predict_mask.numpy().tolist(),
            "id": agent_id,
            "type": agent_type.numpy().tolist(),
            "category": agent_category.numpy().tolist(),
            "position": position.numpy().tolist(),
            "heading": heading.numpy().tolist(),
            "velocity": velocity.numpy().tolist(),
        }

    def _polyline_point_features(
        self,
        left_b: torch.Tensor,
        right_b: torch.Tensor,
        center_b: torch.Tensor,
        left_type: int,
        right_type: int,
    ):
        """Derive per-point map features for a left/right/center triple.

        Returns the (position, orientation, magnitude, height, type, side)
        tensors for the three boundaries concatenated in that order. The
        same routine serves lane segments as well as both traversal
        directions of a pedestrian crossing (callers pass already-flipped
        boundaries for the reverse direction).
        """
        d = self.dim
        center_type = self._point_types.index("CENTERLINE")
        sides = self._point_sides

        left_vec = left_b[1:] - left_b[:-1]
        right_vec = right_b[1:] - right_b[:-1]
        center_vec = center_b[1:] - center_b[:-1]

        def _filled(vec: torch.Tensor, value: int) -> torch.Tensor:
            return torch.full((len(vec),), value, dtype=torch.uint8)

        position = torch.cat(
            [left_b[:-1, :d], right_b[:-1, :d], center_b[:-1, :d]], dim=0
        )
        orientation = torch.cat(
            [
                torch.atan2(left_vec[:, 1], left_vec[:, 0]),
                torch.atan2(right_vec[:, 1], right_vec[:, 0]),
                torch.atan2(center_vec[:, 1], center_vec[:, 0]),
            ],
            dim=0,
        )
        magnitude = torch.norm(
            torch.cat(
                [left_vec[:, :2], right_vec[:, :2], center_vec[:, :2]], dim=0
            ),
            p=2,
            dim=-1,
        )
        height = torch.cat(
            [left_vec[:, 2], right_vec[:, 2], center_vec[:, 2]], dim=0
        )
        ptype = torch.cat(
            [
                _filled(left_vec, left_type),
                _filled(right_vec, right_type),
                _filled(center_vec, center_type),
            ],
            dim=0,
        )
        side = torch.cat(
            [
                _filled(left_vec, sides.index("LEFT")),
                _filled(right_vec, sides.index("RIGHT")),
                _filled(center_vec, sides.index("CENTER")),
            ],
            dim=0,
        )
        return position, orientation, magnitude, height, ptype, side

    def _map_polygon_id_layout(self, map_api: ArgoverseStaticMap):
        lane_ids = map_api.get_scenario_lane_segment_ids()
        crossing_ids = list(map_api.vector_pedestrian_crossings.keys())
        ordered_ids = lane_ids + crossing_ids
        crossing_count = len(crossing_ids)
        slot_count = len(lane_ids) + 2 * crossing_count
        return ordered_ids, slot_count, crossing_count

    def _create_empty_map_buffers(self, slot_count: int):
        empty_pts: List[Optional[torch.Tensor]] = [None] * slot_count
        return (
            torch.zeros(slot_count, self.dim, dtype=torch.float),
            torch.zeros(slot_count, dtype=torch.float),
            torch.zeros(slot_count, dtype=torch.float),
            torch.zeros(slot_count, dtype=torch.uint8),
            torch.zeros(slot_count, dtype=torch.uint8),
            list(empty_pts),
            list(empty_pts),
            list(empty_pts),
            list(empty_pts),
            list(empty_pts),
            list(empty_pts),
        )

    def _fill_lane_polygon_summary(
        self,
        lane_segment,
        centerline: torch.Tensor,
        slot: int,
        poly_pos,
        poly_yaw,
        poly_z,
        poly_kind,
        poly_cross,
    ) -> None:
        anchor = centerline[0, : self.dim]
        tip = centerline[1]
        poly_pos[slot] = anchor
        poly_yaw[slot] = torch.atan2(tip[1] - anchor[1], tip[0] - anchor[0])
        poly_z[slot] = tip[2] - anchor[2]
        poly_kind[slot] = self._polygon_types.index(
            lane_segment.lane_type.value
        )
        poly_cross[slot] = self._polygon_is_intersections.index(
            lane_segment.is_intersection
        )

    def _boundary_xyz_tensor(self, boundary) -> torch.Tensor:
        return torch.from_numpy(boundary.xyz).float()

    @require_packages("av2")
    def get_map_features(
        self, map_api: ArgoverseStaticMap, centerlines: Mapping[str, Polyline]
    ) -> Dict[Union[str, Tuple[str, str, str]], Any]:
        (
            polygon_ids,
            num_polygons,
            crossing_count,
        ) = self._map_polygon_id_layout(map_api)
        (
            polygon_position,
            polygon_orientation,
            polygon_height,
            polygon_type,
            polygon_is_intersection,
            point_position,
            point_orientation,
            point_magnitude,
            point_height,
            point_type,
            point_side,
        ) = self._create_empty_map_buffers(num_polygons)

        for lane_segment in map_api.get_scenario_lane_segments():
            lane_segment_idx = polygon_ids.index(lane_segment.id)
            centerline = torch.from_numpy(
                centerlines[lane_segment.id].xyz
            ).float()
            self._fill_lane_polygon_summary(
                lane_segment,
                centerline,
                lane_segment_idx,
                polygon_position,
                polygon_orientation,
                polygon_height,
                polygon_type,
                polygon_is_intersection,
            )
            left_boundary = self._boundary_xyz_tensor(
                lane_segment.left_lane_boundary
            )
            right_boundary = torch.from_numpy(
                lane_segment.right_lane_boundary.xyz
            ).float()
            (
                point_position[lane_segment_idx],
                point_orientation[lane_segment_idx],
                point_magnitude[lane_segment_idx],
                point_height[lane_segment_idx],
                point_type[lane_segment_idx],
                point_side[lane_segment_idx],
            ) = self._polyline_point_features(
                left_boundary,
                right_boundary,
                centerline,
                self._point_types.index(lane_segment.left_mark_type.value),
                self._point_types.index(lane_segment.right_mark_type.value),
            )

        for crosswalk in map_api.get_scenario_ped_crossings():
            crosswalk_idx = polygon_ids.index(crosswalk.id)
            edge1 = torch.from_numpy(crosswalk.edge1.xyz).float()
            edge2 = torch.from_numpy(crosswalk.edge2.xyz).float()
            start_position = (edge1[0] + edge2[0]) / 2
            end_position = (edge1[-1] + edge2[-1]) / 2
            # The crossing yields two polygons: forward (start->end) and the
            # reverse direction stored ``crossing_count`` slots later.
            rev_idx = crosswalk_idx + crossing_count
            delta = end_position - start_position
            ped_type = self._polygon_types.index("PEDESTRIAN")
            none_inter = self._polygon_is_intersections.index(None)

            polygon_position[crosswalk_idx] = start_position[: self.dim]
            polygon_position[rev_idx] = end_position[: self.dim]
            polygon_orientation[crosswalk_idx] = torch.atan2(
                delta[1], delta[0]
            )
            polygon_orientation[rev_idx] = torch.atan2(-delta[1], -delta[0])
            polygon_height[crosswalk_idx] = delta[2]
            polygon_height[rev_idx] = -delta[2]
            polygon_type[crosswalk_idx] = ped_type
            polygon_type[rev_idx] = ped_type
            polygon_is_intersection[crosswalk_idx] = none_inter
            polygon_is_intersection[rev_idx] = none_inter

            if (
                side_to_directed_lineseg(
                    (edge1[0] + edge1[-1]) / 2, start_position, end_position
                )
                == "LEFT"
            ):
                left_boundary = edge1
                right_boundary = edge2
            else:
                left_boundary = edge2
                right_boundary = edge1
            num_centerline_points = (
                math.ceil(
                    torch.norm(
                        end_position - start_position, p=2, dim=-1
                    ).item()
                    / 2.0
                )
                + 1
            )
            centerline = torch.from_numpy(
                compute_midpoint_line(
                    left_ln_boundary=left_boundary.numpy(),
                    right_ln_boundary=right_boundary.numpy(),
                    num_interp_pts=int(num_centerline_points),
                )[0]
            ).float()

            crosswalk_type = self._point_types.index("CROSSWALK")
            # Forward traversal of the crossing.
            (
                point_position[crosswalk_idx],
                point_orientation[crosswalk_idx],
                point_magnitude[crosswalk_idx],
                point_height[crosswalk_idx],
                point_type[crosswalk_idx],
                point_side[crosswalk_idx],
            ) = self._polyline_point_features(
                left_boundary,
                right_boundary,
                centerline,
                crosswalk_type,
                crosswalk_type,
            )
            # Reverse traversal: swap and flip the boundaries so the shared
            # routine yields the opposite-direction features.
            (
                point_position[rev_idx],
                point_orientation[rev_idx],
                point_magnitude[rev_idx],
                point_height[rev_idx],
                point_type[rev_idx],
                point_side[rev_idx],
            ) = self._polyline_point_features(
                right_boundary.flip(dims=[0]),
                left_boundary.flip(dims=[0]),
                centerline.flip(dims=[0]),
                crosswalk_type,
                crosswalk_type,
            )

        num_points = torch.tensor(
            [point.size(0) for point in point_position], dtype=torch.long
        )
        point_to_polygon_edge_index = torch.stack(
            [
                torch.arange(num_points.sum(), dtype=torch.long),
                torch.arange(num_polygons, dtype=torch.long).repeat_interleave(
                    num_points
                ),
            ],
            dim=0,
        )
        polygon_to_polygon_edge_index = []
        polygon_to_polygon_type = []

        def _resolve_ids(refs):
            resolved = []
            for ref in refs:
                idx = safe_list_index(polygon_ids, ref)
                if idx is not None:
                    resolved.append(idx)
            return resolved

        def _append_edges(src_inds, dst_idx, rel):
            if not src_inds:
                return
            polygon_to_polygon_edge_index.append(
                torch.stack(
                    [
                        torch.tensor(src_inds, dtype=torch.long),
                        torch.full(
                            (len(src_inds),), dst_idx, dtype=torch.long
                        ),
                    ],
                    dim=0,
                )
            )
            polygon_to_polygon_type.append(
                torch.full(
                    (len(src_inds),),
                    self._polygon_to_polygon_types.index(rel),
                    dtype=torch.uint8,
                )
            )

        for lane_segment in map_api.get_scenario_lane_segments():
            lane_segment_idx = polygon_ids.index(lane_segment.id)
            # Predecessor / successor relations (possibly many each).
            _append_edges(
                _resolve_ids(lane_segment.predecessors),
                lane_segment_idx,
                "PRED",
            )
            _append_edges(
                _resolve_ids(lane_segment.successors),
                lane_segment_idx,
                "SUCC",
            )
            # Left / right neighbour relations (at most one each).
            for neighbor_id, rel in (
                (lane_segment.left_neighbor_id, "LEFT"),
                (lane_segment.right_neighbor_id, "RIGHT"),
            ):
                if neighbor_id is None:
                    continue
                nbr_idx = safe_list_index(polygon_ids, neighbor_id)
                if nbr_idx is not None:
                    _append_edges([nbr_idx], lane_segment_idx, rel)
        if len(polygon_to_polygon_edge_index) != 0:
            polygon_to_polygon_edge_index = torch.cat(
                polygon_to_polygon_edge_index, dim=1
            )
            polygon_to_polygon_type = torch.cat(polygon_to_polygon_type, dim=0)
        else:
            polygon_to_polygon_edge_index = torch.tensor(
                [[], []], dtype=torch.long
            )
            polygon_to_polygon_type = torch.tensor([], dtype=torch.uint8)

        map_data = {
            "map_polygon": {},
            "map_point": {},
            "map_point_to_map_polygon": {},
            "map_polygon_to_map_polygon": {},
        }
        map_data["map_polygon"][
            "num_nodes"
        ] = num_polygons  # .numpy().tolist()
        map_data["map_polygon"]["position"] = polygon_position.numpy().tolist()
        map_data["map_polygon"][
            "orientation"
        ] = polygon_orientation.numpy().tolist()
        if self.dim == 3:
            map_data["map_polygon"]["height"] = polygon_height.numpy().tolist()
        map_data["map_polygon"]["type"] = polygon_type.numpy().tolist()
        map_data["map_polygon"][
            "is_intersection"
        ] = polygon_is_intersection.numpy().tolist()
        if len(num_points) == 0:
            map_data["map_point"]["num_nodes"] = 0
            map_data["map_point"]["position"] = []
            map_data["map_point"]["orientation"] = []
            map_data["map_point"]["magnitude"] = []
            if self.dim == 3:
                map_data["map_point"]["height"] = []
            map_data["map_point"]["type"] = []
            map_data["map_point"]["side"] = []
        else:
            map_data["map_point"]["num_nodes"] = num_points.sum().item()

            map_data["map_point"]["num_points"] = num_points.numpy().tolist()
            map_data["map_point"]["position"] = [
                p.numpy().tolist() for p in point_position
            ]
            map_data["map_point"]["orientation"] = [
                p.numpy().tolist() for p in point_orientation
            ]
            map_data["map_point"]["magnitude"] = [
                p.numpy().tolist() for p in point_magnitude
            ]
            if self.dim == 3:
                map_data["map_point"]["height"] = [
                    p.numpy().tolist() for p in point_height
                ]
            map_data["map_point"]["type"] = [
                p.numpy().tolist() for p in point_type
            ]
            map_data["map_point"]["side"] = [
                p.numpy().tolist() for p in point_side
            ]

        map_data["map_point_to_map_polygon"][
            "edge_index"
        ] = point_to_polygon_edge_index.numpy().tolist()
        map_data["map_polygon_to_map_polygon"][
            "edge_index"
        ] = polygon_to_polygon_edge_index.numpy().tolist()
        map_data["map_polygon_to_map_polygon"][
            "type"
        ] = polygon_to_polygon_type.numpy().tolist()

        return map_data


@OBJECT_REGISTRY.register
class Argoverse2PackedDataset(Dataset):
    """Argoverse2 Dataset of packed lmdb format.

    Args:
        data_path: The path to the packed dataset.
        split:  Dataset split name, e.g., 'train', 'val', 'test'.
        transforms: List of data transformations to apply.
        pack_type: The type of packing used for the dataset. here is "lmdb"
        input_dim: input_dim for argoverse2 data decode.
        num_historical_steps: Number of historical time steps.
        num_future_steps: Number of future time steps for prediction.
        pack_kwargs: Additional keyword arguments for dataset packing.
    """

    def __init__(
        self,
        data_path: str,
        split: str = "val",
        transforms: Optional[Callable] = None,
        pack_type: str = "lmdb",
        input_dim: int = 2,
        num_historical_steps: int = 50,
        num_future_steps: int = 60,
        pack_kwargs: Optional[dict] = None,
    ):
        self.data_path = data_path

        if split not in ("train", "val", "test"):
            raise ValueError(f"{split} is not a valid split")
        self.split = split
        self.transforms = transforms
        self.input_dim = input_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.kwargs = {} if pack_kwargs is None else pack_kwargs

        if pack_type is not None:
            self.pack_type = PackTypeMapper[pack_type.lower()]
        else:
            self.pack_type = get_packtype_from_path(data_path)

        self.pack_file = self.pack_type(
            self.data_path, writable=False, **self.kwargs
        )
        self.argorvse2 = Argoverse2Base(
            self.num_historical_steps,
            self.num_future_steps,
            split=split,
        )
        self.pack_file.open()
        self.samples = self.pack_file.get_keys()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        raw_data = self.pack_file.read(self.samples[idx])
        sample = msgpack.unpackb(
            raw_data,
        )
        data = self.argorvse2._decode(sample, self.input_dim)

        if self.transforms:
            data = self.transforms(data)
        return data


class Argoverse2Packer(Packer):
    """Argoverse2 Dataset Packer.

    Args:
        src_data_dir: The directory path where the source data is located.
        target_data_dir: The path where the packed data will be stored.
        split_name: The name of the dataset split to be packed
                 optional: ('train', 'test').
        num_workers: The number of workers to use for parallel processing.
        pack_type: The type of packing to be performed.
        num_samples: The number of samples to pack.
        num_historical_steps: Number of historical time steps.
        num_future_steps: Number of future time steps for prediction.
        **kwargs: Additional keyword arguments for the packing process.
    """

    @require_packages("av2")
    def __init__(
        self,
        src_data_dir: str,
        target_data_dir: str,
        split_name: str = "val",
        num_workers: int = 10,
        pack_type: str = "lmdb",
        num_samples: Optional[int] = None,
        num_historical_steps: int = 50,
        num_future_steps: int = 60,
        **kwargs,
    ):
        self.data_root = src_data_dir
        self.split = split_name
        assert self.split in ["train", "test", "val"]
        self._raw_file_names = [
            name
            for name in os.listdir(os.path.join(self.data_root, self.split))
            if os.path.isdir(os.path.join(self.data_root, self.split, name))
        ]
        self.raw_dir = os.path.join(self.data_root, self.split)
        if num_samples is None:
            num_samples = len(self._raw_file_names)
        self.dim = 3
        self.predict_unseen_agents = False
        self.vector_repr = True
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_steps = self.num_historical_steps + self.num_future_steps
        self._num_samples = {
            "train": 199908,
            "val": 24988,
            "test": 24984,
        }[self.split]
        self.argoverse2 = Argoverse2Base(
            num_historical_steps=self.num_historical_steps,
            num_future_steps=self.num_future_steps,
            split=self.split,
        )
        super(Argoverse2Packer, self).__init__(
            target_data_dir, num_samples, pack_type, num_workers, **kwargs
        )

    def pack_data(self, idx):
        raw_file_name = self._raw_file_names[idx]
        df = pd.read_parquet(
            os.path.join(
                self.raw_dir,
                raw_file_name,
                f"scenario_{raw_file_name}.parquet",
            )
        )
        map_dir = Path(self.raw_dir) / raw_file_name
        map_path = sorted(map_dir.glob("log_map_archive_*.json"))[0]
        map_data = read_json_file(map_path)
        centerlines = {
            lane_segment["id"]: Polyline.from_json_data(
                lane_segment["centerline"]
            )
            for lane_segment in map_data["lane_segments"].values()
        }
        map_api = ArgoverseStaticMap.from_json(map_path)
        data = {}
        data["scenario_id"] = df["scenario_id"].values[0]
        data["city"] = df["city"].values[0]
        data["agent"] = self.argoverse2.get_agent_features(df)
        data.update(self.argoverse2.get_map_features(map_api, centerlines))
        return msgpack.packb(data)
