# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import logging
import warnings
from collections import defaultdict
from functools import lru_cache

import numpy as np
import shapely.geometry as geom
import torch

from heal.data.datasets.navsim.online_mapping_utils import nms_candidates_pts

logger = logging.getLogger(__name__)

try:
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_db
    from nuplan.common.maps.nuplan_map.nuplan_map import NuPlanMap
    from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import (  # noqa
        MapObjectPolylines,
        VectorFeatureLayer,
        VectorFeatureLayerMapping,
    )

    _NUPLAN_IMPORT_ERROR = None
except Exception as e:
    SemanticMapLayer = None
    NuPlanMap = None
    get_maps_db = None
    VectorFeatureLayer = None
    MapObjectPolylines = None
    VectorFeatureLayerMapping = None
    _NUPLAN_IMPORT_ERROR = e
    logger.warning("nuplan map imports failed: %s", e)

try:
    from heal.data.datasets.navsim.nuplan_utils.map_process import (
        _convert_lane_to_fixed_size,
        _get_lane_polylines,
        _lane_polyline_process,
        _prune_route_by_connectivity,
        get_map_object_polygons,
    )
    from heal.data.datasets.navsim.nuplan_utils.utils import (
        vector_set_coordinates_to_local_frame,
    )
except Exception as e:
    _get_lane_polylines = None
    get_map_object_polygons = None
    _convert_lane_to_fixed_size = None
    _lane_polyline_process = None
    _prune_route_by_connectivity = None
    vector_set_coordinates_to_local_frame = None
    logger.warning("nuplan map_process helper imports failed: %s", e)


warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="invalid value encountered in cast",
    module=r"nuplan\.common\.maps\.nuplan_map\.utils",
)

__all__ = [
    "get_lane_gt_by_nms",
    "get_maps_api",
    "get_neighbor_vector_set_map",
    "map_process",
    "_get_discrete_map_edges",
    "_get_rotated_rectangle",
    "discrete_map_process",
]


def get_lane_gt_by_nms(lane_data):
    candi_length = np.sum(
        np.linalg.norm(lane_data[:, 1:] - lane_data[:, :-1], axis=2),
        axis=1,
    )
    fake_labels = np.zeros((lane_data.shape[0], 1), dtype=np.int32)
    filtered_lane_data = nms_candidates_pts(
        candi_length, lane_data, fake_labels, min_close_dist=2.0
    )[-1]

    return filtered_lane_data


@lru_cache(maxsize=32)
def get_maps_api(map_root, map_version, map_name):
    """Get a NuPlanMap object for a particular set of parameters.

    :param map_root: The root folder for the map data.
    :param map_version: The map version to load.
    :param map_name: The map name to load.
    :return: The loaded NuPlanMap object.
    """
    if get_maps_db is None or CustomNuPlanMap is None:
        raise ImportError(
            "nuplan map stack is required for get_maps_api. "
            f"Original import error: {_NUPLAN_IMPORT_ERROR}"
        ) from _NUPLAN_IMPORT_ERROR
    maps_db = get_maps_db(map_root, map_version)
    return CustomNuPlanMap(maps_db, map_name.replace(".gpkg", ""))


def get_neighbor_vector_set_map(map_api, map_features, point, map_pc_range):
    coords, speed_limit = {}, {}
    feature_layers = []

    for feature_name in map_features:
        try:
            feature_layers.append(VectorFeatureLayer[feature_name])
        except KeyError:
            raise ValueError(
                f"Object representation for layer: {feature_name} is unavailable"  # noqa
            )

    # extract lanes
    if VectorFeatureLayer.LANE in feature_layers:
        (
            lanes_mid,
            lanes_left,
            lanes_right,
            lane_ids,
            lane_speed_limit,
            lane_has_speed_limit,
            lane_route,
        ) = _get_lane_polylines(map_api, point, map_pc_range)

        # lane baseline paths
        coords[VectorFeatureLayer.LANE.name] = lanes_mid
        speed_limit["lane_has_speed_limit"] = np.array(
            lane_has_speed_limit, dtype=np.bool_
        )
        speed_limit["lane_speed_limit"] = np.array(
            lane_speed_limit, dtype=np.float32
        )

        # lane boundaries
        if VectorFeatureLayer.LEFT_BOUNDARY in feature_layers:
            coords[VectorFeatureLayer.LEFT_BOUNDARY.name] = MapObjectPolylines(
                lanes_left.polylines
            )
        if VectorFeatureLayer.RIGHT_BOUNDARY in feature_layers:
            coords[VectorFeatureLayer.RIGHT_BOUNDARY.name] = (
                MapObjectPolylines(lanes_right.polylines)
            )

    # extract generic map objects
    for feature_layer in feature_layers:
        if (
            feature_layer
            in VectorFeatureLayerMapping.available_polygon_layers()
        ):
            polygons = get_map_object_polygons(
                map_api,
                point,
                map_pc_range,
                VectorFeatureLayerMapping.semantic_map_layer(feature_layer),
            )
            coords[feature_layer.name] = polygons

    return coords, speed_limit, lane_route


def map_process(
    route_roadblock_ids,
    anchor_ego_state,
    coords,
    speed_limit,
    lane_route,
    map_features,
    max_elements,
    max_points,
    do_nms=False,
    min_length=5.0,
):
    """Process raw vector-set map data into fixed-size arrays.

    :param route_roadblock_ids: route road block ids.
    :param anchor_ego_state: ego current state.
    :param coords: dictionary mapping feature name to polyline vector sets.
    :param speed_limit: speed limit of lanes.
    :param lane_route: road block ids of lanes.
    :param map_features: Name of map features to extract.
    :param max_elements: clip the number of map elements.
    :param max_points: clip the number of point for each element.
    :param do_nms: whether to apply NMS to lane polylines.
    :return: dict of the map elements.
    """
    list_array_data = {}

    for feature_name, feature_coords in coords.items():
        list_feature_coords = []

        # Pack coords into array list
        for element_coords in feature_coords.to_vector():
            list_feature_coords.append(
                np.array(element_coords, dtype=np.float64)
            )
        list_array_data[f"coords.{feature_name}"] = list_feature_coords

    array_output = {}
    for feature_name in map_features:
        if f"coords.{feature_name}" in list_array_data:
            feature_coords = list_array_data[f"coords.{feature_name}"]

            feature_tl_data = None

            if feature_name == "LANE":
                (
                    coords,
                    left_coords,
                    right_coords,
                    tl_data,
                    avails,
                    lane_has_speed_limit_array,
                    lane_speed_limit_array,
                    lane_routes,
                ) = _convert_lane_to_fixed_size(
                    anchor_ego_state,
                    feature_coords,
                    speed_limit,
                    lane_route,
                    list_array_data["coords.LEFT_BOUNDARY"],
                    list_array_data["coords.RIGHT_BOUNDARY"],
                    feature_tl_data,
                    max_elements[feature_name],
                    max_points[feature_name],
                    None,
                )
                left_coords = vector_set_coordinates_to_local_frame(
                    left_coords, avails, anchor_ego_state
                )
                right_coords = vector_set_coordinates_to_local_frame(
                    right_coords, avails, anchor_ego_state
                )
                array_output["vector_set_map.coords.LEFT_BOUNDARY"] = (
                    left_coords
                )
                array_output["vector_set_map.coords.RIGHT_BOUNDARY"] = (
                    right_coords
                )

                lane_on_route = []
                pruned_lane_roadblock_ids = [
                    route
                    for route in route_roadblock_ids
                    if route in lane_routes
                ]
                pruned_route_roadblock_ids = _prune_route_by_connectivity(
                    route_roadblock_ids, pruned_lane_roadblock_ids
                )

                for route in lane_routes:
                    lane_on_route.append(route in pruned_route_roadblock_ids)

            elif (
                feature_name == "LEFT_BOUNDARY"
                or feature_name == "RIGHT_BOUNDARY"
            ):
                continue

            coords = vector_set_coordinates_to_local_frame(
                coords, avails, anchor_ego_state
            )

            array_output[f"vector_set_map.coords.{feature_name}"] = coords
            array_output[f"vector_set_map.availabilities.{feature_name}"] = (
                avails
            )

            if tl_data is not None:
                array_output[
                    f"vector_set_map.traffic_light_data.{feature_name}"
                ] = tl_data

    # Post-process the map elements to different map types.
    for feature_name in map_features:
        if feature_name == "LANE":
            polylines = array_output[f"vector_set_map.coords.{feature_name}"]
            left_boundary = array_output["vector_set_map.coords.LEFT_BOUNDARY"]
            right_boundary = array_output[
                "vector_set_map.coords.RIGHT_BOUNDARY"
            ]
            if (
                f"vector_set_map.traffic_light_data.{feature_name}"
                in array_output
            ):
                traffic_light_state = array_output[
                    f"vector_set_map.traffic_light_data.{feature_name}"
                ]
            else:
                traffic_light_num = polylines.shape[0]
                traffic_light_state = np.zeros(
                    (traffic_light_num, max_points[feature_name], 4)
                )
            avails = array_output[
                f"vector_set_map.availabilities.{feature_name}"
            ]
            vector_map_lanes, peception_polylines = _lane_polyline_process(
                polylines,
                left_boundary,
                right_boundary,
                avails,
                traffic_light_state,
            )
            if do_nms:
                # Remove lanes shorter than ``min_length``.
                lane_length = np.sum(
                    np.linalg.norm(
                        peception_polylines[:, 1:]
                        - peception_polylines[:, :-1],
                        axis=2,
                    ),
                    axis=1,
                )

                peception_polylines = peception_polylines[
                    lane_length > min_length
                ]
                filtered_lane_data_thr20 = get_lane_gt_by_nms(
                    peception_polylines
                )
        else:
            pass

    vector_map_output = {
        "lanes": vector_map_lanes,
        "lanes_pecetion": peception_polylines,
        "lanes_speed_limit": lane_speed_limit_array,
        "lanes_has_speed_limit": lane_has_speed_limit_array,
    }

    if do_nms:
        filter_lane_data = {
            "filtered_lane_data_thr20": filtered_lane_data_thr20,
        }
        vector_map_output.update(filter_lane_data)
    return vector_map_output


def _get_discrete_map_edges(
    discrete_map,
    sample_points=None,
    anchor_ego_state=None,
    return_rotated_rectangle=False,
):
    bbox = discrete_map.polygon.minimum_rotated_rectangle
    coords = np.stack(bbox.exterior.coords.xy, axis=-1)
    edge1 = coords[[3, 0]]  # right boundary
    edge2 = coords[[2, 1]]  # left boundary

    edge1_expanded = np.expand_dims(edge1, axis=0)
    edge2_expanded = np.expand_dims(edge2, axis=0)
    avails = np.ones(edge1_expanded.shape[0], dtype=np.bool_)

    right_coords = vector_set_coordinates_to_local_frame(
        edge1_expanded, avails, anchor_ego_state
    )
    left_coords = vector_set_coordinates_to_local_frame(
        edge2_expanded, avails, anchor_ego_state
    )
    edge1 = np.squeeze(right_coords, axis=0)
    edge2 = np.squeeze(left_coords, axis=0)

    if not return_rotated_rectangle:
        edges = np.stack([(edge1 + edge2) * 0.5, edge2, edge1], axis=0)
        vector = edges[:, 1] - edges[:, 0]
        steps = np.linspace(0, 1, sample_points, endpoint=True)[None, :]
        points = (
            edges[:, 0][:, None, :] + vector[:, None, :] * steps[:, :, None]
        )
        return points
    else:
        coords_ego = np.zeros((4, 2), dtype=np.float32)
        coords_ego[0] = edge1[1]
        coords_ego[1] = edge2[1]
        coords_ego[2] = edge2[0]
        coords_ego[3] = edge1[0]
        rotated_rectangle = _get_rotated_rectangle(coords_ego)
        return rotated_rectangle


def _get_rotated_rectangle(coords):
    center = coords.mean(axis=0)
    x, y = center
    edges, lengths = [], []
    for i in range(4):
        p0 = coords[i]
        p1 = coords[(i + 1) % 4]
        edge = p1 - p0
        length = np.linalg.norm(edge)
        edges.append(edge)
        lengths.append(length)
    edges = np.array(edges)
    lengths = np.array(lengths)
    long_idx = np.argmax(lengths)
    direction = edges[long_idx]
    w = lengths[long_idx]
    h = lengths[(long_idx + 1) % 4]
    yaw = np.arctan2(direction[1], direction[0])
    if yaw < -np.pi / 2:
        yaw += np.pi
    if yaw > np.pi / 2:
        yaw -= np.pi
    return np.array([x, y, w, h, yaw])


def discrete_map_process(
    query_xy,
    map_api,
    map_pc_range,
    anchor_ego_state,
    discrete_map_to_idx,
    max_points=None,
    return_rotated_rectangle=True,
):
    map_objects = map_api.get_proximal_map_objects(
        query_xy,
        map_pc_range,
        [
            SemanticMapLayer.STOP_LINE,
            SemanticMapLayer.CROSSWALK,
        ],
    )
    crosswalk_objects = map_objects[SemanticMapLayer.CROSSWALK]
    stoplane_objects = map_objects[SemanticMapLayer.STOP_LINE]

    M = len(stoplane_objects) + len(crosswalk_objects)
    # (M, 6): [x, y, w, h, yaw, cat_id], cat_id: 0=stop_line, 1=crosswalk
    rotated_bbox = np.zeros((M, 6), dtype=np.float32)

    # stop lines: cat_id=0
    for i, stop_lane in enumerate(stoplane_objects):
        sample_points = max_points["STOP_LINE"]
        stop_line_bbox = _get_discrete_map_edges(
            stop_lane,
            sample_points + 1,
            anchor_ego_state,
            return_rotated_rectangle,
        )
        rotated_bbox[i, :5] = stop_line_bbox
        rotated_bbox[i, 5] = discrete_map_to_idx["stop_line"]

    # crosswalks: cat_id=1
    for i, crosswalk in enumerate(crosswalk_objects):
        sample_points = max_points["CROSSWALK"]
        crosswalk_bbox = _get_discrete_map_edges(
            crosswalk,
            sample_points + 1,
            anchor_ego_state,
            return_rotated_rectangle,
        )
        rotated_bbox[len(stoplane_objects) + i, :5] = crosswalk_bbox
        rotated_bbox[len(stoplane_objects) + i, 5] = discrete_map_to_idx[
            "crosswalk"
        ]
    if rotated_bbox.shape[0] == 0:
        return []
    else:
        return torch.from_numpy(rotated_bbox)


# Module-level subclass so dataset pickle (DataLoader workers / registry
# clone) can resolve CustomNuPlanMap by name. Nested classes are unpickleable.
if NuPlanMap is not None:

    class CustomNuPlanMap(NuPlanMap):
        def get_proximal_map_objects(self, point, map_pc_range, layers):
            """Inherited, see superclass."""
            x_min, x_max = (
                point.x + map_pc_range[0],
                point.x + map_pc_range[3],
            )
            y_min, y_max = (
                point.y + map_pc_range[1],
                point.y + map_pc_range[4],
            )
            patch = geom.box(x_min, y_min, x_max, y_max)

            supported_layers = self.get_available_map_objects()
            unsupported_layers = [
                layer for layer in layers if layer not in supported_layers
            ]

            assert (
                len(unsupported_layers) == 0
            ), f"Object representation for layer(s): {unsupported_layers} is unavailable"  # noqa

            object_map = defaultdict(list)

            for layer in layers:
                object_map[layer] = self._get_proximity_map_object(
                    patch, layer
                )

            return object_map

else:
    # Avoid ``class CustomNuPlanMap(None)`` -> TypeError: NoneType takes no arguments
    CustomNuPlanMap = None
