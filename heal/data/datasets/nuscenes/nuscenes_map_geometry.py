# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
"""Map geometry helpers shared by map dataset and map GT pack."""

import os
from typing import List, Optional, Sequence

import cv2
import numpy as np
import torch

try:
    from nuscenes.eval.common.utils import Quaternion, quaternion_yaw
    from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
except ImportError as e:  # pragma: no cover - surfaces the real failing import
    raise ImportError(
        "nuscenes-devkit is required for map vector GT geometry "
        "(VectorizedLocalMap). The previous code silently set "
        "Quaternion=None here, which later crashed as "
        "'NoneType is not callable' inside the DataLoader worker. "
        "Install nuscenes-devkit==1.1.9 with a compatible numpy/pandas."
    ) from e

try:
    import geopandas as gpd
except ImportError as e:  # pragma: no cover - surfaces the real failing import
    raise ImportError(
        "geopandas (with fiona / Shapely / pyproj) is required for map "
        "vector GT geometry (VectorizedLocalMap). Install geopandas==0.10.2 "
        "with a compatible numpy/pandas."
    ) from e

from hat_engine.utils.apply_func import convert_tensor_v2 as to_tensor
from shapely import affinity, ops
from shapely.geometry import LineString, MultiLineString, MultiPolygon, box

from heal.data.datasets.nuscenes.map_utils.geo_opensfm import (
    TopocentricConverter,
)

__all__ = ["LiDARInstanceLines", "VectorizedLocalMap"]

PV_PROJ_EGO_GROUND_Z = -1.6


def perspective(cam_coords, proj_mat):
    pix_coords = proj_mat @ cam_coords
    valid_idx = pix_coords[2, :] > 0
    pix_coords = pix_coords[:, valid_idx]
    pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + 1e-7)
    pix_coords = pix_coords.transpose(1, 0)
    return pix_coords


def line_ego_to_pvmask(
    line_ego,
    mask,
    lidar2feat,
    color=1,
    thickness=1,
    z=PV_PROJ_EGO_GROUND_Z,
):
    """Rasterize a map line onto a PV feature mask."""
    distances = np.linspace(0, line_ego.length, 200)
    coords = np.array(
        [list(line_ego.interpolate(distance).coords) for distance in distances]
    ).reshape(-1, 2)
    pts_num = coords.shape[0]
    zeros = np.zeros((pts_num, 1))
    zeros[:] = z
    ones = np.ones((pts_num, 1))
    lidar_coords = np.concatenate([coords, zeros, ones], axis=1).transpose(
        1, 0
    )
    pix_coords = perspective(lidar_coords, lidar2feat)
    cv2.polylines(
        mask,
        np.int32([pix_coords]),
        False,
        color=color,
        thickness=thickness,
    )


class LiDARInstanceLines(object):
    """LiDARInstanceLines.

    Args:
        instance_line_list: List of instance lines.
        sample_dist: Distance between samples. Default is 1.
        num_samples: Number of samples. Default is 250.
        padding: Whether to pad the samples. Default is False.
        fixed_num: Fixed number of samples. Default is -1.
        padding_value: Value to use for padding. Default is -10000.
        patch_size: Size of the patch. Default is None.
    """

    def __init__(
        self,
        instance_line_list: list,
        sample_dist: int = 1,
        num_samples: int = 250,
        padding: bool = False,
        fixed_num: int = -1,
        padding_value: int = -10000,
        patch_size: int = None,
        pc_range: Optional[Sequence[float]] = None,
    ):
        assert isinstance(instance_line_list, list)
        # assert patch_size is not None or pc_range is not None
        if len(instance_line_list) != 0:
            assert isinstance(instance_line_list[0], LineString)
        if patch_size is not None:
            self.patch_size = patch_size
            self.max_x = self.patch_size[1] / 2
            self.max_y = self.patch_size[0] / 2
            self.min_x = -self.max_x
            self.min_y = -self.max_y
        else:
            assert (
                pc_range is not None
            ), "pc_range should be provided if patch_size is None"
            self.min_x = pc_range[0]
            self.max_x = pc_range[3]
            self.min_y = pc_range[1]
            self.max_y = pc_range[4]

        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.fixed_num = fixed_num
        self.padding_value = padding_value

        self.instance_list = instance_line_list

    @property
    def bbox(self):
        """Return Tensor([N,4]), in xmin, ymin, xmax, ymax form."""
        assert len(self.instance_list) != 0
        instance_bbox_list = []
        for instance in self.instance_list:
            # bounds is bbox: [xmin, ymin, xmax, ymax]
            instance_bbox_list.append(instance.bounds)
        instance_bbox_array = np.array(instance_bbox_list)
        instance_bbox_tensor = to_tensor(instance_bbox_array)
        instance_bbox_tensor = instance_bbox_tensor.to(dtype=torch.float32)
        instance_bbox_tensor[:, 0] = torch.clamp(
            instance_bbox_tensor[:, 0], min=self.min_x, max=self.max_x
        )
        instance_bbox_tensor[:, 1] = torch.clamp(
            instance_bbox_tensor[:, 1], min=self.min_y, max=self.max_y
        )
        instance_bbox_tensor[:, 2] = torch.clamp(
            instance_bbox_tensor[:, 2], min=self.min_x, max=self.max_x
        )
        instance_bbox_tensor[:, 3] = torch.clamp(
            instance_bbox_tensor[:, 3], min=self.min_y, max=self.max_y
        )
        return instance_bbox_tensor

    @property
    def fixed_num_sampled_points(self):
        """Return Tensor([N,fixed_num,2]), in xmin, ymin, xmax, ymax form."""
        assert len(self.instance_list) != 0
        instance_points_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            sampled_points = np.array(
                [
                    list(instance.interpolate(distance).coords)
                    for distance in distances
                ]
            ).reshape(-1, 2)
            instance_points_list.append(sampled_points)
        instance_points_array = np.array(instance_points_list)
        instance_points_tensor = to_tensor(instance_points_array)
        instance_points_tensor = instance_points_tensor.to(dtype=torch.float32)
        instance_points_tensor[:, :, 0] = torch.clamp(
            instance_points_tensor[:, :, 0], min=self.min_x, max=self.max_x
        )
        instance_points_tensor[:, :, 1] = torch.clamp(
            instance_points_tensor[:, :, 1], min=self.min_y, max=self.max_y
        )
        return instance_points_tensor

    @property
    def shift_fixed_num_sampled_points(self):
        """Return  [instances_num, num_shifts, fixed_num, 2]."""
        fixed_num_sampled_points = self.fixed_num_sampled_points
        instances_list = []
        is_poly = False
        # is_line = False
        # import pdb;pdb.set_trace()
        for fixed_num_pts in fixed_num_sampled_points:
            # [fixed_num, 2]
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            fixed_num = fixed_num_pts.shape[0]
            shift_pts_list = []
            if is_poly:
                # import pdb;pdb.set_trace()
                for shift_right_i in range(fixed_num):
                    shift_pts_list.append(fixed_num_pts.roll(shift_right_i, 0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list, dim=0)

            shift_pts[:, :, 0] = torch.clamp(
                shift_pts[:, :, 0], min=self.min_x, max=self.max_x
            )
            shift_pts[:, :, 1] = torch.clamp(
                shift_pts[:, :, 1], min=self.min_y, max=self.max_y
            )

            if not is_poly:
                padding = torch.full(
                    [fixed_num - shift_pts.shape[0], fixed_num, 2],
                    self.padding_value,
                )
                shift_pts = torch.cat([shift_pts, padding], dim=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v1(self):
        """Return  [instances_num, num_shifts, fixed_num, 2]."""
        fixed_num_sampled_points = self.fixed_num_sampled_points
        instances_list = []
        is_poly = False
        # is_line = False
        # import pdb;pdb.set_trace()
        for fixed_num_pts in fixed_num_sampled_points:
            # [fixed_num, 2]
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            pts_num = fixed_num_pts.shape[0]
            shift_num = pts_num - 1
            if is_poly:
                pts_to_shift = fixed_num_pts[:-1, :]
            shift_pts_list = []
            if is_poly:
                for shift_right_i in range(shift_num):
                    shift_pts_list.append(pts_to_shift.roll(shift_right_i, 0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list, dim=0)

            if is_poly:
                _, _, num_coords = shift_pts.shape
                tmp_shift_pts = shift_pts.new_zeros(
                    (shift_num, pts_num, num_coords)
                )
                tmp_shift_pts[:, :-1, :] = shift_pts
                tmp_shift_pts[:, -1, :] = shift_pts[:, 0, :]
                shift_pts = tmp_shift_pts

            shift_pts[:, :, 0] = torch.clamp(
                shift_pts[:, :, 0], min=self.min_x, max=self.max_x
            )
            shift_pts[:, :, 1] = torch.clamp(
                shift_pts[:, :, 1], min=self.min_y, max=self.max_y
            )

            if not is_poly:
                padding = torch.full(
                    [shift_num - shift_pts.shape[0], pts_num, 2],
                    self.padding_value,
                )
                shift_pts = torch.cat([shift_pts, padding], dim=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v2(self):
        """Return  [instances_num, num_shifts, fixed_num, 2]."""
        assert len(self.instance_list) != 0
        instances_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            poly_pts = np.array(list(instance.coords))
            start_pts = poly_pts[0]
            end_pts = poly_pts[-1]
            is_poly = np.equal(start_pts, end_pts)
            is_poly = is_poly.all()
            shift_pts_list = []
            pts_num, coords_num = poly_pts.shape
            shift_num = pts_num - 1
            final_shift_num = self.fixed_num - 1
            if is_poly:
                pts_to_shift = poly_pts[:-1, :]
                for shift_right_i in range(shift_num):
                    shift_pts = np.roll(pts_to_shift, shift_right_i, axis=0)
                    pts_to_concat = shift_pts[0]
                    pts_to_concat = np.expand_dims(pts_to_concat, axis=0)
                    shift_pts = np.concatenate(
                        (shift_pts, pts_to_concat), axis=0
                    )
                    shift_instance = LineString(shift_pts)
                    shift_sampled_points = np.array(
                        [
                            list(shift_instance.interpolate(distance).coords)
                            for distance in distances
                        ]
                    ).reshape(-1, 2)
                    shift_pts_list.append(shift_sampled_points)
                # import pdb;pdb.set_trace()
            else:
                sampled_points = np.array(
                    [
                        list(instance.interpolate(distance).coords)
                        for distance in distances
                    ]
                ).reshape(-1, 2)
                flip_sampled_points = np.flip(sampled_points, axis=0)
                shift_pts_list.append(sampled_points)
                shift_pts_list.append(flip_sampled_points)

            multi_shifts_pts = np.stack(shift_pts_list, axis=0)
            shifts_num, _, _ = multi_shifts_pts.shape

            if shifts_num > final_shift_num:
                index = np.random.choice(
                    multi_shifts_pts.shape[0], final_shift_num, replace=False
                )
                multi_shifts_pts = multi_shifts_pts[index]

            multi_shifts_pts_tensor = to_tensor(multi_shifts_pts)
            multi_shifts_pts_tensor = multi_shifts_pts_tensor.to(
                dtype=torch.float32
            )

            multi_shifts_pts_tensor[:, :, 0] = torch.clamp(
                multi_shifts_pts_tensor[:, :, 0],
                min=self.min_x,
                max=self.max_x,
            )
            multi_shifts_pts_tensor[:, :, 1] = torch.clamp(
                multi_shifts_pts_tensor[:, :, 1],
                min=self.min_y,
                max=self.max_y,
            )
            # if not is_poly:
            if multi_shifts_pts_tensor.shape[0] < final_shift_num:
                padding = torch.full(
                    [
                        final_shift_num - multi_shifts_pts_tensor.shape[0],
                        self.fixed_num,
                        2,
                    ],
                    self.padding_value,
                )
                multi_shifts_pts_tensor = torch.cat(
                    [multi_shifts_pts_tensor, padding], dim=0
                )
            instances_list.append(multi_shifts_pts_tensor)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v3(self):
        """Return  [instances_num, num_shifts, fixed_num, 2]."""
        assert len(self.instance_list) != 0
        instances_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            poly_pts = np.array(list(instance.coords))
            start_pts = poly_pts[0]
            end_pts = poly_pts[-1]
            is_poly = np.equal(start_pts, end_pts)
            is_poly = is_poly.all()
            shift_pts_list = []
            pts_num, coords_num = poly_pts.shape
            shift_num = pts_num - 1
            final_shift_num = self.fixed_num - 1
            if is_poly:
                pts_to_shift = poly_pts[:-1, :]
                for shift_right_i in range(shift_num):
                    shift_pts = np.roll(pts_to_shift, shift_right_i, axis=0)
                    pts_to_concat = shift_pts[0]
                    pts_to_concat = np.expand_dims(pts_to_concat, axis=0)
                    shift_pts = np.concatenate(
                        (shift_pts, pts_to_concat), axis=0
                    )
                    shift_instance = LineString(shift_pts)
                    shift_sampled_points = np.array(
                        [
                            list(shift_instance.interpolate(distance).coords)
                            for distance in distances
                        ]
                    ).reshape(-1, 2)
                    shift_pts_list.append(shift_sampled_points)
                flip_pts_to_shift = np.flip(pts_to_shift, axis=0)
                for shift_right_i in range(shift_num):
                    shift_pts = np.roll(
                        flip_pts_to_shift, shift_right_i, axis=0
                    )
                    pts_to_concat = shift_pts[0]
                    pts_to_concat = np.expand_dims(pts_to_concat, axis=0)
                    shift_pts = np.concatenate(
                        (shift_pts, pts_to_concat), axis=0
                    )
                    shift_instance = LineString(shift_pts)
                    shift_sampled_points = np.array(
                        [
                            list(shift_instance.interpolate(distance).coords)
                            for distance in distances
                        ]
                    ).reshape(-1, 2)
                    shift_pts_list.append(shift_sampled_points)
            else:
                sampled_points = np.array(
                    [
                        list(instance.interpolate(distance).coords)
                        for distance in distances
                    ]
                ).reshape(-1, 2)
                flip_sampled_points = np.flip(sampled_points, axis=0)
                shift_pts_list.append(sampled_points)
                shift_pts_list.append(flip_sampled_points)

            multi_shifts_pts = np.stack(shift_pts_list, axis=0)
            shifts_num, _, _ = multi_shifts_pts.shape
            if shifts_num > 2 * final_shift_num:
                index = np.random.choice(
                    shift_num, final_shift_num, replace=False
                )
                flip0_shifts_pts = multi_shifts_pts[index]
                flip1_shifts_pts = multi_shifts_pts[index + shift_num]
                multi_shifts_pts = np.concatenate(
                    (flip0_shifts_pts, flip1_shifts_pts), axis=0
                )

            multi_shifts_pts_tensor = to_tensor(multi_shifts_pts)
            multi_shifts_pts_tensor = multi_shifts_pts_tensor.to(
                dtype=torch.float32
            )

            multi_shifts_pts_tensor[:, :, 0] = torch.clamp(
                multi_shifts_pts_tensor[:, :, 0],
                min=self.min_x,
                max=self.max_x,
            )
            multi_shifts_pts_tensor[:, :, 1] = torch.clamp(
                multi_shifts_pts_tensor[:, :, 1],
                min=self.min_y,
                max=self.max_y,
            )
            if multi_shifts_pts_tensor.shape[0] < 2 * final_shift_num:
                padding = torch.full(
                    [
                        final_shift_num * 2 - multi_shifts_pts_tensor.shape[0],
                        self.fixed_num,
                        2,
                    ],
                    self.padding_value,
                )
                multi_shifts_pts_tensor = torch.cat(
                    [multi_shifts_pts_tensor, padding], dim=0
                )
            instances_list.append(multi_shifts_pts_tensor)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v4(self):
        """Return  [instances_num, num_shifts, fixed_num, 2]."""
        fixed_num_sampled_points = self.fixed_num_sampled_points
        instances_list = []
        is_poly = False
        for fixed_num_pts in fixed_num_sampled_points:
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            pts_num = fixed_num_pts.shape[0]
            shift_num = pts_num - 1
            shift_pts_list = []
            if is_poly:
                pts_to_shift = fixed_num_pts[:-1, :]
                for shift_right_i in range(shift_num):
                    shift_pts_list.append(pts_to_shift.roll(shift_right_i, 0))
                flip_pts_to_shift = pts_to_shift.flip(0)
                for shift_right_i in range(shift_num):
                    shift_pts_list.append(
                        flip_pts_to_shift.roll(shift_right_i, 0)
                    )
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list, dim=0)

            if is_poly:
                _, _, num_coords = shift_pts.shape
                tmp_shift_pts = shift_pts.new_zeros(
                    (shift_num * 2, pts_num, num_coords)
                )
                tmp_shift_pts[:, :-1, :] = shift_pts
                tmp_shift_pts[:, -1, :] = shift_pts[:, 0, :]
                shift_pts = tmp_shift_pts

            shift_pts[:, :, 0] = torch.clamp(
                shift_pts[:, :, 0], min=self.min_x, max=self.max_x
            )
            shift_pts[:, :, 1] = torch.clamp(
                shift_pts[:, :, 1], min=self.min_y, max=self.max_y
            )

            if not is_poly:
                padding = torch.full(
                    [shift_num * 2 - shift_pts.shape[0], pts_num, 2],
                    self.padding_value,
                )
                shift_pts = torch.cat([shift_pts, padding], dim=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_sparsedrive(self, padding=1e5):
        """Return  [instances_num, 2*(fixed_num-1), fixed_num, 2]."""
        assert len(self.instance_list) != 0
        instances_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            line = np.array(
                [
                    list(instance.interpolate(distance).coords)
                    for distance in distances
                ]
            ).squeeze()
            is_closed = np.allclose(line[0], line[-1], atol=1e-3)
            # num_points = len(line)
            num_points, coords_dim = line.shape
            permute_num = num_points - 1
            permute_lines_list = []
            if is_closed:
                pts_to_permute = line[
                    :-1, :
                ]  # throw away replicate start end pts
                for shift_i in range(permute_num):
                    permute_lines_list.append(
                        np.roll(pts_to_permute, shift_i, axis=0)
                    )
                flip_pts_to_permute = np.flip(pts_to_permute, axis=0)
                for shift_i in range(permute_num):
                    permute_lines_list.append(
                        np.roll(flip_pts_to_permute, shift_i, axis=0)
                    )
            else:
                permute_lines_list.append(line)
                permute_lines_list.append(np.flip(line, axis=0))

            permute_lines_array = np.stack(permute_lines_list, axis=0)

            if is_closed:
                tmp = np.zeros((permute_num * 2, num_points, coords_dim))
                tmp[:, :-1, :] = permute_lines_array
                tmp[:, -1, :] = permute_lines_array[
                    :, 0, :
                ]  # add replicate start end pts
                permute_lines_array = tmp

            else:
                # padding
                padding = np.full(
                    [permute_num * 2 - 2, num_points, coords_dim], padding
                )
                permute_lines_array = np.concatenate(
                    (permute_lines_array, padding), axis=0
                )

            instances_list.append(to_tensor(permute_lines_array))
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(dtype=torch.float32)
        return instances_tensor


class VectorizedLocalMap(object):
    """VectorizedLocalMap.

    Args:
        dataroot: Root directory of the dataset.
        patch_size: Size of the patch.
        map_classes: Tuple of map classes. Default is ("divider",
            "ped_crossing", "boundary").
        line_classes: Tuple of line classes. Default is ("road_divider",
            "lane_divider").
        ped_crossing_classes: Tuple of pedestrian crossing classes. Default is
            ("ped_crossing").
        contour_classes: Tuple of contour classes. Default is ("road_segment",
            "lane").
        sample_dist: Distance between samples. Default is 1.
        num_samples: Number of samples. Default is 250.
        padding: Whether to pad the samples. Default is False.
        fixed_ptsnum_per_line: Fixed number of points per line. Default is -1.
        padding_value: Value to use for padding. Default is -10000.
        thickness: Thickness of the lines. Default is 3.
        canvas_size: Size of the canvas. Default is (200, 200).
        aux_seg: Auxiliary segmentation information. Default is None.
        sd_map_path: Path to the SD map. Default is None.
        osm_thickness: Thickness of the lines for OSM (OpenStreetMap) features.
            Default is 5.
    """

    CLASS2LABEL = {
        "road_divider": 0,
        "lane_divider": 0,
        "ped_crossing": 1,
        "contours": 2,
        "others": -1,
    }

    def __init__(
        self,
        dataroot: str,
        patch_size: int,
        map_classes: tuple = ("divider", "ped_crossing", "boundary"),
        line_classes: tuple = ("road_divider", "lane_divider"),
        ped_crossing_classes: tuple = ("ped_crossing",),
        contour_classes: tuple = ("road_segment", "lane"),
        sample_dist: int = 1,
        num_samples: int = 250,
        padding: bool = False,
        fixed_ptsnum_per_line: int = -1,
        padding_value: int = -10000,
        thickness: int = 3,
        canvas_size: tuple = (200, 200),
        aux_seg: any = None,
        sd_map_path: Optional[str] = None,
        osm_thickness: int = 5,
    ):
        super().__init__()
        self.data_root = dataroot
        self.MAPS = [
            "boston-seaport",
            "singapore-hollandvillage",
            "singapore-onenorth",
            "singapore-queenstown",
        ]
        self.vec_classes = map_classes
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.polygon_classes = contour_classes
        self.nusc_maps = {}
        self.map_explorer = {}
        for loc in self.MAPS:
            self.nusc_maps[loc] = NuScenesMap(
                dataroot=self.data_root, map_name=loc
            )
            self.map_explorer[loc] = NuScenesMapExplorer(self.nusc_maps[loc])

        self.patch_size = patch_size
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.fixed_num = fixed_ptsnum_per_line
        self.padding_value = padding_value

        # for semantic mask
        self.canvas_size = canvas_size
        self.thickness = thickness
        self.scale_x = self.canvas_size[1] / self.patch_size[1]
        self.scale_y = self.canvas_size[0] / self.patch_size[0]
        self.aux_seg = aux_seg

        # sdmap
        self.osm_thickness = osm_thickness
        self.sd_maps = None
        if sd_map_path is not None:
            self.sd_maps = {}
            options = [
                "trunk",
                "primary",
                "secondary",
                "tertiary",
                "unclassified",
                "residential",
                "trunk_link",
                "primary_link",
                "secondary_link",
                "tertiary_link" "living_street",
                "road",
            ]  # osm(OpenStreetMap) road categories
            map_origin = {
                "boston-seaport": (
                    42.336849169438615,
                    -71.05785369873047,
                    0.0,
                ),
                "singapore-onenorth": (
                    1.2882100868743724,
                    103.78475189208984,
                    0.0,
                ),
                "singapore-hollandvillage": (
                    1.2993652317780957,
                    103.78217697143555,
                    0.0,
                ),
                "singapore-queenstown": (
                    1.2782562240223188,
                    103.76741409301758,
                    0.0,
                ),
            }  # coordinates of osm areas in nuscenes dataset

            for loc in self.MAPS:
                lat, lon, alt = map_origin[loc]
                sd_map = gpd.read_file(
                    os.path.join(sd_map_path, "{}.shp".format(loc))
                )
                converter = TopocentricConverter(lat, lon, alt)
                sd_map = sd_map[sd_map["type"].isin(options)]
                sd_map_topo_list = []
                for _, row in sd_map.iterrows():
                    tmp_sd_data = list(row.geometry.coords)
                    tmp_sd_data_topo = [
                        converter.to_topocentric(lonlat[1], lonlat[0], 0.0)[:2]
                        for lonlat in tmp_sd_data
                    ]
                    sd_map_topo_list.append(tmp_sd_data_topo)
                self.sd_maps[loc] = MultiLineString(sd_map_topo_list)

    def gen_polygon_elements_with_bbox(
        self,
        location,
        homo_translation,
        homo_rotation,
        bbox_classes,
        with_yaw=False,
    ):
        """Use coords2global to get gt stopline bbox."""

        map_pose = homo_translation[:2]
        rotation = Quaternion(homo_rotation)

        patch_box = (
            map_pose[0],
            map_pose[1],
            self.patch_size[0],
            self.patch_size[1],
        )
        patch_angle = quaternion_yaw(rotation) / np.pi * 180

        bbox_infos = {}
        for vec_class in bbox_classes:
            if vec_class == "stop_line":
                polygon_geom = []
                geoms = self.get_contour_line(
                    patch_box, patch_angle, vec_class, location
                )
                polygon_geom.append((vec_class, geoms))
                bbox_infos[vec_class] = self.polygon_to_bbox(
                    polygon_geom, with_yaw
                )
            elif vec_class == "ped_crossing":
                map_geom = []
                geoms = self.get_ped_crossing_line(
                    patch_box, patch_angle, location
                )
                map_geom.append((vec_class, geoms))
                bbox_infos[vec_class] = self.polygon_to_bbox(
                    polygon_geom, with_yaw
                )
            else:
                raise ValueError("Invalid bbox_classes: %s" % vec_class)
        return bbox_infos

    def polygon_to_bbox(self, polygon_geom, with_yaw=False):
        bboxes = []
        MultiPolygons = polygon_geom[0][1]
        for MP in MultiPolygons:
            if MP.geom_type != "MultiPolygon":
                continue
            for poly in MP.geoms:
                coords = torch.tensor(
                    poly.exterior.coords, dtype=torch.float32
                )
                if not poly.is_valid or poly.area < 1e-3:
                    continue
                if coords.shape[0] <= 4:
                    continue

                if not with_yaw:
                    x = coords[:, 0]
                    y = coords[:, 1]
                    x_min, x_max = min(x), max(x)
                    y_min, y_max = min(y), max(y)
                    bbox = torch.Tensor(
                        [
                            (x_max + x_min) / 2,
                            (y_max + y_min) / 2,
                            (x_max - x_min),
                            (y_max - y_min),
                        ]
                    )
                    if bbox[2] <= 0.15 or bbox[3] <= 1:
                        continue
                    bboxes.append(bbox)
                else:
                    obb = poly.minimum_rotated_rectangle
                    rect_pts = np.array(obb.exterior.coords)[:-1]
                    center = rect_pts.mean(axis=0)
                    x, y = center
                    edges, lengths = [], []
                    for i in range(4):
                        p0 = rect_pts[i]
                        p1 = rect_pts[(i + 1) % 4]
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
                    if h <= 0.15 or w <= 1:
                        continue
                    yaw = np.arctan2(direction[1], direction[0])
                    if yaw < -np.pi / 2:
                        yaw += np.pi
                    if yaw > np.pi / 2:
                        yaw -= np.pi
                    bboxes.append(torch.Tensor([x, y, w, h, yaw]))
        return bboxes

    def gen_vectorized_samples(
        self,
        location,
        homo_translation,
        homo_rotation,
        img_shape,
        homo2img,
        num_cam=6,
        pv_proj_z: float = PV_PROJ_EGO_GROUND_Z,
    ):
        """Use coords2global to get gt map layers."""

        map_pose = homo_translation[:2]
        rotation = Quaternion(homo_rotation)

        patch_box = (
            map_pose[0],
            map_pose[1],
            self.patch_size[0],
            self.patch_size[1],
        )
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        vectors = []
        for vec_class in self.vec_classes:
            if vec_class == "divider":
                line_geom = self.get_map_geom(
                    patch_box, patch_angle, self.line_classes, location
                )
                line_instances_dict = self.line_geoms_to_instances(line_geom)
                for line_type, instances in line_instances_dict.items():
                    for instance in instances:
                        vectors.append(
                            (
                                LineString(np.array(instance.coords)),
                                self.CLASS2LABEL.get(line_type, -1),
                            )
                        )
            elif vec_class == "ped_crossing":
                ped_geom = self.get_map_geom(
                    patch_box, patch_angle, self.ped_crossing_classes, location
                )
                ped_instance_list = self.ped_poly_geoms_to_instances(ped_geom)
                for instance in ped_instance_list:
                    vectors.append(
                        (
                            LineString(np.array(instance.coords)),
                            self.CLASS2LABEL.get("ped_crossing", -1),
                        )
                    )
            elif vec_class == "boundary":
                polygon_geom = self.get_map_geom(
                    patch_box, patch_angle, self.polygon_classes, location
                )
                poly_bound_list = self.poly_geoms_to_instances(polygon_geom)
                for instance in poly_bound_list:
                    vectors.append(
                        (
                            LineString(np.array(instance.coords)),
                            self.CLASS2LABEL.get("contours", -1),
                        )
                    )
            else:
                raise ValueError(f"WRONG vec_class: {vec_class}")

        gt_labels = []
        gt_instance = []
        if self.aux_seg is not None:
            if self.aux_seg["bev_seg"]:
                gt_semantic_mask = np.zeros(
                    (1, self.canvas_size[0], self.canvas_size[1]),
                    dtype=np.uint8,
                )
            else:
                gt_semantic_mask = None
            # import ipdb;ipdb.set_trace()
            if self.aux_seg["pv_seg"] and not self.aux_seg.get(
                "defer_pv_seg_rasterize"
            ):
                num_cam = num_cam
                img_shape = img_shape
                gt_pv_semantic_masks = []
                lidar2feats = []
                for _, stride in self.aux_seg["feat_down_sample"]:
                    feat_down_sample = stride
                    gt_pv_semantic_mask = np.zeros(
                        (
                            num_cam,
                            1,
                            img_shape[0] // feat_down_sample,
                            img_shape[1] // feat_down_sample,
                        ),
                        dtype=np.uint8,
                    )
                    homo2img = homo2img
                    scale_factor = np.eye(4)
                    scale_factor[0, 0] *= 1 / feat_down_sample
                    scale_factor[1, 1] *= 1 / feat_down_sample
                    lidar2feat = [scale_factor @ l2i for l2i in homo2img]
                    gt_pv_semantic_masks.append(gt_pv_semantic_mask)
                    lidar2feats.append(lidar2feat)
            else:
                gt_pv_semantic_masks = None

            for instance, type in vectors:
                if type != -1:
                    gt_instance.append(instance)
                    gt_labels.append(type)
                    if instance.geom_type == "LineString":
                        if self.aux_seg["bev_seg"]:
                            self.line_ego_to_mask(
                                instance,
                                gt_semantic_mask[0],
                                color=1,
                                thickness=self.thickness,
                            )
                        if self.aux_seg["pv_seg"] and not self.aux_seg.get(
                            "defer_pv_seg_rasterize"
                        ):
                            for i, (
                                gt_pv_semantic_mask,
                                lidar2feat,
                            ) in enumerate(
                                zip(gt_pv_semantic_masks, lidar2feats)
                            ):
                                for cam_index in range(num_cam):
                                    self.line_ego_to_pvmask(
                                        instance,
                                        gt_pv_semantic_mask[cam_index][0],
                                        lidar2feat[cam_index],
                                        color=1,
                                        thickness=self.aux_seg["pv_thickness"],
                                        z=pv_proj_z,
                                    )
                                gt_pv_semantic_masks[i] = gt_pv_semantic_mask
                    else:
                        print(instance.geom_type)

        else:
            for instance, type in vectors:
                if type != -1:
                    gt_instance.append(instance)
                    gt_labels.append(type)
            gt_semantic_mask = None
            gt_pv_semantic_masks = None

        gt_instance = LiDARInstanceLines(
            gt_instance,
            self.sample_dist,
            self.num_samples,
            self.padding,
            self.fixed_num,
            self.padding_value,
            patch_size=self.patch_size,
        )

        # sdmap
        if self.sd_maps is not None:
            osm_geom = self.get_osm_geom(patch_box, patch_angle, location)
            osm_vector_list = []

            for line in osm_geom:
                if not line.is_empty:
                    if line.geom_type == "MultiLineString":
                        for single_line in line.geoms:
                            pts, pts_num = self.sample_fixed_pts_from_line(
                                single_line, padding=False, fixed_num=50
                            )
                            if pts_num >= 2:
                                osm_vector_list.append(LineString(pts))
                    elif line.geom_type == "LineString":
                        pts, pts_num = self.sample_fixed_pts_from_line(
                            line, padding=False, fixed_num=50
                        )
                        if pts_num >= 2:
                            osm_vector_list.append(LineString(pts))
                    else:
                        raise NotImplementedError

            local_box = (0.0, 0.0, self.patch_size[0], self.patch_size[1])
            osm_mask = self.line_osms_to_mask(
                osm_vector_list,
                local_box,
                self.canvas_size,
                thickness=self.osm_thickness,
            )
            osm_vectors = LiDARInstanceLines(
                osm_vector_list,
                self.sample_dist,
                self.num_samples,
                self.padding,
                self.fixed_num,
                self.padding_value,
                patch_size=self.patch_size,
            )

        anns_results = {
            "gt_vecs_pts_loc": gt_instance,
            "gt_vecs_label": gt_labels,
            "gt_semantic_mask": gt_semantic_mask,
            "gt_pv_semantic_mask": gt_pv_semantic_masks,
            "osm_vectors": osm_vectors if self.sd_maps is not None else None,
            "osm_mask": osm_mask if self.sd_maps is not None else None,
        }
        return anns_results

    def line_ego_to_pvmask(
        self,
        line_ego,
        mask,
        lidar2feat,
        color=1,
        thickness=1,
        z=PV_PROJ_EGO_GROUND_Z,
    ):
        """Rasterize a map line onto a PV feature mask."""
        line_ego_to_pvmask(
            line_ego, mask, lidar2feat, color=color, thickness=thickness, z=z
        )

    def line_ego_to_mask(
        self,
        line_ego: LineString,
        mask: np.ndarray,
        color: int = 1,
        thickness: int = 3,
    ):
        """Rasterize a single line to mask.

        Args:
            line_ego: LineString representing the line.
            mask: Semantic mask to paint on.
            color: Positive label. Default is 1.
            thickness: Thickness of rasterized lines. Default is 3.
        """

        trans_x = self.canvas_size[1] / 2
        trans_y = self.canvas_size[0] / 2
        line_ego = affinity.scale(
            line_ego, self.scale_x, self.scale_y, origin=(0, 0)
        )
        line_ego = affinity.affine_transform(
            line_ego, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y]
        )
        # print(np.array(list(line_ego.coords), dtype=np.int32).shape)
        coords = np.array(list(line_ego.coords), dtype=np.int32)[:, :2]
        coords = coords.reshape((-1, 2))
        assert len(coords) >= 2

        cv2.polylines(
            mask, np.int32([coords]), False, color=color, thickness=thickness
        )

    def line_osms_to_mask(
        self,
        lines: List[LineString],
        local_box: Sequence[float],
        canvas_size: Sequence[int],
        color: int = 1,
        thickness: int = 5,
    ):
        """Convert OSM lines to a mask image with specified thickness and color.

        Args:
            lines: List of OSM line geometries to be converted to a mask.
            local_box: Coordinates of the local box.
            canvas_size: Size of the canvas for the output mask.
            color: Color of the lines in the mask. Default is 1.
            thickness: Thickness of the lines in pixels. Default is 5.

        Return:
            A mask array with the OSM lines drawn.
        """

        patch_x, patch_y, patch_h, patch_w = local_box
        patch = NuScenesMapExplorer.get_patch_coord(local_box)
        canvas_h = canvas_size[0]
        canvas_w = canvas_size[1]
        scale_height = canvas_h / patch_h
        scale_width = canvas_w / patch_w
        trans_x = -patch_x + patch_w / 2.0
        trans_y = -patch_y + patch_h / 2.0

        osm_mask = np.zeros(canvas_size, np.uint8)
        for line in lines:
            new_line = line.intersection(patch)
            if not new_line.is_empty:
                new_line = affinity.affine_transform(
                    new_line, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y]
                )
                new_line = affinity.scale(
                    new_line,
                    xfact=scale_width,
                    yfact=scale_height,
                    origin=(0, 0),
                )

                coords = np.array(list(new_line.coords), dtype=np.int32)
                coords = coords.reshape((-1, 2))
                assert len(coords) >= 2

                cv2.polylines(
                    osm_mask,
                    np.int32([coords]),
                    False,
                    color=color,
                    thickness=thickness,
                )
        osm_mask = osm_mask[np.newaxis, :, :]
        osm_mask = osm_mask.astype(np.int32)
        osm_mask[np.where(osm_mask == 0)] = -1
        osm_mask = osm_mask.astype(np.float32)
        return osm_mask

    def get_map_geom(self, patch_box, patch_angle, layer_names, location):
        map_geom = []
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                geoms = self.get_divider_line(
                    patch_box, patch_angle, layer_name, location
                )
                map_geom.append((layer_name, geoms))
            elif layer_name in self.polygon_classes:
                geoms = self.get_contour_line(
                    patch_box, patch_angle, layer_name, location
                )
                map_geom.append((layer_name, geoms))
            elif layer_name in self.ped_crossing_classes:
                geoms = self.get_ped_crossing_line(
                    patch_box, patch_angle, location
                )
                map_geom.append((layer_name, geoms))
        return map_geom

    def _one_type_line_geom_to_instances(self, line_geom):
        line_instances = []

        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == "MultiLineString":
                    for single_line in line.geoms:
                        line_instances.append(single_line)
                elif line.geom_type == "LineString":
                    line_instances.append(line)
                else:
                    raise NotImplementedError
        return line_instances

    def _local_patch_box(self, margin: float):
        """Axis-aligned patch; +margin expands, -margin shrinks."""
        half_h = self.patch_size[0] * 0.5
        half_w = self.patch_size[1] * 0.5
        return box(
            -half_w - margin,
            -half_h - margin,
            half_w + margin,
            half_h + margin,
        )

    @staticmethod
    def _as_multipolygon(geom):
        if geom.geom_type == "MultiPolygon":
            return geom
        return MultiPolygon([geom])

    @staticmethod
    def _collect_polygon_rings(union_geom):
        multipoly = VectorizedLocalMap._as_multipolygon(union_geom)
        outer_rings = []
        hole_rings = []
        for poly in multipoly.geoms:
            outer_rings.append(poly.exterior)
            hole_rings.extend(poly.interiors)
        return outer_rings, hole_rings

    @staticmethod
    def _orient_and_clip_ring(ring, patch, *, exterior: bool):
        # Exterior rings are forced CW; hole rings are forced CCW.
        needs_flip = ring.is_ccw if exterior else (not ring.is_ccw)
        if needs_flip:
            ring.coords = list(ring.coords)[::-1]
        clipped = ring.intersection(patch)
        if isinstance(clipped, MultiLineString):
            clipped = ops.linemerge(clipped)
        return clipped

    def _polygon_union_to_line_instances(self, union_segments, margin: float):
        patch = self._local_patch_box(margin)
        outer_rings, hole_rings = self._collect_polygon_rings(union_segments)
        clipped = [
            self._orient_and_clip_ring(ring, patch, exterior=True)
            for ring in outer_rings
        ]
        clipped.extend(
            self._orient_and_clip_ring(ring, patch, exterior=False)
            for ring in hole_rings
        )
        return self._one_type_line_geom_to_instances(clipped)

    def ped_poly_geoms_to_instances(self, ped_geom):
        ped = ped_geom[0][1]
        union_segments = ops.unary_union(ped)
        # Expand patch slightly so pedestrian contours near the border remain.
        return self._polygon_union_to_line_instances(
            union_segments, margin=0.2
        )

    def poly_geoms_to_instances(self, polygon_geom):
        roads = polygon_geom[0][1]
        lanes = polygon_geom[1][1]
        union_segments = ops.unary_union(
            [ops.unary_union(roads), ops.unary_union(lanes)]
        )
        # Shrink patch slightly to keep road/lane bounds inside the ROI.
        return self._polygon_union_to_line_instances(
            union_segments, margin=-0.2
        )

    def line_geoms_to_instances(self, line_geom):
        line_instances_dict = {}
        for line_type, a_type_of_lines in line_geom:
            one_type_instances = self._one_type_line_geom_to_instances(
                a_type_of_lines
            )
            line_instances_dict[line_type] = one_type_instances

        return line_instances_dict

    @staticmethod
    def _to_patch_local_frame(geom, patch_x, patch_y, patch_angle):
        """Rotate around the patch origin, then translate into local frame."""
        rotated = affinity.rotate(
            geom,
            -patch_angle,
            origin=(patch_x, patch_y),
            use_radians=False,
        )
        return affinity.affine_transform(
            rotated, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y]
        )

    @staticmethod
    def _wrap_single_polygon(geom):
        if geom.geom_type == "Polygon":
            return MultiPolygon([geom])
        return geom

    def _intersect_and_localize(
        self,
        geom,
        patch,
        patch_x,
        patch_y,
        patch_angle,
        *,
        as_multipoly: bool = False,
    ):
        """Clip ``geom`` to ``patch`` and map into the ego-local frame."""
        clipped = geom.intersection(patch)
        if clipped.is_empty:
            return None
        localized = self._to_patch_local_frame(
            clipped, patch_x, patch_y, patch_angle
        )
        if as_multipoly:
            localized = self._wrap_single_polygon(localized)
        return localized

    def get_contour_line(self, patch_box, patch_angle, layer_name, location):
        map_api = self.map_explorer[location].map_api
        if layer_name not in map_api.non_geometric_polygon_layers:
            raise ValueError("{} is not a polygonal layer".format(layer_name))

        patch_x, patch_y = patch_box[0], patch_box[1]
        patch = self.map_explorer[location].get_patch_coord(
            patch_box, patch_angle
        )
        records = getattr(map_api, layer_name)
        polygon_list = []

        if layer_name == "drivable_area":
            for record in records:
                for token in record["polygon_tokens"]:
                    poly = map_api.extract_polygon(token)
                    localized = self._intersect_and_localize(
                        poly,
                        patch,
                        patch_x,
                        patch_y,
                        patch_angle,
                        as_multipoly=True,
                    )
                    if localized is not None:
                        polygon_list.append(localized)
        else:
            for record in records:
                if "stop_line_type" in record and record[
                    "stop_line_type"
                ] not in ["PED_CROSSING", "TRAFFIC_LIGHT"]:
                    continue
                poly = map_api.extract_polygon(record["polygon_token"])
                if not poly.is_valid:
                    continue
                localized = self._intersect_and_localize(
                    poly,
                    patch,
                    patch_x,
                    patch_y,
                    patch_angle,
                    as_multipoly=True,
                )
                if localized is not None:
                    polygon_list.append(localized)

        return polygon_list

    def get_divider_line(self, patch_box, patch_angle, layer_name, location):
        map_api = self.map_explorer[location].map_api
        if layer_name not in map_api.non_geometric_line_layers:
            raise ValueError("{} is not a line layer".format(layer_name))
        if layer_name == "traffic_light":
            return None

        patch_x, patch_y = patch_box[0], patch_box[1]
        patch = self.map_explorer[location].get_patch_coord(
            patch_box, patch_angle
        )
        line_list = []
        for record in getattr(map_api, layer_name):
            line = map_api.extract_line(record["line_token"])
            if line.is_empty:
                continue
            localized = self._intersect_and_localize(
                line, patch, patch_x, patch_y, patch_angle
            )
            if localized is not None:
                line_list.append(localized)
        return line_list

    def get_ped_crossing_line(self, patch_box, patch_angle, location):
        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer[location].get_patch_coord(
            patch_box, patch_angle
        )
        polygon_list = []
        records = self.map_explorer[location].map_api.ped_crossing
        # records = getattr(self.nusc_maps[location], 'ped_crossing')
        for record in records:
            polygon = self.map_explorer[location].map_api.extract_polygon(
                record["polygon_token"]
            )
            if polygon.is_valid:
                new_polygon = polygon.intersection(patch)
                if not new_polygon.is_empty:
                    new_polygon = affinity.rotate(
                        new_polygon,
                        -patch_angle,
                        origin=(patch_x, patch_y),
                        use_radians=False,
                    )
                    new_polygon = affinity.affine_transform(
                        new_polygon, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y]
                    )
                    if new_polygon.geom_type == "Polygon":
                        new_polygon = MultiPolygon([new_polygon])
                    polygon_list.append(new_polygon)

        return polygon_list

    def sample_fixed_pts_from_line(self, line, padding=False, fixed_num=100):
        if padding:
            distances = np.arange(0, line.length, self.sample_dist)
            sampled_points = np.array(
                [
                    list(line.interpolate(distance).coords)
                    for distance in distances
                ]
            ).reshape(-1, 2)
        else:
            distances = np.linspace(0, line.length, fixed_num)
            sampled_points = np.array(
                [
                    list(line.interpolate(distance).coords)
                    for distance in distances
                ]
            ).reshape(-1, 2)

        num_valid = len(sampled_points)

        if num_valid < fixed_num:
            padding = np.zeros((fixed_num - len(sampled_points), 2))
            sampled_points = np.concatenate([sampled_points, padding], axis=0)
        elif num_valid > fixed_num:
            sampled_points = sampled_points[:fixed_num, :]
            num_valid = fixed_num

        num_valid = len(sampled_points)
        return sampled_points, num_valid

    def get_osm_geom(self, patch_box, patch_angle, location):
        osm_map = self.sd_maps[location]
        patch_x = patch_box[0]
        patch_y = patch_box[1]
        patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
        line_list = []
        for geom_line in osm_map.geoms:
            if geom_line.is_empty:
                continue
            new_line = geom_line.intersection(patch)
            if not new_line.is_empty:
                new_line = affinity.rotate(
                    new_line,
                    -patch_angle,
                    origin=(patch_x, patch_y),
                    use_radians=False,
                )
                new_line = affinity.affine_transform(
                    new_line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y]
                )
                line_list.append(new_line)
        return line_list
