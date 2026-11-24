# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

# Voxel generation utilities
from typing import Tuple

import numpy as np

from heal.core.numba_lazy import lazy_jit


@lazy_jit(nopython=True)
def _point_inside_voxel_grid(
    point,
    bounds,
    vsize,
    grid_dims,
    swap_axes,
    coord_out,
):
    for src_axis in range(3):
        rel = (point[src_axis] - bounds[src_axis]) / vsize[src_axis]
        grid_idx = np.floor(rel)
        if grid_idx < 0 or grid_idx >= grid_dims[src_axis]:
            return False
        dst_axis = 2 - src_axis if swap_axes else src_axis
        coord_out[dst_axis] = np.int32(grid_idx)
    return True


@lazy_jit(nopython=True)
def _store_point_in_voxel(
    point_row,
    coord,
    index_map,
    counts,
    voxels,
    coors,
    active_cells,
    cap_voxels,
    cap_points,
):
    cell_id = index_map[coord[0], coord[1], coord[2]]
    if cell_id == -1:
        if active_cells >= cap_voxels:
            return active_cells
        cell_id = active_cells
        index_map[coord[0], coord[1], coord[2]] = cell_id
        coors[cell_id] = coord
        active_cells += 1
    slot = counts[cell_id]
    if slot < cap_points:
        voxels[cell_id, slot] = point_row
        counts[cell_id] = slot + 1
    return active_cells


@lazy_jit(nopython=True)
def _points_to_voxel_kernel(
    points: np.ndarray,
    voxel_size: np.ndarray,
    coors_range: np.ndarray,
    num_points_per_voxel: np.ndarray,
    coor_to_voxelidx: np.ndarray,
    voxels: np.ndarray,
    coors: np.ndarray,
    max_points: int = 35,
    max_voxels: int = 20000,
    reverse_index: bool = True,
) -> int:
    """Kernel of the voxel generation function.

    Generates voxels and assign points to each voxel.

    Args:
        points : an [N, >=3] array containing point clouds.
        voxel_size : xyz dimension of each voxel.
        coors_range : xyzxyz range of the entire point cloud.
        num_points_per_voxel : a 1-d array that is used to record
            number of points in each voxel.
        coor_to_voxelidx : mapping from coordinate to voxel id.
        voxels : a 3-d tensor of voxels. Each voxel is a collection
            of points. The max number of points in each voxel is defined by
            max_points.
            For more details, can refer to PointPillars paper.
        coors : a [P, 3] matrix recording each voxel's coordinate.
        max_points : max number of points in a voxel.
            Defaults to 35.
        max_voxels : max number of voxels. Defaults to 20000.

    Returns:
        int: number of non-empty voxels. Additionally, voxels will be created
            and points will be assigned to voxels.
    """
    # NOTE: need mutex if written in cuda, but numba.cuda doesn't support
    # mutex. Also, pytorch doesn't support cuda in dataloader(tf supports this)

    # Put all computations in one loop.
    # Should not create large array in main JIT code, causing performance drop
    N = points.shape[0]  # number of points
    ndim = 3
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)

    coord_buf = np.zeros(ndim, dtype=np.int32)
    active_cells = 0

    for pt_idx in range(N):
        if not _point_inside_voxel_grid(
            points[pt_idx],
            coors_range,
            voxel_size,
            grid_size,
            reverse_index,
            coord_buf,
        ):
            continue
        active_cells = _store_point_in_voxel(
            points[pt_idx],
            coord_buf,
            coor_to_voxelidx,
            num_points_per_voxel,
            voxels,
            coors,
            active_cells,
            max_voxels,
            max_points,
        )
    return active_cells


def _coerce_ndarray(values, dtype) -> np.ndarray:
    return (
        values
        if isinstance(values, np.ndarray)
        else np.array(values, dtype=dtype)
    )


def _compute_voxel_grid_shape(
    bounds: np.ndarray, voxel_size: np.ndarray, flip_axes: bool
) -> tuple:
    extent = bounds[3:] - bounds[:3]
    grid = tuple(np.round(extent / voxel_size).astype(np.int32))
    return grid[::-1] if flip_axes else grid


def _create_voxel_workspace(
    cap_voxels: int,
    cap_points: int,
    point_dim: int,
    grid_shape: tuple,
    dtype,
) -> dict:
    return {
        "counts": np.zeros(cap_voxels, dtype=np.int32),
        "index_map": -np.ones(grid_shape, dtype=np.int32),
        "voxels": np.zeros((cap_voxels, cap_points, point_dim), dtype=dtype),
        "coords": np.zeros((cap_voxels, 3), dtype=np.int32),
    }


def points_to_voxel(
    points: np.ndarray,
    voxel_size: np.ndarray,
    coors_range: np.ndarray,
    max_points: int = 35,
    reverse_index: bool = True,
    max_voxels: int = 20000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert lidar points [N, >=3] to voxels.

    With the help of JIT
    this function runs fast. Note that it seems to run faster under Ubuntu
    than under Windows 10.

    Args:
        points : [N, >=3] float array or tensor. The first 3
            columns contain xyz points, and the rest columns contain other
            information.
        voxel_size : xyz dimension of the voxel.
        coors_range : xyzxyz dimension of the point cloud.
        max_points : maximum number of points in a voxel.
            Defaults to 35.
        reverse_index: When True, voxel coordinates are emitted in zyx order
            while per-point features remain xyz.
        max_voxels: Hard cap on allocated voxels; excess points are skipped.

    Returns:
        Tuple of `(voxels, coordinates, num_points_per_voxel)` trimmed to the
        number of non-empty voxels produced by the JIT kernel.
    """
    vsize = _coerce_ndarray(voxel_size, points.dtype)
    bounds = _coerce_ndarray(coors_range, points.dtype)
    grid_shape = _compute_voxel_grid_shape(bounds, vsize, reverse_index)
    workspace = _create_voxel_workspace(
        max_voxels,
        max_points,
        points.shape[-1],
        grid_shape,
        points.dtype,
    )

    filled_count = _points_to_voxel_kernel(
        points,
        vsize,
        bounds,
        workspace["counts"],
        workspace["index_map"],
        workspace["voxels"],
        workspace["coords"],
        max_points,
        max_voxels,
        reverse_index,
    )

    return (
        workspace["voxels"][:filled_count],
        workspace["coords"][:filled_count],
        workspace["counts"][:filled_count],
    )


class VoxelGenerator(object):
    def __init__(
        self,
        voxel_size: Tuple[float, float, float],
        point_cloud_range: Tuple[float, ...],
        max_num_points: int,
        max_voxels: int = 20000,
    ):
        """Generate voxels from point cloud.

        Args:
            voxel_size : xyz dimension
                of each voxel.
            point_cloud_range : xyzxyz of point
                cloud range. Taking ego car as origin.
            max_num_points : maximum number of points in a voxel.
            max_voxels : maximum number of voxels.
                Defaults to 20000.
        """
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = np.divide(
            (point_cloud_range[3:] - point_cloud_range[:3]), voxel_size
        )
        grid_size = np.round(grid_size).astype(np.int64)
        # NOTE: grid_size has to be integers. We don't do a dtype check here,
        # so the user has to manually calculate point_cloud_range and
        # voxel_size to satisfy both dtype requirement and downsampling
        # requirement.
        # TODO: grid_size is not used anywhere. The kernels compute grid_size
        # by themselves. Can consider removing it.
        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_num_points = max_num_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size

    def generate(self, points: np.ndarray) -> np.ndarray:
        return points_to_voxel(
            points,
            self._voxel_size,
            self._point_cloud_range,
            self._max_num_points,
            True,
            self._max_voxels,
        )

    @property
    def voxel_size(self) -> np.ndarray:
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self) -> int:
        return self._max_num_points

    @property
    def point_cloud_range(self) -> np.ndarray:
        return self._point_cloud_range

    @property
    def grid_size(self) -> np.ndarray:
        return self._grid_size
