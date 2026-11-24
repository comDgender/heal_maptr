# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

# Copyright (c) Open-MMLab. All rights reserved.
# @misc{mmdet3d2020,
#     title={{MMDetection3D: OpenMMLab} next-generation platform for general {3D} object detection},  # noqa E501
#     author={MMDetection3D Contributors},
#     howpublished = {\url{https://github.com/open-mmlab/mmdetection3d}},
#     year={2020}
# }
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Added line36-line288, line292-line448  # noqa E501

import warnings
from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from heal.core.box3d_utils import rotation_3d_in_axis


def _as_points_tensor(tensor, points_dim: int) -> torch.Tensor:
    device = (
        tensor.device
        if isinstance(tensor, torch.Tensor)
        else torch.device("cpu")
    )
    tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
    if tensor.numel() == 0:
        tensor = tensor.reshape((0, points_dim)).to(
            dtype=torch.float32, device=device
        )
    assert tensor.dim() == 2 and tensor.size(-1) == points_dim, tensor.size()
    return tensor


def _split_bounds_tuple(bounds: Tuple) -> Tuple[Tuple, Tuple]:
    mid = len(bounds) // 2
    return bounds[:mid], bounds[mid:]


def _points_inside_bounds(
    positions: torch.Tensor, bounds: Tuple
) -> torch.Tensor:
    ndim = len(bounds) // 2
    lower, upper = _split_bounds_tuple(bounds)
    lower_t = positions.new_tensor(lower)
    upper_t = positions.new_tensor(upper)
    values = positions[:, :ndim]
    return (values > lower_t).all(dim=1) & (values < upper_t).all(dim=1)


class BasePoints(object):
    """Base class for Points.

    Args:
        tensor: a N x points_dim matrix.
        points_dim: Number of the dimension of a point.
            Each row is (x, y, z). Defaults to 3.
        attribute_dims: Dictionary to indicate the
            meaning of extra dimension. Defaults to None.
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        points_dim: Optional[int] = 3,
        attribute_dims: Optional[Dict] = None,
    ):
        self.tensor = _as_points_tensor(tensor, points_dim)
        self.points_dim = points_dim
        self.attribute_dims = attribute_dims
        self.rotation_axis = 0

    def _reshape_tensor(self, tensor, shape):
        try:
            tensor = tensor.reshape(shape)
        except (RuntimeError, ValueError):
            raise ValueError(f"got unexpected shape {tensor.shape}")
        if not isinstance(tensor, torch.Tensor):
            tensor = self.tensor.new_tensor(tensor)
        return tensor

    def _get_attribute(self, key: str):
        dims = self.attribute_dims
        if dims is not None and key in dims:
            return self.tensor[:, dims[key]]
        return None

    def _with_tensor(
        self,
        tensor: torch.Tensor,
        points_dim: Optional[int] = None,
        attribute_dims: Optional[Dict] = None,
    ):
        return type(self)(
            tensor,
            points_dim=(
                points_dim if points_dim is not None else self.points_dim
            ),
            attribute_dims=(
                attribute_dims
                if attribute_dims is not None
                else self.attribute_dims
            ),
        )

    def _ensure_attribute_dims(self) -> Dict:
        if self.attribute_dims is None:
            self.attribute_dims = {}
        return self.attribute_dims

    def _append_attribute_column(
        self, name: str, values: torch.Tensor
    ) -> None:
        dims = self._ensure_attribute_dims()
        col_idx = self.shape[1]
        self.tensor = torch.cat([self.tensor, values.unsqueeze(1)], dim=1)
        dims[name] = col_idx
        self.points_dim += 1

    def _append_attribute_columns(
        self, name: str, values: torch.Tensor
    ) -> None:
        dims = self._ensure_attribute_dims()
        start_col = self.shape[1]
        width = values.shape[1]
        self.tensor = torch.cat([self.tensor, values], dim=1)
        dims[name] = list(range(start_col, start_col + width))
        self.points_dim += width

    def _warn_color_out_of_range(self, values: torch.Tensor) -> None:
        if values.max() >= 256 or values.min() < 0:
            warnings.warn("point got color value beyond [0, 255]")

    def _as_rotation_tensor(self, rotation) -> torch.Tensor:
        rot = (
            rotation
            if isinstance(rotation, torch.Tensor)
            else self.tensor.new_tensor(rotation)
        )
        if rot.shape != torch.Size([3, 3]) and rot.numel() != 1:
            raise AssertionError(f"invalid rotation shape {rot.shape}")
        return rot

    def _validate_translation_vector(self, delta: torch.Tensor) -> None:
        if delta.dim() == 1:
            if delta.shape[0] != 3:
                raise NotImplementedError(
                    f"Unsupported translation vector of shape {delta.shape}"
                )
        elif delta.dim() == 2:
            n_pts, n_axes = delta.shape
            if n_pts != self.tensor.shape[0] or n_axes != 3:
                raise NotImplementedError(
                    f"Unsupported translation vector of shape {delta.shape}"
                )
        else:
            raise NotImplementedError(
                f"Unsupported translation vector of shape {delta.shape}"
            )

    def _normalize_column_index(self, col_spec):
        if isinstance(col_spec, slice):
            begin = 0 if col_spec.start is None else col_spec.start
            end = (
                self.tensor.shape[1]
                if col_spec.stop is None
                else col_spec.stop
            )
            stride = 1 if col_spec.step is None else col_spec.step
            return list(range(begin, end, stride))
        if isinstance(col_spec, int):
            return [col_spec]
        return col_spec

    def _attribute_dims_for_columns(
        self, columns: List[int]
    ) -> Optional[Dict]:
        if self.attribute_dims is None:
            return None
        extra_cols = set(range(3, self.tensor.shape[1]))
        selected_extra = set(columns) & extra_cols
        filtered = {}
        for key, dims in self.attribute_dims.items():
            dim_list = [dims] if isinstance(dims, int) else list(dims)
            overlap = [d for d in dim_list if d in selected_extra]
            if len(overlap) == 1:
                filtered[key] = overlap[0]
            elif len(overlap) > 1:
                filtered[key] = overlap
        return filtered

    @property
    def coord(self):
        """torch.Tensor: Coordinates of each point in shape (N, 3)."""
        return self.tensor[:, :3]

    @coord.setter
    def coord(self, tensor):
        """Set the coordinates of each point."""
        self.tensor[:, :3] = self._reshape_tensor(tensor, (self.shape[0], 3))

    @property
    def height(self):
        """torch.Tensor: A vector with height of each point in shape (N, 1), or None."""  # noqa: E501
        return self._get_attribute("height")

    @height.setter
    def height(self, tensor):
        """Set the height of each point."""
        values = self._reshape_tensor(tensor, (self.shape[0],))
        if self._get_attribute("height") is not None:
            self.tensor[:, self.attribute_dims["height"]] = values
        else:
            self._append_attribute_column("height", values)

    @property
    def color(self):
        """torch.Tensor: A vector with color of each point in shape (N, 3), or None."""  # noqa: E501
        return self._get_attribute("color")

    @color.setter
    def color(self, tensor):
        """Set the color of each point."""
        values = self._reshape_tensor(tensor, (self.shape[0], 3))
        self._warn_color_out_of_range(values)
        if self._get_attribute("color") is not None:
            self.tensor[:, self.attribute_dims["color"]] = values
        else:
            self._append_attribute_columns("color", values)

    @property
    def shape(self):
        """torch.Shape: Shape of points."""
        return self.tensor.shape

    def shuffle(self):
        """Shuffle the points.

        Returns:
            torch.Tensor: The shuffled index.
        """
        perm = torch.randperm(len(self), device=self.tensor.device)
        self.tensor = self.tensor[perm]
        return perm

    def rotate(self, rotation, axis=None):
        """Rotate points with the given rotation matrix or angle.

        Args:
            rotation: Rotation matrix
                or angle.
            axis: Axis to rotate at. Defaults to None.
        """
        rot = self._as_rotation_tensor(rotation)
        rot_axis = self.rotation_axis if axis is None else axis

        if rot.numel() == 1:
            rotated, rot_mat_T = rotation_3d_in_axis(
                self.coord[None], rot, axis=rot_axis, return_mat=True
            )
            self.tensor[:, :3] = rotated.squeeze(0)
            return rot_mat_T.squeeze(0)

        self.tensor[:, :3] = self.coord @ rot
        return rot

    @abstractmethod
    def flip(self, bev_direction="horizontal"):
        """Flip the points along given BEV direction.

        Args:
            bev_direction: Flip direction (horizontal or vertical).
        """

    def translate(self, trans_vector):
        """Translate points with the given translation vector.

        Args:
            trans_vector: Translation
                vector of size 3 or nx3.
        """
        delta = (
            trans_vector
            if isinstance(trans_vector, torch.Tensor)
            else self.tensor.new_tensor(trans_vector)
        )
        delta = delta.squeeze(0)
        self._validate_translation_vector(delta)
        self.tensor[:, :3] = self.coord + delta

    def in_range_3d(self, point_range: Tuple) -> torch.Tensor:
        """Test whether XYZ coordinates fall inside an open axis-aligned box.

        Args:
            point_range: Lower and upper bounds as
                (x_min, y_min, z_min, x_max, y_max, z_max).

        Returns:
            Boolean mask of shape (N,) with True for in-range points.
        """
        return _points_inside_bounds(self.coord, point_range)

    @property
    def bev(self):
        """Plan-view XY coordinates, shape (N, 2)."""
        return self.tensor[:, 0:2]

    def in_range_bev(self, point_range: Tuple) -> torch.Tensor:
        """Test whether BEV XY coordinates fall inside an open rectangle.

        Args:
            point_range: Lower and upper bounds as
                (x_min, y_min, x_max, y_max).

        Returns:
            Boolean mask of shape (N,) with True for in-range points.
        """
        return _points_inside_bounds(self.tensor[:, 0:2], point_range)

    @abstractmethod
    def convert_to(self, dst, rt_mat=None):
        """Re-express points in another coordinate representation.

        Args:
            dst: Target coordinate mode.
            rt_mat: Optional 4x4 rigid transform from the current frame to
                ``dst``. Needed when the conversion switches sensor frames.

        Returns:
            A points instance of the same type in ``dst``.
        """

    def scale(self, scale_factor):
        """Multiply XYZ coordinates by ``scale_factor``."""
        scaled_xyz = self.coord * scale_factor
        self.tensor[:, :3] = scaled_xyz

    def __getitem__(self, item):
        attr_dims = self.attribute_dims
        if isinstance(item, int):
            subset = self.tensor[item].view(1, -1)
        elif isinstance(item, tuple) and len(item) == 2:
            row_idx, col_spec = item
            columns = self._normalize_column_index(col_spec)
            subset = self.tensor[row_idx, columns]
            attr_dims = self._attribute_dims_for_columns(columns)
        elif isinstance(item, (slice, np.ndarray, torch.Tensor)):
            subset = self.tensor[item]
        else:
            raise NotImplementedError(f"Invalid slice {item}!")

        if subset.dim() != 2:
            raise ValueError(
                f"Indexing on Points with {item} failed to return a matrix!"
            )
        return self._with_tensor(
            subset, points_dim=subset.shape[1], attribute_dims=attr_dims
        )

    def __len__(self):
        """int: Number of points in the current object."""
        return self.tensor.shape[0]

    def __repr__(self):
        """str: Return a strings that describes the object."""
        return self.__class__.__name__ + "(\n    " + str(self.tensor) + ")"

    @classmethod
    def cat(cls, points_list):
        """Concatenate a list of Points into a single Points.

        Args:
            points_list: List of points.

        Returns:
            The concatenated Points.
        """
        assert isinstance(points_list, (list, tuple))
        if not points_list:
            return cls(torch.empty(0))
        assert all(isinstance(entry, cls) for entry in points_list)

        head = points_list[0]
        merged = torch.cat([entry.tensor for entry in points_list], dim=0)
        return cls(
            merged,
            points_dim=head.tensor.shape[1],
            attribute_dims=head.attribute_dims,
        )

    def to(self, device):
        """Convert current points to a specific device.

        Args:
            device: The name of the device.

        Returns:
            A new boxes object on the
                specific device.
        """
        return self._with_tensor(self.tensor.to(device))

    def clone(self):
        """Clone the Points.

        Returns:
            Box object with the same properties
                as self.
        """
        return self._with_tensor(self.tensor.clone())

    @property
    def device(self):
        """str: The device of the points are on."""
        return self.tensor.device

    def __iter__(self):
        """Yield a point as a Tensor of shape (4,) at a time.

        Returns:
            torch.Tensor: A point of shape (4,).
        """
        for row in self.tensor:
            yield row

    def new_point(
        self, data: Union[torch.Tensor, List, np.ndarray]
    ) -> "BasePoints":  # noqa
        """Create a new point object with data.

        The new point and its tensor has the similar properties
            as self and self.tensor, respectively.

        Args:
            data: Data to be copied.

        Returns:
            :obj:`BasePoints`: A new point object with ``data``,
                the object's other properties are similar to ``self``.
        """
        if isinstance(data, torch.Tensor):
            tensor = data.to(device=self.device)
        else:
            tensor = self.tensor.new_tensor(data)
        return self._with_tensor(tensor)
