# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import Optional, Tuple, Union

import numpy as np
import torch

from heal.core.box_utils import _CORNERS_ND_REORDER
from heal.core.points import BasePoints
from .box3d_utils import rotation_3d_in_axis, yaw2local


def _box_corners_from_dims(
    dims: torch.Tensor, origin: Tuple[float, float, float] = (0.5, 1.0, 0.5)
) -> torch.Tensor:
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
    corners_norm = corners_norm[_CORNERS_ND_REORDER[3]]
    corners_norm = corners_norm - np.array(origin, dtype=np.float32)
    corners_norm = torch.from_numpy(corners_norm).to(
        device=dims.device, dtype=dims.dtype
    )
    return dims.view(-1, 1, 3) * corners_norm.reshape(1, 8, 3)


def _rotate_points_with_matrix(
    points: Union[torch.Tensor, np.ndarray, BasePoints],
    rot_mat_T: torch.Tensor,
) -> Union[torch.Tensor, np.ndarray, BasePoints]:
    if isinstance(points, torch.Tensor):
        points[:, :3] = points[:, :3] @ rot_mat_T
    elif isinstance(points, np.ndarray):
        points[:, :3] = np.dot(points[:, :3], rot_mat_T.cpu().numpy())
    elif isinstance(points, BasePoints):
        points.rotate(rot_mat_T)
    else:
        raise ValueError
    return points


def _flip_points_bev(
    points: Union[torch.Tensor, np.ndarray, BasePoints],
    bev_direction: str,
    horizontal_axis: int = 0,
    vertical_axis: int = 2,
) -> Union[torch.Tensor, np.ndarray, BasePoints]:
    if isinstance(points, (torch.Tensor, np.ndarray)):
        axis = (
            horizontal_axis if bev_direction == "horizontal" else vertical_axis
        )
        points[:, axis] = -points[:, axis]
    elif isinstance(points, BasePoints):
        points.flip(bev_direction)
    return points


class CameraInstance3DBoxes(object):
    """Camera-frame 3D instance boxes.

    Camera axes (y points down, yaw around y):

    .. code-block:: none

                z (forward, yaw=-pi/2)
               /
              /
             0 ------> x (right, yaw=0)
             |
             v
             y (down)

    Bottom-center relative origin is (0.5, 1.0, 0.5); yaw axis index is 1.
    Yaw equals 0 along +x and decreases toward +z.

    Attributes:
        tensor: (N, box_dim) float matrix; each row is
            (x, y, z, x_size, y_size, z_size, yaw, ...).
        box_dim: Feature width of each box row.
        with_yaw: Whether boxes carry a meaningful yaw (False for padded 6-d).
    """

    YAW_AXIS = 1
    _DEFAULT_ORIGIN = (0.5, 1.0, 0.5)

    def __init__(
        self,
        tensor: torch.Tensor,
        box_dim: int = 7,
        with_yaw: bool = True,
        origin: Tuple[float] = (0.5, 1.0, 0.5),
    ):
        device = (
            tensor.device
            if isinstance(tensor, torch.Tensor)
            else torch.device("cpu")
        )
        data = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if data.numel() == 0:
            data = data.reshape(0, box_dim).to(
                dtype=torch.float32, device=device
            )
        assert data.dim() == 2 and data.size(-1) == box_dim, data.size()

        if data.shape[-1] == 6:
            assert box_dim == 6
            yaw_pad = data.new_zeros(data.shape[0], 1)
            data = torch.cat([data, yaw_pad], dim=-1)
            self.box_dim = box_dim + 1
            self.with_yaw = False
        else:
            self.box_dim = box_dim
            self.with_yaw = with_yaw
        self.tensor = data.clone()

        if origin != self._DEFAULT_ORIGIN:
            dst = self.tensor.new_tensor(self._DEFAULT_ORIGIN)
            src = self.tensor.new_tensor(origin)
            self.tensor[:, :3] = self.tensor[:, :3] + self.tensor[:, 3:6] * (
                dst - src
            )

    @property
    def height(self):
        """torch.Tensor: Box heights in shape (N,)."""
        return self.tensor[:, 4]

    @property
    def top_height(self):
        """torch.Tensor: Top y of each box in shape (N,)."""
        # Camera y+ points downward, so top is below bottom in value.
        return self.bottom_height - self.height

    @property
    def bottom_height(self):
        """torch.Tensor: Bottom y of each box in shape (N,)."""
        return self.tensor[:, 1]

    @property
    def bottom_center(self):
        """torch.Tensor: Bottom centers in shape (N, 3)."""
        return self.tensor[:, :3]

    @property
    def dims(self):
        """torch.Tensor: Size dimensions of each box in shape (N, 3)."""
        return self.tensor[:, 3:6]

    @property
    def yaw(self):
        """torch.Tensor: A vector with yaw of each box in shape (N, )."""
        return self.tensor[:, 6]

    @property
    def local_yaw(self):
        yaw = self.yaw
        loc = self.gravity_center
        local_yaw = yaw2local(yaw, loc)

        return local_yaw

    @property
    def gravity_center(self):
        """torch.Tensor: Gravity centers in shape (N, 3)."""
        center = self.bottom_center.clone()
        # Lift from bottom face to geometric mid-height along -y.
        center[:, 1] = center[:, 1] - self.tensor[:, 4] * 0.5
        return center

    @property
    def corners(self):
        """torch.Tensor: Box corners in shape (N, 8, 3).

        Corner order (clockwise):
        (x0y0z0, x0y0z1, x0y1z1, x0y1z0, x1y0z0, x1y0z1, x1y1z1, x1y1z0).

        .. code-block:: none

                         z (front)
                              /
               (x0,y0,z1) +--------+ (x1,y0,z1)
                         /|       /|
            (x0,y0,z0) +--------+ | (x1,y1,z1)
                       |  +-----|-+
                       | /      |/
            (x0,y1,z0) +--------+-------> x
                                (x1,y1,z0)
                       |
                       v y (down)
        """
        if self.tensor.numel() == 0:
            return torch.empty([0, 8, 3], device=self.tensor.device)

        local = _box_corners_from_dims(self.dims)
        local = rotation_3d_in_axis(
            local, self.tensor[:, 6], axis=self.YAW_AXIS
        )
        return local + self.tensor[:, :3].view(-1, 1, 3)

    @property
    def bev(self):
        """torch.Tensor: 2D BEV box of each box with rotation in XYWHR format, in shape (N, 5)."""  # noqa: E501
        bev = self.tensor[:, [0, 2, 3, 5, 6]].clone()
        # positive direction of the gravity axis
        # in cam coord system points to the earth
        # so the bev yaw angle needs to be reversed
        bev[:, -1] = -bev[:, -1]
        return bev

    def rotate(
        self,
        angle: Union[torch.Tensor, np.ndarray],
        points: Optional[Union[torch.Tensor, np.ndarray, BasePoints]] = None,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """Rotate boxes with points (optional) with the given angle or rotation \
        matrix.

        Args:
            angle: Scalar yaw delta or a 3x3 rotation matrix.
            points: Optional points to rotate with the same transform.

        Returns:
            ``None`` if ``points`` is None; otherwise
            ``(rotated_points, rot_mat_T)``.
        """
        rot = angle
        if not isinstance(rot, torch.Tensor):
            rot = self.tensor.new_tensor(rot)

        assert (
            rot.shape == torch.Size([3, 3]) or rot.numel() == 1
        ), f"invalid rotation angle shape {rot.shape}"

        if rot.numel() == 1:
            yaw_delta = rot
            self.tensor[:, 0:3], rot_mat_T = rotation_3d_in_axis(
                self.tensor[:, 0:3],
                yaw_delta,
                axis=self.YAW_AXIS,
                return_mat=True,
            )
        else:
            rot_mat_T = rot
            yaw_delta = np.arctan2(rot_mat_T[2, 0], rot_mat_T[0, 0])
            self.tensor[:, 0:3] = self.tensor[:, 0:3] @ rot_mat_T

        self.tensor[:, 6] = self.tensor[:, 6] + yaw_delta

        if points is not None:
            return _rotate_points_with_matrix(points, rot_mat_T), rot_mat_T

    def flip(
        self,
        bev_direction: str = "horizontal",
        points: Optional[Union[torch.Tensor, np.ndarray, BasePoints]] = None,
    ) -> Union[torch.Tensor, np.ndarray, BasePoints]:
        """Flip boxes in BEV along horizontal (x) or vertical (z).

        Args:
            bev_direction: ``horizontal`` or ``vertical``.
            points: Optional points to flip together.

        Returns:
            Flipped points when provided; otherwise ``None``.
        """
        assert bev_direction in ("horizontal", "vertical")
        if bev_direction == "horizontal":
            self.tensor[:, 0::7] = -self.tensor[:, 0::7]
            if self.with_yaw:
                self.tensor[:, 6] = np.pi - self.tensor[:, 6]
        else:
            self.tensor[:, 2::7] = -self.tensor[:, 2::7]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6]

        if points is not None:
            assert isinstance(points, (torch.Tensor, np.ndarray, BasePoints))
            return _flip_points_bev(points, bev_direction)

    @classmethod
    def height_overlaps(
        cls,
        boxes1: "CameraInstance3DBoxes",
        boxes2: "CameraInstance3DBoxes",
        mode: str = "iou",
    ) -> torch.Tensor:
        """1D height overlap between two camera box sets.

        Args:
            boxes1: N boxes.
            boxes2: M boxes.
            mode: Reserved IoU mode flag (unused).

        Returns:
            (N, M) height overlap lengths.
        """
        assert isinstance(boxes1, CameraInstance3DBoxes)
        assert isinstance(boxes2, CameraInstance3DBoxes)

        top1 = boxes1.top_height[:, None]
        bot1 = boxes1.bottom_height[:, None]
        top2 = boxes2.top_height[None, :]
        bot2 = boxes2.bottom_height[None, :]

        # Camera gravity axis points down; overlap is along y.
        overlap_lo = torch.maximum(top1, top2)
        overlap_hi = torch.minimum(bot1, bot2)
        return torch.clamp(overlap_hi - overlap_lo, min=0)

    def to(self, device):
        """Convert current boxes to a specific device.

        Args:
            device: The name of the device.

        Returns:
            A new boxes object on the specific device.
        """
        original_type = type(self)
        return original_type(
            self.tensor.to(device),
            box_dim=self.box_dim,
            with_yaw=self.with_yaw,
        )

    def __repr__(self):
        """str: Return a strings that describes the object."""
        return self.__class__.__name__ + "(\n    " + str(self.tensor) + ")"
