# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
import collections
import logging
from collections.abc import Sequence
from typing import Any, Dict, List, Tuple

import horizon_plugin_pytorch.nn as hnn
import numpy as np
import torch
from hat_engine.models.base_modules.conv_module import ConvModule2d
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.model_helpers import fx_wrap
from horizon_plugin_pytorch.dtype import qint16
from horizon_plugin_pytorch.nn.quantized import FloatFunctional
from horizon_plugin_pytorch.quantization import QuantStub
from torch import Tensor, nn

from heal.core.nus_box3d_utils import adjust_coords, get_min_max_coords

try:
    from hbdk.torch_script.placeholder import placeholder
except ImportError:
    placeholder = None

logger = logging.getLogger(__name__)


class ViewTransformer(nn.Module):
    """The view transform structure for converting image view to bev view.

    Args:
        num_views: Number for image views.
        bev_size: Bev size.
        grid_size: Grid size.
        mode: Mode for grid sample.
        padding_mode: Padding mode for grid sample.
        grid_quant_scale: Quanti scale for grid sample.

    """

    def __init__(
        self,
        num_views: int,
        bev_size: Tuple[float],
        grid_size: Tuple[float],
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        grid_quant_scale: float = 1 / 512,
        homo_key="ego2img",
    ):
        super(ViewTransformer, self).__init__()
        self.num_views = num_views
        self.bev_size = bev_size
        self.grid_size = grid_size
        self.floatFs = FloatFunctional()

        self.quant_stub = QuantStub(grid_quant_scale)
        self.grid_sample = hnn.GridSample(
            mode=mode,
            padding_mode=padding_mode,
        )
        self.homo_key = homo_key

    def _gen_3d_points(self, z_range: Tuple[int], cal_minmax=True) -> Tensor:
        """Generate 3D points in the Bird's Eye View (BEV) space.

        Args:
            z_range: The range of z-coordinates.

        Returns:
            coords: The generated 3D points in BEV space.
        """
        # Get the minimum and maximum x and y coordinates in the BEV space
        if cal_minmax is True:
            bev_min_x, bev_max_x, bev_min_y, bev_max_y = get_min_max_coords(
                self.bev_size
            )
        else:
            bev_min_x, bev_max_x, bev_min_y, bev_max_y = (
                -self.bev_size[0],
                self.bev_size[0],
                -self.bev_size[1],
                self.bev_size[1],
            )

        W = self.grid_size[0]
        H = self.grid_size[1]

        if len(z_range) == 3:
            Z = int((z_range[1] - z_range[0]) / z_range[2])
        else:
            Z = int(z_range[1] - z_range[0])

        # Generate a tensor `x` containing the x-coordinates of the grid
        x = (
            torch.linspace(bev_min_x, bev_max_x, W)
            .reshape((1, W, 1))
            .repeat(H, 1, Z)
        ).double()

        # Generate a tensor `y` containing the y-coordinates of the grid
        y = (
            torch.linspace(bev_min_y, bev_max_y, H)
            .reshape((H, 1, 1))
            .repeat(1, W, Z)
        ).double()

        # Generate a tensor `z` containing the z-coordinates of the grid
        # based on the given z_range
        z = (
            torch.linspace(self.z_range[0], self.z_range[1], Z)
            .reshape((1, 1, Z))
            .repeat(H, W, 1)
        ).double()

        # Generate a tensor `ones` containing all ones of shape (H, W, Z)
        ones = torch.ones((H, W, Z)).double()
        coords = torch.stack([x, y, z, ones], dim=-1)
        return coords

    def _convert_p2tensor(self, points: Any) -> Any:
        """Convert points to tensors.

        Args:
            points: The points to be converted.

        Returns:
            points: The converted points as tensors.
        """

        if isinstance(points, collections.abc.Sequence):
            for i in range(len(points)):
                points[i] = self._convert_p2tensor(points[i])
        if placeholder is not None and isinstance(points, placeholder):
            points = points.sample
        return points

    @fx_wrap()
    def forward(
        self, feats: Tensor, meta: Dict, compile_model: bool, **kwargs
    ) -> Any:
        """Perform the forward pass through the modules.

        Processes the input features and meta information to perform
        spatial transformation and generate reference points.

        Args:
            feats: The input features.
            meta: The meta information.
            compile_model: A flag indicating whether to compile the model.

        Returns:
            transformed_feats: The transformed features.
            points: The reference points.
        """

        feats = self._extract(feats)
        if compile_model is True:
            points = self._get_points_from_meta(meta)
        else:
            with torch.no_grad():
                points = self.gen_reference_point(meta, feats)
        return self._spatial_transfom(feats, points), None

    def gen_reference_point(self, meta: Dict, feats: Tensor) -> Any:
        """Generate refrence points.

        Args:
            meta: A dictionary containing the input data.
            feats: The input for reference point generator.

        Returns:
            The Reference points.
        """
        with torch.no_grad():
            homography = self._get_homography(
                meta, feats.shape[2:], self.homo_key
            )
            points = self._gen_reference_point(homography, feats.shape[2:])
        return points

    def export_reference_points(
        self, meta: Dict, feat_hw: Tuple[int, int]
    ) -> Dict:
        """Export refrence points.

        Args:
            meta: A dictionary containing the input data.
            feat_hw: View transformer input shape
                     for generationg reference points.

        Returns:
            The Reference points.
        """
        homography = self._get_homography(meta, feat_hw, self.homo_key)
        points = self._gen_reference_point(homography, feat_hw)
        if not isinstance(points, Sequence):
            points = [points]

        ref_p_dict = {}
        for i, ref_p in enumerate(points):
            ref_p_dict[f"points{i}"] = ref_p
        return ref_p_dict

    def _extract(self, feats: Tensor) -> Tensor:
        """Extract the input features.

        Args:
            feats: The input features.

        Returns:
            feats: The input features.
        """
        return feats

    def _get_homography(
        self, meta: Dict, feat_hw: Tuple[int, int], homo_key="ego2img"
    ) -> Tensor:
        """Compute the homography matrix for mapping coordinates.

        Args:
            meta: The meta information.
            feat_hw: view transformer input shape
                     for generationg reference points.

        Returns:
            homography: The computed homography matrix.
        """

        # Get the ego2img homography matrix and the input
        # and original feature heights and widths
        homography = meta[homo_key]
        orig_hw = meta["img"][0].shape[1:]
        scales = (feat_hw[0] / orig_hw[0], feat_hw[1] / orig_hw[1])
        view = np.eye(4)
        view[0, 0] = scales[1]
        view[1, 1] = scales[0]
        view = torch.tensor(view).to(device=homography.device).double()
        # Perform the matrix multiplication between
        # the view transformation matrix and the homography matrix
        homography = torch.matmul(view, homography.double())
        return homography

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""

        from hat_engine.utils import qconfig_manager

        self.qconfig = qconfig_manager.get_default_qat_qconfig()

        self.quant_stub.qconfig = qconfig_manager.get_qconfig(
            activation_qat_qkwargs={"dtype": qint16, "saturate": True},
            activation_calibration_qkwargs={"dtype": qint16, "saturate": True},
        )


@OBJECT_REGISTRY.register
class LSSTransformer(ViewTransformer):
    """The Lift-Splat-Shoot view transform for converting image view to bev view.

    Args:
        in_channels: In channel of feature.
        feat_channels: Feature channel of lift.
        z_range: The range of Z for bev coordarin.
        num_points: Num points for each voxel.
        depth: Depth value.
        mode: Mode for grid sample.
        padding_mode: Padding mode for grid sample.
        dgrid_quant_scale: Quanti scale for depth grid sample.

    """

    def __init__(
        self,
        in_channels: int,
        feat_channels: int,
        z_range: Tuple[float] = (-10.0, 10.0),
        num_points: int = 10,
        depth: int = 60,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        depth_grid_quant_scale: float = 1 / 512,
        use_vtv2: bool = False,
        cal_minmax: bool = True,
        **kwargs,
    ):
        super(LSSTransformer, self).__init__(
            mode=mode, padding_mode=padding_mode, **kwargs
        )
        self.depth = depth
        self.z_range = z_range
        self.depth_net = ConvModule2d(
            in_channels=in_channels,
            out_channels=depth,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False,
        )
        self.feat_net = ConvModule2d(
            in_channels=in_channels,
            out_channels=feat_channels,
            kernel_size=1,
            padding=0,
            stride=1,
            bias=False,
        )
        self.softmax = nn.Softmax(dim=1)
        self.dquant_stub = QuantStub(depth_grid_quant_scale)
        self.dgrid_sample = hnn.GridSample(
            mode=mode,
            padding_mode=padding_mode,
        )
        self.use_vtv2 = use_vtv2
        self.cal_minmax = cal_minmax
        self.num_points = num_points

    def gen_reference_point(self, meta: Dict, feats: List[Tensor]) -> Any:
        """Generate refrence points.

        Args:
            meta: A dictionary containing the input data.
            feats: The input for reference point generator.

        Returns:
            The Reference points.
        """
        return super().gen_reference_point(meta, feats[0])

    def _get_points_from_meta(self, meta: Dict) -> List[Tensor]:
        """Extract points from metadata dictionary.

        Args:
            meta: Metadata dictionary.

        Returns:
            points: List of extracted points as converted tensors.
        """
        points = []
        for k in meta.keys():
            if k.startswith("points"):
                points.append(self._convert_p2tensor(meta[k]))
        assert len(points) == 2
        return points

    def _gen_reference_point(
        self, homography: Tensor, feat_hw: Tuple[int, int]
    ) -> Tuple[Tensor]:
        """Generate reference points using homography matrix and feature tensor.

        Args:
            homography: Homography matrix.
            feat_hw: View transformer input shape
                     for generationg reference points.

        Returns:
            Tuple containing the generated feature points and depth points.
        """

        coords = self._gen_3d_points(
            self.z_range, cal_minmax=self.cal_minmax
        ).to(device=homography.device)
        H, W, Z = coords.shape[:3]
        new_coords = []
        for homo in homography:
            new_coord = torch.matmul(coords, homo.permute((1, 0))).float()
            new_coord = new_coord.permute((2, 0, 1, 3))
            new_coords.append(new_coord)
        new_coords = torch.stack(new_coords, dim=1)
        B = new_coords.shape[1] // self.num_views

        new_coords = (
            new_coords.view(-1, B, self.num_views, H, W, 4)
            .permute(0, 2, 1, 3, 4, 5)
            .contiguous()
        )

        d = torch.clamp(new_coords[..., 2], min=0.05)
        X = (new_coords[..., 0] / d).long()
        Y = (new_coords[..., 1] / d).long()
        D = new_coords[..., 2].long()

        idx = (
            (
                torch.linspace(0, self.num_views - 1, self.num_views)
                .reshape((1, self.num_views, 1, 1, 1))
                .repeat(Z, 1, B, H, W)
            )
            .long()
            .to(device=homography.device)
        )
        new_coords = torch.stack([X, Y, D, idx], dim=-1)

        feat_h, feat_w = feat_hw
        invalid = (
            (new_coords[..., 0] < 0)
            | (new_coords[..., 0] >= feat_w)
            | (new_coords[..., 1] < 0)
            | (new_coords[..., 1] >= feat_h)
            | (new_coords[..., 2] < 0)
            | (new_coords[..., 2] >= self.depth)
        )
        new_coords[invalid] = torch.tensor(
            65535,
        ).to(device=homography.device)
        new_coords = new_coords.view(-1, B, H, W, 4)
        coords_xy = new_coords[..., :2].to(dtype=torch.float)
        coords_xy[..., 0] = coords_xy[..., 0] / feat_w
        coords_xy[..., 1] = coords_xy[..., 1] / feat_h
        center = (
            torch.tensor(
                (0.5, 0),
            )
            .to(device=homography.device)
            .view(1, -1)
        )
        dist = torch.cdist(coords_xy, center, p=2).squeeze(-1)

        dist_sorted, indices_sorted = torch.sort(dist, dim=0)
        indices_sorted = indices_sorted.unsqueeze(-1).expand(-1, -1, -1, -1, 4)
        coords_sorted = torch.gather(new_coords, dim=0, index=indices_sorted)
        mask = torch.zeros_like(dist_sorted, dtype=torch.bool)

        mask[1:] = dist_sorted[1:] == dist_sorted[:-1]
        dist_cleaned = dist_sorted.masked_fill(mask, 65535)
        mask = mask.unsqueeze(-1).expand(-1, -1, -1, -1, 4)
        coords_cleaned = coords_sorted.masked_fill(mask, 65535)
        _, indices = dist_cleaned.topk(self.num_points, dim=0, largest=False)
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, -1, 4)
        topk_coords = torch.gather(coords_cleaned, dim=0, index=indices)
        X = topk_coords[..., 0]
        Y = topk_coords[..., 1]
        D = topk_coords[..., 2]
        idx = topk_coords[..., 3]

        idx_Y = idx * feat_h + Y
        feat_coords = torch.stack((X, idx_Y), dim=-1)
        feat_points = adjust_coords(feat_coords, self.grid_size)

        X_Y = Y * feat_w + X
        idx_D = idx * self.depth + D
        depth_coords = torch.stack((X_Y, idx_D), dim=-1)
        depth_points = adjust_coords(depth_coords, self.grid_size)
        feat_points = feat_points.view(-1, H, W, 2)
        depth_points = depth_points.view(-1, H, W, 2)
        return (feat_points, depth_points)

    def _extract(
        self, feats: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Extract features and depth using the feature tensor.

        Args:
            feats: Feature tensor.

        Returns:
            Tuple containing the extracted features and depth.
        """

        new_feats = []
        depth = self.softmax(self.depth_net(feats))
        new_feats = self.feat_net(feats)
        return new_feats, depth

    def _spatial_transfom(self, feats: Tensor, points: Tensor) -> Tensor:
        """Apply spatial transformation to the features using the given points.

        Args:
            feats: Tuple of feature tensor and depth tensor.
            points: Tuple of feature points and depth points.

        Returns:
            The transformed feature tensor.
        """
        feat, dfeat = feats
        fpoints, dpoints = points
        fpoints = self.quant_stub(fpoints)
        dpoints = self.dquant_stub(dpoints)

        B = feat.shape[0] // self.num_views
        C, H, W = feat.shape[1:]

        if self.training or B > 1:
            feat = feat.view(B, self.num_views, C, H, W)
            feat = feat.permute(0, 2, 1, 3, 4).contiguous()
        else:
            feat = feat.permute(1, 0, 2, 3).contiguous()

        feat = feat.view(B, C, -1, W)

        dfeat = dfeat.view(B, 1, -1, H * W)
        homo_feats = []
        for i in range(self.num_points):
            homo_feat = self.grid_sample(
                feat,
                fpoints[i * B : (i + 1) * B],
            )

            homo_dfeat = self.dgrid_sample(
                dfeat,
                dpoints[i * B : (i + 1) * B],
            )
            homo_feat = self.floatFs.mul(homo_feat, homo_dfeat)
            homo_feats.append(homo_feat)

        trans_feat = homo_feats[0]
        for f in homo_feats[1:]:
            trans_feat = self.floatFs.add(trans_feat, f)
        return trans_feat

    def fuse_model(self):
        """Perform model fusion on the modules."""
        self.depth_net.fuse_model()
        self.feat_net.fuse_model()

    def set_qconfig(self) -> None:
        """Set the quantization configuration."""

        from hat_engine.utils import qconfig_manager

        self.dquant_stub.qconfig = qconfig_manager.get_qconfig(
            activation_qat_qkwargs={"dtype": qint16, "saturate": True},
            activation_calibration_qkwargs={"dtype": qint16, "saturate": True},
        )
        super().set_qconfig()
