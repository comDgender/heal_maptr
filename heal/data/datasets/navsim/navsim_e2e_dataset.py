# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import os
import pickle
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import geopandas as gpd
import msgpack
import msgpack_numpy
import numpy as np
import shapely.affinity as affinity
import torch
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.pack_type.utils import get_packtype_from_path
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString, box
from torch.utils.data import Dataset

from heal.data.datasets.navsim.navsim_perception_dataset import (
    NavsimPerceptionMultitaskDataset,
)

try:
    from nuplan.common.actor_state.state_representation import (
        Point2D,
        TimePoint,
    )
    from nuplan.common.actor_state.vehicle_parameters import (
        get_pacifica_parameters,
    )

    from heal.data.datasets.navsim.nuplan_utils.map_process import map_process
except ImportError:
    Point2D = None
    map_process = None
    get_pacifica_parameters = None
    TimePoint = None

msgpack_numpy.patch()


# ── Camera name mapping: Scene Camera attribute → LMDB cam key ──────────

_CAM_ATTR_TO_KEY = {
    "cam_f0": "CAM_F0",
    "cam_l0": "CAM_L0",
    "cam_l1": "CAM_L1",
    "cam_l2": "CAM_L2",
    "cam_r0": "CAM_R0",
    "cam_r1": "CAM_R1",
    "cam_r2": "CAM_R2",
    "cam_b0": "CAM_B0",
}

# _CAM_NAMES_ORDERED = [
#     "CAM_F0", "CAM_L0", "CAM_L1", "CAM_L2",
#     "CAM_R0", "CAM_R1", "CAM_R2", "CAM_B0",
# ]
_CAM_NAMES_ORDERED = [
    "CAM_F0",
    "CAM_L0",
    "CAM_R0",
    "CAM_L1",
    "CAM_R1",
    "CAM_L2",
    "CAM_R2",
    "CAM_B0",
]


def _to_tensor(arr, dtype=torch.float32):
    if dtype == torch.long:
        numpy_type = np.int64
    elif dtype == torch.bool:
        numpy_type = np.bool_
    else:
        numpy_type = np.float32

    if isinstance(arr, torch.Tensor):
        if arr.dtype != dtype:
            if dtype == torch.long:
                arr = arr.long()
            elif dtype == torch.bool:
                arr = arr.bool()
            else:
                arr = arr.float()
        return arr

    if isinstance(arr, np.ndarray):
        if arr.dtype != numpy_type:
            arr = arr.astype(numpy_type, copy=False)
        return torch.from_numpy(arr)

    return torch.tensor(arr, dtype=dtype)


def _camera_to_perception_dict(
    frame,
    lidar2global: np.ndarray,
    scene_token: Optional[str] = None,
    cam_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Convert a Scene Frame's Camera objects to a perception-model input dict.

    Builds the same structure as ``NavsimPerceptionSampler.__call__`` but
    operating on Frame/Camera dataclasses instead of raw LMDB dicts.

    Returns:
        Dict with keys: ``img``, ``lidar2img``, ``cam2lidar``,
        ``camera_intrinsic``, ``cam_name``, ``layout``, ``color_space``,
        ``timestamp``, ``sample_token``, plus ego pose keys.
    """
    cameras = frame.cameras
    imgs = []
    lidar2img_list = []
    global2img_list = []
    cam2lidar_list = []
    cam_intrin_list = []

    if cam_names is None:
        cam_names = _CAM_NAMES_ORDERED
    for cam_key in cam_names:
        attr = cam_key.lower()  # CAM_F0 → cam_f0
        cam = getattr(cameras, attr, None)
        if cam is None or cam.image is None:
            continue

        # Image: Camera stores PIL Image (from _decode_image) or numpy
        from PIL import Image as PILImage

        if isinstance(cam.image, PILImage.Image):
            pil_img = cam.image.convert("RGB")
        elif isinstance(cam.image, np.ndarray):
            pil_img = PILImage.fromarray(cam.image).convert("RGB")
        else:
            continue

        # Extrinsic: sensor2lidar → cam2lidar
        cam2lidar = np.eye(4, dtype=np.float64)
        if cam.sensor2lidar_rotation is not None:
            cam2lidar[:3, :3] = cam.sensor2lidar_rotation
        if cam.sensor2lidar_translation is not None:
            cam2lidar[:3, 3] = cam.sensor2lidar_translation

        # lidar2img = intrinsic @ [R^T | -R^T @ t]
        if cam.intrinsics is not None:
            intrinsic = np.array(cam.intrinsics, dtype=np.float64)
        else:
            intrinsic = np.eye(3, dtype=np.float64)

        R = cam2lidar[:3, :3]
        t = cam2lidar[:3, 3]
        Rt = R.T
        tt = -Rt @ t
        lidar2img = np.eye(4, dtype=np.float64)
        lidar2img[:3, :4] = intrinsic @ np.hstack([Rt, tt.reshape(3, 1)])

        global2lidar = np.linalg.inv(lidar2global)
        global2img = lidar2img @ global2lidar

        imgs.append(pil_img)
        lidar2img_list.append(lidar2img.astype(np.float32))
        global2img_list.append(global2img.astype(np.float32))
        cam2lidar_list.append(cam2lidar.astype(np.float32))
        cam_intrin_list.append(intrinsic.astype(np.float32))

    result = {
        "img": imgs,
        "cam_name": _CAM_NAMES_ORDERED[: len(imgs)],
        "lidar2img": lidar2img_list,
        "cam2lidar": cam2lidar_list,
        "camera_intrinsic": np.array(cam_intrin_list),
        "layout": "chw",
        "color_space": "rgb",
        "timestamp": frame.timestamp,
        "sample_token": frame.token,
        "scene": scene_token if scene_token is not None else frame.scene_token,
        "ego2global": frame.ego2global.astype(np.float32),
        "lidar2global": frame.lidar2global.astype(np.float32),
        "lidar2ego": frame.lidar2ego.astype(np.float32),
        "lidar2ego_translation": frame.lidar2ego_translation.astype(
            np.float32
        ),
        "lidar2ego_rotation": np.array(frame.lidar2ego_rotation).astype(
            np.float32
        ),
        "ego2global_translation": frame.ego2global_translation.astype(
            np.float32
        ),
        "ego2global_rotation": frame.ego2global_rotation.astype(np.float32),
    }
    return result


@OBJECT_REGISTRY.register
class NavsimE2EDataset(Dataset):
    """Scene-based E2E dataset for Navsim perception → planning.

    Wraps ``NavsimDataset`` to obtain Scene objects (multi-frame), then:
    1. Converts each history frame's Camera images into perception-model
       input format (PIL images + calibration matrices).
    2. Applies perception transforms (resize, normalize, BgrToYuv) to all
       frames.
    3. Extracts GT map features (route_lanes, speed_limit) from the Scene's
       map_api.
    4. Returns multi-frame data so the E2E Wrapper can run per-frame
       perception inference.

    Args:
        navsim_dataset: Config dict for ``NavsimDataset`` (built via
            registry).  Must have ``sensor_config`` that loads camera images
            for all history frames.
        transforms: Perception image transforms (same as perception training).
    """

    def __init__(
        self,
        navsim_dataset: Dataset,
        transforms: Optional[Callable] = None,
        sd_map_path: Optional[str] = None,
        osm_thickness: int = 5,
        canvas_size: Tuple[int, int] = (200, 200),
        patch_size: Tuple[int, int] = (100, 40),
        split_scene: Optional[bool] = False,
        return_bbox3d: bool = False,
        cam_names: Optional[Sequence[str]] = None,
        compute_planner_features: bool = False,
        planner_mode: str = "test",
    ):
        self.navsim_dataset = navsim_dataset
        self.transforms = transforms
        self.return_bbox3d = return_bbox3d
        self.cam_names = cam_names
        self._planner_feature_builder = None
        if compute_planner_features:
            from heal.data.datasets.navsim.navsim_feature_builder import (
                DiffusionPlannerFeatureBuilder,
            )

            self._planner_feature_builder = DiffusionPlannerFeatureBuilder(
                mode=planner_mode,
                return_bbox3d=return_bbox3d,
            )

        # osm_mask parameters
        self.osm_thickness = osm_thickness
        self.canvas_size = canvas_size
        self.patch_size = patch_size
        self.split_scene = split_scene

        # Load SD maps for osm_mask generation
        self.sd_maps = None
        if sd_map_path is not None:
            self.sd_maps = {}
            MAPS = [
                "sg-one-north",
                "us-ma-boston",
                "us-nv-las-vegas-strip",
                "us-pa-pittsburgh-hazelwood",
            ]
            MAP_EPSG = {
                "us-ma-boston": 32619,
                "us-pa-pittsburgh-hazelwood": 32617,
                "us-nv-las-vegas-strip": 32611,
                "sg-one-north": 32648,
            }
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
                "tertiary_link",
                "living_street",
                "road",
            ]
            for loc in MAPS:
                sd_map = gpd.read_file(os.path.join(sd_map_path, f"{loc}.shp"))
                assert sd_map.crs.to_epsg() == 4326
                sd_map = sd_map[sd_map["type"].isin(options)]
                epsg = MAP_EPSG[loc]
                transformer = Transformer.from_crs(
                    "EPSG:4326",
                    f"EPSG:{epsg}",
                    always_xy=True,
                )
                sd_map_topo_list = []
                for _, row in sd_map.iterrows():
                    local_pts = []
                    for lon, lat in list(row.geometry.coords):
                        x, y = transformer.transform(lon, lat)
                        local_pts.append((x, y))
                    if len(local_pts) >= 2:
                        sd_map_topo_list.append(LineString(local_pts))
                self.sd_maps[loc] = MultiLineString(sd_map_topo_list)

    def __len__(self):
        return len(self.navsim_dataset)

    @staticmethod
    def _get_patch_coord(
        patch_box: Tuple[float, float, float, float],
        patch_angle: float = 0.0,
    ):
        patch_x, patch_y, patch_h, patch_w = patch_box
        x_min = patch_x - patch_w / 5.0
        y_min = patch_y - patch_h / 2.0
        x_max = patch_x + patch_w / 5.0 * 4.0
        y_max = patch_y + patch_h / 2.0
        patch = box(x_min, y_min, x_max, y_max)
        patch = affinity.rotate(
            patch,
            patch_angle,
            origin=(patch_x, patch_y),
            use_radians=False,
        )
        return patch

    @staticmethod
    def _sample_fixed_pts_from_line(
        line,
        fixed_num=100,
    ):
        distances = np.linspace(0, line.length, fixed_num)
        sampled_points = np.array(
            [list(line.interpolate(d).coords) for d in distances]
        ).reshape(-1, 2)
        num_valid = len(sampled_points)
        if num_valid < fixed_num:
            padding = np.zeros((fixed_num - num_valid, 2))
            sampled_points = np.concatenate([sampled_points, padding], axis=0)
        else:
            sampled_points = sampled_points[:fixed_num]
        return sampled_points

    def _get_osm_geom(self, patch_box, patch_angle, location):
        osm_map = self.sd_maps[location]
        patch_x = patch_box[0]
        patch_y = patch_box[1]
        patch = self._get_patch_coord(patch_box, patch_angle)
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
                    new_line,
                    [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y],
                )
                line_list.append(new_line)
        return line_list

    def _line_osms_to_mask(
        self,
        lines: List[LineString],
        local_box: Sequence[float],
        canvas_size: Sequence[int],
        color: int = 1,
        thickness: int = 5,
    ):
        patch_x, patch_y, patch_h, patch_w = local_box
        patch = self._get_patch_coord(local_box)
        canvas_h = canvas_size[0]
        canvas_w = canvas_size[1]
        scale_height = canvas_h / patch_h
        scale_width = canvas_w / patch_w
        trans_x = -patch_x + patch_w / 5.0
        trans_y = -patch_y + patch_h / 2.0

        osm_mask = np.zeros(canvas_size, np.uint8)
        for line in lines:
            new_line = line.intersection(patch)
            if not new_line.is_empty:
                new_line = affinity.affine_transform(
                    new_line,
                    [1.0, 0.0, 0.0, 1.0, trans_x, trans_y],
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

    def _generate_osm_mask(
        self,
        ego_pose: np.ndarray,
        map_location: str,
    ) -> Optional[np.ndarray]:
        if self.sd_maps is None:
            return None
        x, y, heading = ego_pose
        patch_box = (x, y, self.patch_size[0], self.patch_size[1])
        patch_angle = heading / np.pi * 180

        # map_location format: e.g. "us-nv-las-vegas-strip"
        osm_geom = self._get_osm_geom(
            patch_box,
            patch_angle,
            map_location,
        )
        osm_vector_list = []
        for line in osm_geom:
            if not line.is_empty:
                if line.geom_type == "MultiLineString":
                    for single_line in line.geoms:
                        pts = self._sample_fixed_pts_from_line(
                            single_line,
                            fixed_num=50,
                        )
                        osm_vector_list.append(LineString(pts))
                elif line.geom_type == "LineString":
                    pts = self._sample_fixed_pts_from_line(
                        line,
                        fixed_num=50,
                    )
                    osm_vector_list.append(LineString(pts))

        local_box = (
            0.0,
            0.0,
            self.patch_size[0],
            self.patch_size[1],
        )
        osm_mask = self._line_osms_to_mask(
            osm_vector_list,
            local_box,
            self.canvas_size,
            thickness=self.osm_thickness,
        )
        return osm_mask

    def __getitem__(self, idx):
        navsim_data = self.navsim_dataset[idx]

        scene = navsim_data.get("scene", None)
        if scene is None:
            return None

        token = navsim_data["token"]
        num_history = scene.scene_metadata.num_history_frames

        # ── 1. Build per-frame perception input dicts ──────────────────
        frame_data_list = []
        for i in range(num_history):
            frame = scene.frames[i]
            lidar2global = frame.lidar2global
            if self.split_scene:
                frame_dict = _camera_to_perception_dict(
                    frame, lidar2global, token, self.cam_names
                )
            else:
                frame_dict = _camera_to_perception_dict(
                    frame, lidar2global, None, self.cam_names
                )
            # ── Generate osm_mask ──────────────────────────────────────
            ego_translation = frame.ego2global_translation
            ego_rotation = frame.ego2global_rotation
            w, x, y, z = ego_rotation
            heading = np.arctan2(
                2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)
            )
            anchor_ego_state = np.array(
                [ego_translation[0], ego_translation[1], heading],
                dtype=np.float64,
            )
            map_location = scene.scene_metadata.map_name
            osm_mask = self._generate_osm_mask(anchor_ego_state, map_location)
            frame_dict["osm_mask"] = torch.tensor(
                osm_mask, dtype=torch.float32
            )
            frame_data_list.append(frame_dict)

        # ── 2. Apply transforms to each frame's images ─────────────────
        if self.transforms is not None:
            for i in range(len(frame_data_list)):
                frame_data_list[i] = self.transforms(frame_data_list[i])

        # ── 2b. Planner features from Scene (eval path; no GT pack needed)
        planner_features = None
        if self._planner_feature_builder is not None:
            planner_features = self._planner_feature_builder.compute_features(
                scene
            )

        # ── 3. Assemble output ─────────────────────────────────────────
        # Use the *last* (current) frame's data as the primary "data" dict
        # for perception model compatibility.  Multi-frame data is stored
        # under "history_frames".
        current_frame_data = frame_data_list[-1]
        gt_features = navsim_data.get("features")
        if gt_features is None:
            gt_features = planner_features
        if not gt_features:
            result = {}
        else:
            result = {
                k: v
                for k, v in gt_features.items()
                if isinstance(v, torch.Tensor)
            }

        # Copy current-frame keys (img, lidar2img, etc.)
        for k, v in current_frame_data.items():
            result[k] = v

        result["token"] = token
        # result["features"] = map_features
        result["num_history_frames"] = num_history

        # Ego future GT in current-ego frame (x, y, heading) for eval / viz.
        if "ego_future_gt" not in result:
            future_traj = scene.get_future_trajectory()
            result["ego_future_gt"] = torch.tensor(
                future_traj.poses, dtype=torch.float32
            )

        # Store all history frames for per-frame perception inference.
        # Each entry is a dict with: img (list of tensors), lidar2img, etc.
        result["history_frames"] = frame_data_list

        # Per-frame ego poses (for coordinate transformation in wrapper)
        # result["ego_poses"] = ego_poses

        # Seq_meta compatibility: perception heads expect this key
        result["seq_meta"] = [current_frame_data]

        # Add GT annotation info for seq_meta (perception model heads
        # may need these during post-processing even in E2E mode)
        # current_frame = scene.frames[num_history - 1]
        result["gt_moving_objects_info"] = {
            "lidar_bboxes_labels": torch.zeros(
                0,
                11,
                dtype=torch.float32,
            ),
            "instance_ids": [],
        }
        result["gt_static_objects_info"] = {
            "lidar_bboxes_labels": torch.zeros(
                0,
                11,
                dtype=torch.float32,
            ),
            "instance_ids": [],
        }
        result["gt_discrete_map_info"] = {
            "lidar_bboxes_labels": torch.zeros(
                0,
                7,
                dtype=torch.float32,
            ),
        }

        return result


@OBJECT_REGISTRY.register
class NavsimE2EJointDataset(NavsimPerceptionMultitaskDataset):
    """Perception dataset extended with planning ground-truth.

    All perception-related fields are inherited from
    NavsimPerceptionMultitaskDataset.  On top of that, this class
    injects planning features:

    - ``ego_current_state``  (B, 9)
    - ``route_lanes``        (B, P, V, D)
    - ``route_lanes_speed_limit``    (B, 1, 1)
    - ``route_lanes_has_speed_limit`` (B, 1, 1)
    - ``neighbor_agents_past``  (B, 32, 4, 11)  — GT agents for training
    - ``ego_future_gt``        (B, T, 3)  — (x, y, heading)
    - ``neighbors_future_gt``  (B, 32, T, 3)

    During training, transforms such as ``StatePerturbation`` and
    ``NoiseScheduleSampler`` should be applied to the batch *after*
    collation (as batch_transforms in the processor).

    Args:
        with_planning_gt: Whether to load planning GT data.
        planner_lmdb_path: Path to pre-packed planner LMDB directory.
        planner_token_to_idx_path: Path to planner token_to_idx pkl.
        use_neighbors_future: Whether to load neighbor future trajectory.
    """

    def __init__(
        self,
        with_planning_gt: bool = True,
        planner_lmdb_path: Optional[str] = None,
        planner_token_to_idx_path: Optional[str] = None,
        use_neighbors_future: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_neighbors_future = use_neighbors_future
        self.with_planning_gt = with_planning_gt

        if self.with_planning_gt:
            assert planner_lmdb_path, "planner_lmdb_path is None"
            self._init_planner_lmdb(
                planner_lmdb_path, planner_token_to_idx_path
            )

    def _decode_lmdb(self, raw):
        """Decode msgpack-serialized bytes from LMDB."""

        def _decode_hook(obj):
            def _decode_bytes(obj):
                if isinstance(obj, bytes):
                    obj = obj.decode("utf-8")
                return obj

            new_obj = {}
            for k, v in obj.items():
                k = _decode_bytes(k)
                v = _decode_bytes(v)
                new_obj[k] = v
            return new_obj

        return msgpack.unpackb(raw, object_hook=_decode_hook, raw=True)

    def _init_planner_lmdb(self, lmdb_path: str, pkl_path: str):
        """Initialize planner LMDB reader."""

        pack_type_cls = get_packtype_from_path(lmdb_path)
        self._lmdb_pack_file = pack_type_cls(
            lmdb_path,
            writable=False,
        )
        self._lmdb_pack_file.open()
        with open(pkl_path, "rb") as f:
            self.token_to_idx = pickle.load(f)

    def __getitem__(self, idx):
        """Get a single sample with both perception & planning data."""
        data = super().__getitem__(idx)

        if not self.with_planning_gt:
            return data

        planner_gt = self._get_planning_gt(data)
        data.update(planner_gt)
        return data

    def _get_planning_gt(self, perception_data: dict) -> dict:
        """Fetch planning GT features for this sample."""
        result = {}

        # Try to extract sample_token from perception data
        sample_token = perception_data["sample_token"]

        result = self._get_planning_from_lmdb(sample_token)

        return result

    def _get_planning_from_lmdb(self, sample_token: str) -> dict:
        """Read planning features from LMDB."""
        idx = self.token_to_idx[sample_token]
        raw = self._lmdb_pack_file.read_idx(idx)
        info = self._decode_lmdb(raw)
        result = {
            "ego_current_state": _to_tensor(
                info["ego_current_state"], dtype=torch.float32
            ),
            "ego_future_gt": _to_tensor(
                info["ego_future_gt"], dtype=torch.float32
            ),
            "route_lanes": _to_tensor(
                info["route_lanes"], dtype=torch.float32
            ),
            "route_lanes_speed_limit": _to_tensor(
                info["route_lanes_speed_limit"], dtype=torch.float32
            ),
            "route_lanes_has_speed_limit": _to_tensor(
                info["route_lanes_has_speed_limit"], dtype=torch.bool
            ),
        }
        if self.use_neighbors_future:
            result["neighbors_future_gt"] = _to_tensor(
                info["neighbors_future_gt"], dtype=torch.float32
            )
            result["neighbor_agents_past"] = _to_tensor(
                info["neighbor_agents_past"], dtype=torch.float32
            )

        return result
