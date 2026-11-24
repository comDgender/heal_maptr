# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import hashlib
import io
import logging
import math
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import geopandas as gpd
import msgpack
import msgpack_numpy
import numpy as np
import torch
from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.apply_func import convert_tensor_v2 as to_tensor
from hat_engine.utils.pack_type import PackTypeMapper
from hat_engine.utils.pack_type.utils import get_packtype_from_path
from PIL import Image
from pyproj import Transformer
from shapely import affinity
from shapely.geometry import LineString, MultiLineString, box
from torch.utils.data import Dataset

from heal.data.datasets.navsim.navsim_perception_dataset_utils import (  # noqa
    discrete_map_process,
    get_lane_gt_by_nms,
    get_maps_api,
    get_neighbor_vector_set_map,
    map_process,
)
from heal.data.datasets.nuscenes.nuscenes_map_geometry import (
    LiDARInstanceLines,
    perspective,
)

try:
    from nuplan.common.actor_state.state_representation import Point2D
except ImportError:
    Point2D = None

msgpack_numpy.patch()

logger = logging.getLogger(__name__)

__all__ = [
    "NavsimPerceptionMultitaskDataset",
    "get_lane_gt_by_nms",
    "get_maps_api",
]


@OBJECT_REGISTRY.register
class NavsimPerceptionMultitaskDataset(Dataset):
    """Dataset object for packed navsim.

    Args:
        data_path: packed dataset path.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        max_samples (int, optional): Keep only the first N samples after
            load_interval. Useful for quick alignment runs. Defaults to None.
        transforms: A function transform that takes input
            sample and its target as entry and returns a transformed version.
        pack_type: The pack type.
        pack_kwargs: Kwargs for pack type.
        mode: train or test.
    """

    def __init__(
        self,
        data_path: str,
        load_interval: int = 1,
        max_samples: Optional[int] = None,
        transforms: Optional[Callable] = None,
        pack_type: Optional[str] = None,
        pack_kwargs: Optional[dict] = None,
        num_split: int = 1,
        with_moving_objects: bool = True,
        with_static_objects: bool = True,
        with_discrete_roadmarking: bool = True,
        with_continuous_map: bool = True,
        with_occ_gt: bool = True,
        filter_small_bbox: bool = False,
        bev_resolution: float = 0.8,
        mode: str = "train",
        map_pc_range: Optional[Sequence[float]] = None,
        aux_seg: Optional[dict] = None,
        canvas_size: Tuple[int, int] = (200, 200),
        patch_size: Tuple[int, int] = (100, 40),
        map_root: Optional[str] = None,
        map_version: str = "nuplan-maps-v1.0",
        sd_map_path: Optional[str] = None,
        osm_thickness: int = 5,
        map_do_nms: bool = False,
        map_min_length: float = 5.0,
        pc_range: Optional[Sequence[float]] = None,
        occ_pc_range: Optional[Sequence[float]] = None,
        occupancy_size: Tuple[int, int, int] = (0.5, 0.5, 0.5),
        lane_num: int = -1,
        lane_len: int = 20,
        cam_names: Optional[Sequence[str]] = None,
    ):
        self.root = data_path
        self.transforms = transforms
        self.pack_files = []
        self.samples = []

        self.kwargs = {} if pack_kwargs is None else pack_kwargs
        self.meta = {"scene_info": {}}
        self.map_class = {
            "lanes": 0,
        }
        self.mode = mode

        self.sampler = NavsimPerceptionSampler(
            with_moving_objects=with_moving_objects,
            with_static_objects=with_static_objects,
            with_discrete_roadmarking=with_discrete_roadmarking,
            with_continuous_map=with_continuous_map,
            with_occ_gt=with_occ_gt,
            filter_small_bbox=filter_small_bbox,
            bev_resolution=bev_resolution,
            map_root=map_root,
            map_version=map_version,
            map_pc_range=map_pc_range,
            canvas_size=canvas_size,
            patch_size=patch_size,
            aux_seg=aux_seg,
            sd_map_path=sd_map_path,
            osm_thickness=osm_thickness,
            map_do_nms=map_do_nms,
            map_min_length=map_min_length,
            pc_range=pc_range,
            occ_pc_range=occ_pc_range,
            occupancy_size=occupancy_size,
            lane_num=lane_num,
            lane_len=lane_len,
            cam_names=cam_names,
        )

        # Iterate over each rank's path, init pack_file and merge samples
        for idx, data_path in enumerate(self.root):
            try:
                cur_pack_type = get_packtype_from_path(data_path)
            except NotImplementedError:
                assert pack_type is not None
                cur_pack_type = PackTypeMapper(pack_type.lower())

            # Init the current rank's pack_file
            pack_file = cur_pack_type(data_path, writable=False, **self.kwargs)
            pack_file.open()
            self.pack_files.append(pack_file)

            # Get the current rank's samples, sampled by load_interval
            cur_samples = pack_file.get_keys()[::load_interval]
            cur_samples_with_rank = [(idx, sample) for sample in cur_samples]
            self.samples.extend(cur_samples_with_rank)

            cur_meta = self._decode_meta(pack_file)
            if self.meta is None:
                logger.warning(
                    "Packer data donot include meta. "
                )  # noqa [E501]
            else:
                self.meta["scene_info"].update(cur_meta["scene_info"])
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        if self.meta["scene_info"] != {}:
            org_len = len(self.meta["scene_info"])

            if num_split > 1:
                self._split_scenes(num_split)
                new_len = len(self.meta["scene_info"])
                print(f"Split scenes from {org_len} to {new_len}")

    def get_aug(self):
        aug_cfg = {}
        if self.transforms is not None:
            for trans in self.transforms.transforms:
                if hasattr(trans, "get_aug"):
                    name, cfg = trans.get_aug()
                    aug_cfg[name] = cfg
        return aug_cfg

    def set_aug(self, scene, aug_cfg):
        self.scenes_aug[scene] = aug_cfg

    def _split_scenes(self, num_split):
        new_scene_info = {}
        for k, v in self.meta["scene_info"].items():
            sequence_length = np.array(
                list(
                    range(
                        0,
                        v,
                        math.ceil(v / num_split),
                    )
                )
                + [v]
            )
            for idx, sub_seq_len in enumerate(
                sequence_length[1:] - sequence_length[:-1]
            ):
                name = k + f"_{idx}"
                new_scene_info[name] = sub_seq_len

        self.meta["scene_info"] = new_scene_info

    def get_meta(self):
        return self.meta

    def _decode_meta(self, pack_file):
        meta = self._decode(pack_file, "__meta__")
        return meta

    def _decode(self, pack_file, sample):
        def _decode_bytes(obj, is_binary_field: bool = False):
            """Decode bytes: only attempt UTF-8 for non-binary fields."""
            if isinstance(obj, bytes):
                if is_binary_field:
                    return obj
                try:
                    return obj.decode("utf-8")
                except UnicodeDecodeError:
                    return obj
            return obj

        def _decode_hook(obj):
            """Msgpack deserialization hook: distinguish binary fields."""
            if not isinstance(obj, dict):
                return _decode_bytes(obj)

            new_obj = {}
            binary_fields = [
                "img",
                "voxel_semantics",
                "mask_lidar",
                "mask_camera",
                "past_img_pil",
            ]

            for k, v in obj.items():
                k = _decode_bytes(k)

                if k == "img":
                    img_data = {}
                    for img_key, img_val in v.items():
                        if isinstance(img_val, list):
                            frame_list = []
                            for frame in img_val:
                                frame_dict = {}
                                for cam, cam_bytes in frame.items():
                                    frame_dict[_decode_bytes(cam)] = cam_bytes
                                frame_list.append(frame_dict)
                            img_data[img_key] = frame_list
                        else:
                            img_data[img_key] = _decode_bytes(
                                img_val, is_binary_field=True
                            )
                    new_obj[k] = img_data
                else:
                    is_binary = k in binary_fields
                    if isinstance(v, list):
                        v_decode = [
                            _decode_bytes(i, is_binary_field=is_binary)
                            for i in v
                        ]
                    else:
                        v_decode = _decode_bytes(v, is_binary_field=is_binary)
                    new_obj[k] = v_decode
            return new_obj

        sample = pack_file.read(sample)
        sample = msgpack.unpackb(sample, object_hook=_decode_hook, raw=True)
        return sample

    def __len__(self):
        return len(self.samples)

    def is_sample_valid(self, sample_data):
        if sample_data is None:
            return False
        assert isinstance(sample_data, Dict)
        return True

    def __getitem__(self, item):
        def _load_single_sample(idx, aug_cfg):
            """Load sample at given index (pure loading, no validation)."""
            rank_idx, sample_name = self.samples[idx]
            pack_file = self.pack_files[rank_idx]
            sample = self._decode(pack_file, sample_name)
            data = self.sampler(sample, idx)

            if data is not None:
                if aug_cfg:
                    data["scenes_aug"] = aug_cfg
                if self.transforms is not None:
                    data = self.transforms(data)
            return data, idx, sample_name

        def _get_valid_sample(init_idx, aug_cfg, retry_count=0):
            # Recursively obtain a valid sample
            # until found or retries exceeded.
            if retry_count > 100:
                raise RuntimeError(
                    f"Exceeded 100 retries without finding a valid sample, "
                    f"initial index: {init_idx}, please check the dataset!"
                )

            sample_data, curr_idx, sample_name = _load_single_sample(
                init_idx, aug_cfg
            )

            if self.mode == "train":
                if self.is_sample_valid(sample_data):
                    return sample_data
                else:
                    new_idx = np.random.randint(0, len(self) - 1)
                    print(
                        f"Sample[{curr_idx}]({sample_name}) invalid, "
                        f"replaced by sample[{new_idx}], "
                        f"retries: {retry_count+1}"
                    )
                    return _get_valid_sample(new_idx, aug_cfg, retry_count + 1)
            else:
                return sample_data

        aug_cfg = None
        if isinstance(item, dict):
            idx = item["idx"]
            aug_cfg = item["aug"]
        else:
            idx = item
        return _get_valid_sample(idx, aug_cfg)


class NavsimPerceptionSampler:

    MAPELEMENT2LABEL = {
        "lane_boundary_lines": 0,
        "others": -1,
    }

    def __init__(
        self,
        with_moving_objects: bool = True,
        with_static_objects: bool = True,
        with_discrete_roadmarking: bool = True,
        with_continuous_map: bool = True,
        with_occ_gt: bool = True,
        pc_range: Optional[Sequence[float]] = None,
        occ_pc_range: Optional[Sequence[float]] = None,
        occupancy_size: Tuple[int, int, int] = (0.5, 0.5, 0.5),
        filter_small_bbox: bool = False,
        bev_resolution: float = 0.8,
        map_root: Optional[str] = None,
        map_version: str = "nuplan-maps-v1.0",
        map_pc_range: Optional[Sequence[float]] = None,
        canvas_size: Tuple[int, int] = (200, 200),
        patch_size: Tuple[int, int] = (100, 40),
        aux_seg: Optional[dict] = None,
        sd_map_path: Optional[str] = None,
        osm_thickness: int = 5,
        map_do_nms: bool = False,
        map_min_length: float = 5.0,
        lane_num: int = -1,
        lane_len: int = 20,
        cam_names: Optional[Sequence[str]] = None,
    ):
        self.with_moving_objects = with_moving_objects
        self.with_static_objects = with_static_objects
        self.with_discrete_roadmarking = with_discrete_roadmarking
        self.with_continuous_map = with_continuous_map
        self.with_occ_gt = with_occ_gt
        self.filter_small_bbox = filter_small_bbox
        self.bev_resolution = bev_resolution
        self.map_pc_range = map_pc_range
        self.aux_seg = aux_seg
        self.canvas_size = canvas_size
        self.patch_size = patch_size
        self.pc_range = pc_range
        self.occ_pc_range = occ_pc_range
        self.occupancy_size = occupancy_size
        self.map_do_nms = map_do_nms
        self.map_min_length = map_min_length
        self.cam_names = cam_names
        self.moving_category_to_idx = {
            "vehicle": 0,
            "pedestrian": 1,
            "bicycle": 2,
        }
        self.static_category_to_idx = {
            "czone_sign": 0,
            "barrier": 1,
            "traffic_cone": 2,
            "generic_object": 3,
        }
        self.discrete_map_to_idx = {
            "stop_line": 0,
            "crosswalk": 1,
        }

        self.occ_xdim = int(
            (self.occ_pc_range[3] - self.occ_pc_range[0])
            / self.occupancy_size[0]
        )
        self.occ_ydim = int(
            (self.occ_pc_range[4] - self.occ_pc_range[1])
            / self.occupancy_size[1]
        )
        self.occ_zdim = int(
            (self.occ_pc_range[5] - self.occ_pc_range[2])
            / self.occupancy_size[2]
        )

        # map
        if self.with_continuous_map or self.with_discrete_roadmarking:
            assert map_root is not None, (
                "map_root must be provided when with_continuous_map "
                "or with_discrete_roadmarking is True"
            )
            self.MAPS = [
                "sg-one-north",
                "us-ma-boston",
                "us-nv-las-vegas-strip",
                "us-pa-pittsburgh-hazelwood",
            ]
            self.map_api_info = {}
            for loc in self.MAPS:
                self.map_api_info[loc] = get_maps_api(
                    map_root, map_version, loc
                )
            self._map_features = [
                "LANE",
                "LEFT_BOUNDARY",
                "RIGHT_BOUNDARY",
                "STOP_LINE",
                "CROSSWALK",
            ]
            self._max_elements = {
                "LANE": lane_num,
                "LEFT_BOUNDARY": lane_num,
                "RIGHT_BOUNDARY": lane_num,
                "STOP_LINE": lane_num,
                "CROSSWALK": lane_num,
            }
            self._max_points = {
                "LANE": lane_len,
                "LEFT_BOUNDARY": lane_len,
                "RIGHT_BOUNDARY": lane_len,
                "STOP_LINE": lane_len,
                "CROSSWALK": lane_len,
            }

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
                MAP_EPSG = {
                    "us-ma-boston": 32619,
                    "us-pa-pittsburgh-hazelwood": 32617,
                    "us-nv-las-vegas-strip": 32611,
                    "sg-one-north": 32648,
                }

                for loc in self.MAPS:
                    sd_map = gpd.read_file(
                        os.path.join(sd_map_path, "{}.shp".format(loc))
                    )
                    assert sd_map.crs.to_epsg() == 4326

                    sd_map = sd_map[sd_map["type"].isin(options)]
                    # WGS84 -> UTM
                    epsg = MAP_EPSG[loc]
                    transformer = Transformer.from_crs(
                        "EPSG:4326",
                        f"EPSG:{epsg}",
                        always_xy=True,
                    )
                    sd_map_topo_list = []
                    for _, row in sd_map.iterrows():
                        tmp_sd_data = list(row.geometry.coords)
                        local_pts = []
                        for lon, lat in tmp_sd_data:
                            x, y = transformer.transform(lon, lat)

                            local_pts.append((x, y))

                        if len(local_pts) >= 2:
                            sd_map_topo_list.append(LineString(local_pts))

                    self.sd_maps[loc] = MultiLineString(sd_map_topo_list)

    @staticmethod
    def bytes_to_pil(img_bytes):
        """Convert image byte stream to PIL image."""
        if isinstance(img_bytes, bytes):
            img_byte_arr = io.BytesIO(img_bytes)
            pil_img = Image.open(img_byte_arr).convert("RGB")
            return pil_img
        elif isinstance(img_bytes, Image.Image):
            return img_bytes.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(img_bytes)}")

    def get_category_objects(self, data, category_to_idx):
        if "anns" not in data or "gt_boxes" not in data["anns"]:
            return [], []
        anns = data["anns"]
        gt_boxes = anns["gt_boxes"]
        labels = anns["gt_names"]
        gt_velocity_3d = anns["gt_velocity_3d"]
        track_id = anns["track_tokens"]

        # Filter labels, keep only those in category_to_idx
        valid_indices = [
            i for i, label in enumerate(labels) if label in category_to_idx
        ]

        if len(valid_indices) == 0:
            return [], []

        # Filtered data
        filtered_gt_boxes = gt_boxes[valid_indices]  # (m, 7)
        filtered_gt_velocity_3d = gt_velocity_3d[valid_indices]  # (m, 3)
        filtered_labels = [labels[i] for i in valid_indices]
        filtered_track_id = [track_id[i] for i in valid_indices]

        # If pc_range is not None, filter out data outside the range
        if self.pc_range is not None:
            x_min, y_min, z_min, x_max, y_max, z_max = self.pc_range
            mask = (
                (filtered_gt_boxes[:, 0] >= x_min)
                & (filtered_gt_boxes[:, 0] <= x_max)
                & (filtered_gt_boxes[:, 1] >= y_min)
                & (filtered_gt_boxes[:, 1] <= y_max)
                & (filtered_gt_boxes[:, 2] >= z_min)
                & (filtered_gt_boxes[:, 2] <= z_max)
            )
            filtered_gt_boxes = filtered_gt_boxes[mask]
            filtered_gt_velocity_3d = filtered_gt_velocity_3d[mask]
            filtered_labels = [
                filtered_labels[i]
                for i in range(len(filtered_labels))
                if mask[i]
            ]
            filtered_track_id = [
                filtered_track_id[i]
                for i in range(len(filtered_track_id))
                if mask[i]
            ]

            if len(filtered_labels) == 0:
                return [], []

        label_ids = np.array(
            [category_to_idx[label] for label in filtered_labels]
        )[
            :, np.newaxis
        ]  # (m, 1)

        lidar_bboxes_labels = np.concatenate(
            [filtered_gt_boxes, filtered_gt_velocity_3d[:, :2], label_ids],
            axis=1,
        ).astype(np.float32)
        lidar_bboxes_labels = torch.from_numpy(lidar_bboxes_labels)

        track_id_int = [self.str_to_int(tid) for tid in filtered_track_id]

        return lidar_bboxes_labels, track_id_int

    def get_lane_objects(self, lanes_pecetion):
        gt_labels_map = []
        for _ in lanes_pecetion:
            gt_labels_map.append(self.map_class["lanes"])
            pass

    def str_to_int(self, s, bits=63):
        """Map string to unique int."""
        h = hashlib.sha256(s.encode("utf-8")).digest()
        value = int.from_bytes(h, byteorder="big")
        return value & ((1 << bits) - 1)

    def get_cam2lidar(self, rotation, translation):
        cam2lidar = np.eye(4)
        cam2lidar[:3, :3] = rotation
        cam2lidar[:3, 3] = translation
        return cam2lidar

    def get_lidar2img(self, intrinsic, cam2lidar):
        R = cam2lidar[:3, :3]
        t = cam2lidar[:3, 3]

        R = R.T
        t = -R @ t

        lidar2img = np.eye(4)
        lidar2img[:3, :4] = intrinsic @ np.hstack([R, t.reshape(3, 1)])

        return lidar2img

    def get_global2img(self, lidar2img, lidar2global):
        global2lidar = np.linalg.inv(lidar2global)
        global2img = lidar2img @ global2lidar
        return global2img

    def get_cam_by_name(self, data, name, lidar2global):
        img = self.bytes_to_pil(data["camera_data"][name])
        camera_info = data["cams"][name]
        cam2lidar_rotation = camera_info["sensor2lidar_rotation"]
        cam2lidar_translation = camera_info["sensor2lidar_translation"]
        cam_intrinsic = camera_info["cam_intrinsic"]
        cam2lidar = self.get_cam2lidar(
            cam2lidar_rotation, cam2lidar_translation
        )
        lidar2img = self.get_lidar2img(cam_intrinsic, cam2lidar)
        global2img = self.get_global2img(lidar2img, lidar2global)
        return {
            "img": img,
            "intrinsic": cam_intrinsic,
            "cam2lidar": cam2lidar,
            "lidar2img": lidar2img,
            "global2img": global2img,
        }

    def fliter_lane_gt_by_length(self, lane_data, min_length=5.0):
        lane_length = np.sum(
            np.linalg.norm(lane_data[:, 1:] - lane_data[:, :-1], axis=2),
            axis=1,
        )

        return lane_data[lane_length > min_length]

    def rm_out_of_range_points(self, lane_data, pc_range):
        eps = 1e-4
        x_min, y_min, _, x_max, y_max, _ = pc_range

        x = lane_data[..., 0]
        y = lane_data[..., 1]

        valid_mask = (
            (x >= x_min - eps)
            & (x <= x_max + eps)
            & (y >= y_min - eps)
            & (y <= y_max + eps)
        )

        lane_data_list = []
        for i in range(lane_data.shape[0]):
            inst = lane_data[i][valid_mask[i]]
            if inst.shape[0] > 1:
                lane_data_list.append(inst)

        return lane_data_list

    def get_patch_coord(
        self,
        patch_box: Tuple[float, float, float, float],
        patch_angle: float = 0.0,
    ):
        """Convert patch_box to shapely Polygon coordinates."""
        patch_x, patch_y, patch_h, patch_w = patch_box

        x_min = patch_x - patch_w / 5.0
        y_min = patch_y - patch_h / 2.0
        x_max = patch_x + patch_w / 5.0 * 4.0
        y_max = patch_y + patch_h / 2.0

        patch = box(x_min, y_min, x_max, y_max)
        patch = affinity.rotate(
            patch, patch_angle, origin=(patch_x, patch_y), use_radians=False
        )

        return patch

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
        patch = self.get_patch_coord(patch_box, patch_angle)
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

    def gen_vectorized_samples(
        self,
        lane_pts,
        lane_classes: str = "lane_boundary_lines",
        homo2img: Optional[List[np.ndarray]] = None,
        ego_pose=None,
        location=None,
    ):
        vectors = []
        for line in lane_pts:
            vectors.append(
                (
                    LineString(line),
                    self.MAPELEMENT2LABEL.get(lane_classes, -1),
                )
            )

        gt_labels = []
        gt_instance = []
        if self.aux_seg is not None:
            canvas_size = self.canvas_size
            if self.aux_seg["bev_seg"]:
                gt_semantic_mask = np.zeros(
                    (1, canvas_size[0], canvas_size[1]),
                    dtype=np.uint8,
                )
            else:
                gt_semantic_mask = None
            if self.aux_seg["pv_seg"]:
                num_cam = self.aux_seg["num_cam"]
                img_shape = self.aux_seg["img_shape"]
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
                                thickness=3,
                                canvas_size=canvas_size,
                                patch_size=self.patch_size,
                            )
                        if self.aux_seg["pv_seg"]:
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
            instance_line_list=gt_instance,
            fixed_num=20,
            padding_value=-10000,
            pc_range=self.map_pc_range,
        )

        # sdmap
        x, y, heading = ego_pose
        patch_box = (
            x,
            y,
            self.patch_size[0],
            self.patch_size[1],
        )
        patch_angle = heading / np.pi * 180
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

        anns_results = {
            "gt_vecs_pts_loc": gt_instance,
            "gt_vecs_label": gt_labels,
            "gt_semantic_mask": gt_semantic_mask,
            "gt_pv_semantic_mask": gt_pv_semantic_masks,
            "osm_mask": osm_mask if self.sd_maps is not None else None,
        }
        return anns_results

    def line_ego_to_pvmask(
        self, line_ego, mask, lidar2feat, color=1, thickness=1, z=-1.6
    ):
        distances = np.linspace(0, line_ego.length, 200)
        coords = np.array(
            [
                list(line_ego.interpolate(distance).coords)
                for distance in distances
            ]
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

    def line_ego_to_mask(
        self,
        line_ego: LineString,
        mask: np.ndarray,
        color: int = 1,
        thickness: int = 3,
        canvas_size: Tuple[int, int] = (200, 200),
        patch_size: Tuple[int, int] = (100, 40),
    ):
        """Rasterize a single line to mask.

        Args:
            line_ego: LineString representing the line.
            mask: Semantic mask to paint on.
            color: Positive label. Default is 1.
            thickness: Thickness of rasterized lines. Default is 3.
        """

        trans_x = canvas_size[1] / 5
        trans_y = canvas_size[0] / 2
        scale_x = canvas_size[1] / patch_size[1]
        scale_y = canvas_size[0] / patch_size[0]
        line_ego = affinity.scale(line_ego, scale_x, scale_y, origin=(0, 0))
        line_ego = affinity.affine_transform(
            line_ego, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y]
        )
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
        patch_x, patch_y, patch_h, patch_w = local_box
        patch = self.get_patch_coord(local_box)
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

    def obtain_points_label(self, occ):
        occ_index, occ_cls = occ[:, 0], occ[:, 1]

        # Check for invalid classes 1, 2, 3
        invalid_mask = np.isin(occ_cls, [1, 2, 3])
        if invalid_mask.any():
            raise ValueError(
                f"Found invalid class labels "
                f"{np.unique(occ_cls[invalid_mask])} in occ_cls. "
                f"Classes 1, 2, 3 are not allowed."
            )

        # Map classes: [0,4,5,6,7,8,9,10] -> [0,1,2,3,4,5,6,7]
        class_mapping = np.array(
            [0, -1, -1, -1, 1, 2, 3, 4, 5, 6, 7], dtype=np.int8
        )
        occ_cls = class_mapping[occ_cls]

        occ_grid = (
            np.ones(
                (self.occ_xdim, self.occ_ydim, self.occ_zdim), dtype=np.int8
            )
            * 8
        )
        x = occ_index % self.occ_xdim
        y = (occ_index // self.occ_xdim) % self.occ_ydim
        z = occ_index // (self.occ_xdim * self.occ_ydim)
        occ_grid[x, y, z] = occ_cls
        return occ_grid

    def __call__(self, data, idx):
        try:
            res_data = {}
            if self.cam_names is not None:
                cam_names = self.cam_names
            else:
                cam_names = list(data["cams"].keys())

            imgs = []
            lidar2img = []
            cam2lidar = []
            cam_intrin = []
            cam_global2img = []
            for name in cam_names:
                cam = self.get_cam_by_name(data, name, data["lidar2global"])
                imgs.append(cam["img"])
                cam2lidar.append(cam["cam2lidar"])
                cam_intrin.append(cam["intrinsic"])
                lidar2img.append(cam["lidar2img"])
                cam_global2img.append(cam["global2img"])

            res_data["img"] = imgs
            res_data["cam_name"] = cam_names
            res_data["lidar2img"] = lidar2img
            res_data["cam2lidar"] = cam2lidar
            res_data["camera_intrinsic"] = np.array(cam_intrin)
            res_data["layout"] = "chw"
            res_data["color_space"] = "rgb"
            res_data["timestamp"] = data["timestamp"]
            res_data["sample_token"] = data["token"]
            res_data["scene"] = data["scene_token"]
            res_data["ego2global"] = data["ego2global"]
            res_data["lidar2global"] = data["lidar2global"]
            res_data["lidar2ego"] = data["lidar2ego"]
            res_data["lidar2ego_translation"] = data["lidar2ego_translation"]
            res_data["lidar2ego_rotation"] = data["lidar2ego_rotation"]
            res_data["ego2global_translation"] = data["ego2global_translation"]
            res_data["ego2global_rotation"] = data["ego2global_rotation"]

            if self.with_continuous_map or self.with_discrete_roadmarking:
                if Point2D is None:
                    raise ImportError(
                        "nuplan is required for navsim map GT; "
                        "install nuplan-devkit."
                    )
                map_location = data["map_location"]
                map_api = self.map_api_info[map_location]

                ego_translation = data["ego2global_translation"]
                ego_rotation = data["ego2global_rotation"]
                w, x, y, z = ego_rotation
                heading = np.arctan2(
                    2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)
                )
                anchor_ego_state = np.array(
                    [ego_translation[0], ego_translation[1], heading],
                    dtype=np.float64,
                )
                ego_coords = Point2D(ego_translation[0], ego_translation[1])

            # moving objects
            if self.with_moving_objects:
                lidar_bboxes_labels, instance_ids = self.get_category_objects(
                    data, self.moving_category_to_idx
                )
                res_data["gt_moving_objects_info"] = {
                    "lidar_bboxes_labels": lidar_bboxes_labels,
                    "instance_ids": instance_ids,
                }

            # static objects
            if self.with_static_objects:
                lidar_bboxes_labels, instance_ids = self.get_category_objects(
                    data, self.static_category_to_idx
                )
                res_data["gt_static_objects_info"] = {
                    "lidar_bboxes_labels": lidar_bboxes_labels,
                    "instance_ids": instance_ids,
                }

            if self.with_continuous_map:
                route_roadblock_ids = data.get("roadblock_ids", [])

                coords, speed_limit, lane_route = get_neighbor_vector_set_map(
                    map_api, self._map_features, ego_coords, self.map_pc_range
                )

                vector_map = map_process(
                    route_roadblock_ids,
                    anchor_ego_state,
                    coords,
                    speed_limit,
                    lane_route,
                    self._map_features,
                    self._max_elements,
                    self._max_points,
                    False,
                )

                if self.map_do_nms:
                    if "filtered_lane_data_thr20" in vector_map:
                        lanes_pecetion = np.array(
                            vector_map["filtered_lane_data_thr20"],
                            dtype=np.float32,
                        )
                    else:
                        lanes_pecetion = vector_map["lanes_pecetion"]
                        lanes_pecetion = self.fliter_lane_gt_by_length(
                            lanes_pecetion, min_length=self.map_min_length
                        )
                        lanes_pecetion = get_lane_gt_by_nms(lanes_pecetion)

                lanes_pecetion = self.rm_out_of_range_points(
                    lanes_pecetion, pc_range=self.map_pc_range
                )
                lanes_anns = self.gen_vectorized_samples(
                    lanes_pecetion,
                    lane_classes="lane_boundary_lines",
                    homo2img=res_data["lidar2img"],
                    ego_pose=anchor_ego_state,
                    location=map_location,
                )
                gt_vecs_label = to_tensor(lanes_anns["gt_vecs_label"])
                if isinstance(
                    lanes_anns["gt_vecs_pts_loc"], LiDARInstanceLines
                ):
                    gt_vecs_pts_loc = lanes_anns["gt_vecs_pts_loc"]
                else:
                    gt_vecs_pts_loc = to_tensor(lanes_anns["gt_vecs_pts_loc"])
                    try:
                        gt_vecs_pts_loc = gt_vecs_pts_loc.flatten(1).to(
                            dtype=torch.float32
                        )
                    except BaseException:
                        gt_vecs_pts_loc = gt_vecs_pts_loc

                res_data["gt_labels_map"] = gt_vecs_label
                res_data["gt_instances"] = gt_vecs_pts_loc
                if lanes_anns["gt_semantic_mask"] is not None:
                    res_data["gt_seg_mask"] = to_tensor(
                        lanes_anns["gt_semantic_mask"]
                    )
                if lanes_anns["gt_pv_semantic_mask"] is not None:
                    res_data["gt_pv_seg_mask"] = [
                        to_tensor(mask)
                        for mask in lanes_anns["gt_pv_semantic_mask"]
                    ]
                if lanes_anns["osm_mask"] is not None:
                    res_data["osm_mask"] = to_tensor(lanes_anns["osm_mask"])

            # discrete map
            if self.with_discrete_roadmarking:
                roadmarking_bbox = discrete_map_process(
                    query_xy=ego_coords,
                    map_api=map_api,
                    map_pc_range=self.map_pc_range,
                    anchor_ego_state=anchor_ego_state,
                    discrete_map_to_idx=self.discrete_map_to_idx,
                    max_points=self._max_points,
                    return_rotated_rectangle=True,
                )
                res_data["gt_discrete_map_info"] = {
                    "lidar_bboxes_labels": roadmarking_bbox
                }

            # occ
            if self.with_occ_gt:
                if data["voxel_semantics"] is None:
                    voxel_semantics = np.full(
                        (self.occ_xdim, self.occ_ydim, self.occ_zdim),
                        8,
                        dtype=np.int8,
                    )
                    mask_camera = np.zeros(
                        (self.occ_xdim, self.occ_ydim, self.occ_zdim),
                    )
                else:
                    voxel_semantics = np.frombuffer(
                        data["voxel_semantics"], dtype=np.int32
                    ).reshape(-1, 2)
                    voxel_semantics = self.obtain_points_label(voxel_semantics)
                    mask_camera = np.ones(
                        (self.occ_xdim, self.occ_ydim, self.occ_zdim),
                    )
                res_data["gt_occ_info"] = {
                    "voxel_semantics": voxel_semantics,
                    "mask_camera": mask_camera,
                }

            return res_data
        except Exception:
            print(f"[VAL DATA ERROR] idx={idx}", flush=True)
            return None
