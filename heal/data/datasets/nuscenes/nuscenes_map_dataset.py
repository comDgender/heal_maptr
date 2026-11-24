# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
import logging
import os
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

try:
    import geopandas  # noqa: F401
    from nuscenes.eval.common.utils import Quaternion, quaternion_yaw
    from nuscenes.map_expansion.map_api import (  # noqa: F401
        NuScenesMap,
        NuScenesMapExplorer,
    )
except ImportError as e:  # pragma: no cover - surfaces the real failing import
    raise ImportError(
        "nuscenes-devkit / geopandas are required for map vector GT "
        "(NuscenesMapDataset / apply_online_vectormap). The previous code "
        "silently set Quaternion=None here, which later crashed as "
        "'NoneType is not callable' inside the DataLoader worker. "
        "Install nuscenes-devkit==1.1.9 and geopandas==0.10.2 (fiona, "
        "Shapely, pyproj) with a compatible numpy/pandas."
    ) from e

from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.apply_func import convert_tensor_v2 as to_tensor
from hat_engine.utils.pack_type import PackTypeMapper, get_packtype_from_path
from hat_engine.utils.package_helper import require_packages
from shapely.geometry import LineString

from heal.data.datasets.nuscenes.nuscenes_map_geometry import (
    LiDARInstanceLines,
    VectorizedLocalMap,
)
from .nuscenes_dataset import NuscenesBevDataset

logger = logging.getLogger(__name__)

__all__ = [
    "LiDARInstanceLines",
    "NuscenesMapDataset",
    "VectorizedLocalMap",
    "apply_online_vectormap",
    "ensure_lidar2global",
    "get_lidar2ego",
]


def perspective(cam_coords, proj_mat):
    pix_coords = proj_mat @ cam_coords
    valid_idx = pix_coords[2, :] > 0
    pix_coords = pix_coords[:, valid_idx]
    pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + 1e-7)
    pix_coords = pix_coords.transpose(1, 0)
    return pix_coords


PV_PROJ_EGO_GROUND_Z = -1.6


def pv_proj_ground_z(
    use_lidar_gt: bool,
    lidar2ego: Optional[np.ndarray] = None,
    ego_ground_z: float = PV_PROJ_EGO_GROUND_Z,
) -> float:
    """Ground height in the active map frame for PV line projection.

    Map lines are stored as 2D (x, y) on the local ground plane. MapTR uses
    ``ego_ground_z`` in ego frame; in lidar frame the same physical ground
    point must be transformed via ``lidar2ego``.
    """
    if not use_lidar_gt:
        return ego_ground_z
    if lidar2ego is None:
        return ego_ground_z
    lidar2ego = np.asarray(lidar2ego, dtype=np.float64)
    ego2lidar = np.linalg.inv(lidar2ego)
    pt_lidar = ego2lidar @ np.array([0.0, 0.0, ego_ground_z, 1.0])
    return float(pt_lidar[2])


def get_lidar2ego(example):
    if "lidar2ego" in example:
        return np.asarray(example["lidar2ego"], dtype=np.float64)
    lidar2ego = np.eye(4, dtype=np.float64)
    lidar2ego[:3, :3] = Quaternion(
        example["meta"]["sensor2ego_rotation"]
    ).rotation_matrix
    lidar2ego[:3, 3] = example["meta"]["sensor2ego_translation"]
    return lidar2ego


def ensure_lidar2global(example):
    if "lidar2global" in example:
        return example
    lidar2ego = np.eye(4)
    lidar2ego[:3, :3] = Quaternion(
        example["meta"]["sensor2ego_rotation"]
    ).rotation_matrix
    lidar2ego[:3, 3] = example["meta"]["sensor2ego_translation"]
    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(
        example["meta"]["ego2global_rotation"]
    ).rotation_matrix
    ego2global[:3, 3] = example["meta"]["ego2global_translation"]
    example["lidar2global"] = ego2global @ lidar2ego
    return example


def apply_online_vectormap(
    example, vector_map, use_lidar_gt=True, aux_seg=None
):
    """Build map vector GT from the current (possibly augmented) sample."""
    example = ensure_lidar2global(example)
    lidar2global = example["lidar2global"]
    translation = list(lidar2global[:3, 3])
    rotation = list(Quaternion(matrix=lidar2global).q)
    homo2img = example["lidar2img"]

    if not use_lidar_gt:
        translation = example["meta"]["ego2global_translation"]
        rotation = example["meta"]["ego2global_rotation"]
        homo2img = example["ego2img"]

    location = example["meta"]["location"]
    num_cam = len(example["img"])
    if aux_seg:
        if isinstance(example["img"][0], Image.Image):
            img_shape = example["img"][0].size[::-1]
        else:
            img_shape = example["img"][0].shape[1:]
    else:
        img_shape = None

    anns_results = vector_map.gen_vectorized_samples(
        location,
        translation,
        rotation,
        img_shape=img_shape,
        homo2img=homo2img,
        num_cam=num_cam,
        pv_proj_z=pv_proj_ground_z(
            use_lidar_gt,
            get_lidar2ego(example) if use_lidar_gt else None,
        ),
    )

    gt_vecs_label = to_tensor(anns_results["gt_vecs_label"])
    if isinstance(anns_results["gt_vecs_pts_loc"], LiDARInstanceLines):
        gt_vecs_pts_loc = anns_results["gt_vecs_pts_loc"]
    else:
        gt_vecs_pts_loc = to_tensor(anns_results["gt_vecs_pts_loc"])
        try:
            gt_vecs_pts_loc = gt_vecs_pts_loc.flatten(1).to(
                dtype=torch.float32
            )
        except BaseException:
            gt_vecs_pts_loc = gt_vecs_pts_loc
    example["gt_labels_map"] = gt_vecs_label
    example["gt_instances"] = gt_vecs_pts_loc
    if anns_results["osm_vectors"] is not None:
        example["osm_vectors"] = anns_results["osm_vectors"]
    if anns_results["osm_mask"] is not None:
        example["osm_mask"] = to_tensor(anns_results["osm_mask"])
    if anns_results["gt_semantic_mask"] is not None:
        example["gt_seg_mask"] = to_tensor(anns_results["gt_semantic_mask"])
    if anns_results["gt_pv_semantic_mask"] is not None:
        example["gt_pv_seg_mask"] = [
            to_tensor(mask) for mask in anns_results["gt_pv_semantic_mask"]
        ]
    return example


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


def _get_pv_img_shape(example):
    if isinstance(example["img"][0], Image.Image):
        return example["img"][0].size[::-1]
    return example["img"][0].shape[1:]


def _get_pv_homo2img_from_example(example, use_lidar_gt):
    if use_lidar_gt:
        return example["lidar2img"]
    return example["ego2img"]


def _get_lidar2ego_from_example(example):
    if "lidar2ego" in example:
        return np.asarray(example["lidar2ego"], dtype=np.float64)
    lidar2ego = np.eye(4, dtype=np.float64)
    lidar2ego[:3, :3] = Quaternion(
        example["meta"]["sensor2ego_rotation"]
    ).rotation_matrix
    lidar2ego[:3, 3] = example["meta"]["sensor2ego_translation"]
    return lidar2ego


def rasterize_pv_seg_masks(
    example, instance_lines, aux_seg, use_lidar_gt=True
):
    """Rasterize map polylines into per-camera PV segmentation masks."""
    if not aux_seg or not aux_seg.get("pv_seg"):
        return example

    num_cam = len(example["img"])
    img_shape = _get_pv_img_shape(example)
    homo2img = _get_pv_homo2img_from_example(example, use_lidar_gt)
    pv_proj_z = pv_proj_ground_z(
        use_lidar_gt,
        _get_lidar2ego_from_example(example) if use_lidar_gt else None,
    )
    gt_pv_semantic_masks = []
    for _, stride in aux_seg["feat_down_sample"]:
        gt_pv_semantic_mask = np.zeros(
            (
                num_cam,
                1,
                img_shape[0] // stride,
                img_shape[1] // stride,
            ),
            dtype=np.uint8,
        )
        scale_factor = np.eye(4)
        scale_factor[0, 0] *= 1 / stride
        scale_factor[1, 1] *= 1 / stride
        cam2feat = [scale_factor @ h2i for h2i in homo2img]
        for instance in instance_lines:
            if instance.geom_type == "LineString":
                for cam_index in range(num_cam):
                    line_ego_to_pvmask(
                        instance,
                        gt_pv_semantic_mask[cam_index][0],
                        cam2feat[cam_index],
                        color=1,
                        thickness=aux_seg["pv_thickness"],
                        z=pv_proj_z,
                    )
        gt_pv_semantic_masks.append(gt_pv_semantic_mask)
    example["gt_pv_seg_mask"] = [
        to_tensor(mask) for mask in gt_pv_semantic_masks
    ]
    return example


@OBJECT_REGISTRY.register
class NuscenesMapDataset(NuscenesBevDataset):
    """Dataset object for packed NuScenes.

    This dataset adds static map elements.

    Args:
        pc_range: Range of the point cloud.
        queue_length: Length of the queue.
        bev_size: Size of the BEV image. Default is (200, 200).
        fixed_ptsnum_per_line: Fixed number of points per line. Default is -1.
        padding_value: Value to use for padding. Default is -10000.
        map_classes: Tuple of map classes. Default is None.
        map_path: Path to the map. Default is None.
        aux_seg: Auxiliary segmentation information. Default is None.
        test_mode: Whether in test mode. Default is False.
        filter_empty_gt: Whether to filter empty ground truth. Default is True.
        use_lidar_gt: Whether to use LiDAR ground truth. Default is True.
        sd_map_path: Path to the SD map. Default is None.
        **kwargs: Additional keyword arguments.
    """

    MAPCLASSES: Tuple[str] = ("divider",)

    @require_packages("nuscenes")
    def __init__(
        self,
        pc_range: List[int],
        queue_length: int = 1,
        bev_size: Tuple[int, int] = (200, 200),
        fixed_ptsnum_per_line: int = -1,
        padding_value: int = -10000,
        map_classes: Optional[Tuple[str]] = None,
        map_path: Optional[str] = None,
        aux_seg: Optional[any] = None,
        test_mode: Optional[bool] = False,
        filter_empty_gt: Optional[bool] = True,
        use_lidar_gt: bool = True,
        sd_map_path: Optional[str] = None,
        map_gt_path: Optional[str] = None,
        defer_vectormap: bool = False,
        **kwargs,
    ):
        super(NuscenesMapDataset, self).__init__(**kwargs)

        self.queue_length = queue_length
        self.bev_size = bev_size
        self.defer_vectormap = defer_vectormap

        self.MAPCLASSES = self.get_map_classes(map_classes)
        self.NUM_MAPCLASSES = len(self.MAPCLASSES)
        self.pc_range = pc_range
        patch_h = pc_range[4] - pc_range[1]
        patch_w = pc_range[3] - pc_range[0]
        self.patch_size = (patch_h, patch_w)
        self.padding_value = padding_value
        self.fixed_num = fixed_ptsnum_per_line
        self.aux_seg = aux_seg
        self.vector_map = VectorizedLocalMap(
            map_path,
            patch_size=self.patch_size,
            map_classes=self.MAPCLASSES,
            fixed_ptsnum_per_line=fixed_ptsnum_per_line,
            padding_value=self.padding_value,
            canvas_size=bev_size,
            aux_seg=aux_seg,
            sd_map_path=sd_map_path,
        )

        self.filter_empty_gt = filter_empty_gt
        self.test_mode = test_mode
        self.flag = np.zeros(len(self), dtype=np.uint8)
        self.use_lidar_gt = use_lidar_gt
        self.map_gt_path = map_gt_path
        self.map_gt_pack = None
        self.map_gt_samples = None
        if map_gt_path:
            if os.path.isdir(map_gt_path):
                self._open_map_gt_pack(map_gt_path)
            else:
                logger.warning(
                    "map_gt_path %s not found, fallback to online map GT",
                    map_gt_path,
                )

    def _open_map_gt_pack(self, path: str):
        try:
            pack_type = get_packtype_from_path(path)
        except NotImplementedError:
            pack_type = PackTypeMapper["lmdb"]
        self.map_gt_pack = pack_type(path, writable=False)
        self.map_gt_pack.open()
        self.map_gt_samples = self.map_gt_pack.get_keys()
        if len(self.map_gt_samples) != len(self.samples):
            raise ValueError(
                f"map GT pack length {len(self.map_gt_samples)} != "
                f"base pack length {len(self.samples)} for {path}"
            )
        logger.info("Loaded precomputed map GT from %s", path)

    def _get_lidar2ego(self, example):
        return get_lidar2ego(example)

    def _pv_proj_ground_z(self, example):
        return pv_proj_ground_z(
            self.use_lidar_gt,
            self._get_lidar2ego(example) if self.use_lidar_gt else None,
        )

    def _pv_homo2img(self, example):
        if self.use_lidar_gt:
            return example["lidar2img"]
        return example["ego2img"]

    def _load_precomputed_map_gt(self, idx: int):
        from heal.data.datasets.nuscenes.nuscenes_map_gt import (
            build_lidar_instance_lines,
            decode_map_gt_sample,
        )

        raw = self.map_gt_pack.read(self.map_gt_samples[idx])
        sample = decode_map_gt_sample(raw)
        gt_instances = build_lidar_instance_lines(
            sample["gt_instance_coords"],
            patch_size=self.patch_size,
            fixed_num=self.fixed_num,
            padding_value=self.padding_value,
        )
        return (
            sample["gt_labels_map"],
            gt_instances,
            gt_instances.instance_list,
        )

    def _ensure_lidar2global(self, example):
        return ensure_lidar2global(example)

    def _map_patch_from_example(self, example):
        if not self.use_lidar_gt:
            translation = list(example["meta"]["ego2global_translation"])
            rotation = list(example["meta"]["ego2global_rotation"])
        else:
            example = self._ensure_lidar2global(example)
            lidar2global = example["lidar2global"]
            translation = list(lidar2global[:3, 3])
            rotation = list(Quaternion(matrix=lidar2global).q)
        map_pose = translation[:2]
        patch_box = (
            map_pose[0],
            map_pose[1],
            self.patch_size[0],
            self.patch_size[1],
        )
        patch_angle = quaternion_yaw(Quaternion(rotation)) / np.pi * 180
        location = example["meta"]["location"]
        return location, patch_box, patch_angle

    def _rasterize_bev_seg(self, example, instance_lines):
        if not self.aux_seg or not self.aux_seg.get("bev_seg"):
            return example

        gt_semantic_mask = np.zeros(
            (1, self.bev_size[0], self.bev_size[1]),
            dtype=np.uint8,
        )
        for instance in instance_lines:
            if instance.geom_type == "LineString":
                self.vector_map.line_ego_to_mask(
                    instance,
                    gt_semantic_mask[0],
                    color=1,
                    thickness=self.vector_map.thickness,
                )
        example["gt_seg_mask"] = to_tensor(gt_semantic_mask)
        return example

    def _apply_osm_gt(self, example):
        if self.vector_map.sd_maps is None:
            return example

        location, patch_box, patch_angle = self._map_patch_from_example(
            example
        )
        osm_geom = self.vector_map.get_osm_geom(
            patch_box, patch_angle, location
        )
        osm_vector_list = []
        for line in osm_geom:
            if line.is_empty:
                continue
            if line.geom_type == "MultiLineString":
                for single_line in line.geoms:
                    pts, pts_num = self.vector_map.sample_fixed_pts_from_line(
                        single_line, padding=False, fixed_num=50
                    )
                    if pts_num >= 2:
                        osm_vector_list.append(LineString(pts))
            elif line.geom_type == "LineString":
                pts, pts_num = self.vector_map.sample_fixed_pts_from_line(
                    line, padding=False, fixed_num=50
                )
                if pts_num >= 2:
                    osm_vector_list.append(LineString(pts))
            else:
                raise NotImplementedError

        local_box = (0.0, 0.0, self.patch_size[0], self.patch_size[1])
        osm_mask = self.vector_map.line_osms_to_mask(
            osm_vector_list,
            local_box,
            self.bev_size,
            thickness=self.vector_map.osm_thickness,
        )
        osm_vectors = LiDARInstanceLines(
            osm_vector_list,
            self.vector_map.sample_dist,
            self.vector_map.num_samples,
            self.vector_map.padding,
            self.vector_map.fixed_num,
            self.vector_map.padding_value,
            patch_size=self.patch_size,
        )
        example["osm_vectors"] = osm_vectors
        example["osm_mask"] = to_tensor(osm_mask)
        return example

    def _rasterize_pv_seg(self, example, instance_lines):
        if self.aux_seg and self.aux_seg.get("defer_pv_seg_rasterize"):
            return example
        return rasterize_pv_seg_masks(
            example, instance_lines, self.aux_seg, self.use_lidar_gt
        )

    def __getstate__(self):
        state = super().__getstate__()
        if self.map_gt_pack is not None:
            self.map_gt_pack.close()
        state["map_gt_pack"] = None
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        if self.map_gt_path and os.path.isdir(self.map_gt_path):
            self._open_map_gt_pack(self.map_gt_path)

    @classmethod
    def get_map_classes(
        cls, map_classes: Optional[Sequence[str]] = None
    ) -> List[str]:
        """Get class names of current dataset.

        Args:
            map_classes : If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Return:
            A list of class names.
        """
        if map_classes is None:
            return cls.MAPCLASSES

        if isinstance(map_classes, (tuple, list)):
            class_names = map_classes
        else:
            raise ValueError(
                f"Unsupported type {type(map_classes)} of map classes."
            )

        return class_names

    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def vectormap_pipeline(self, example, idx=None):
        if self.map_gt_pack is not None and idx is not None:
            (
                gt_labels,
                gt_instances,
                instance_lines,
            ) = self._load_precomputed_map_gt(idx)
            example["gt_labels_map"] = to_tensor(gt_labels)
            example["gt_instances"] = gt_instances
            example = self._rasterize_bev_seg(example, instance_lines)
            example = self._rasterize_pv_seg(example, instance_lines)
            return self._apply_osm_gt(example)

        return apply_online_vectormap(
            example,
            self.vector_map,
            use_lidar_gt=self.use_lidar_gt,
            aux_seg=self.aux_seg,
        )

    def prepare_train_data(self, idx, aug_cfg=None):
        start = idx
        end = idx - self.queue_length
        queue_list = list(range(start, end, -1))

        # queue_list = [item]
        # index_list = list(range(item-self.queue_length, item))
        # random.shuffle(index_list)
        # index_list = sorted(index_list[1:], reverse=True)
        # queue_list.extend(index_list)

        seq_data = []
        for i in queue_list:
            idx = max(i, 0)
            fetch_item = (
                {"idx": idx, "aug": aug_cfg} if aug_cfg is not None else idx
            )
            example = super().__getitem__(fetch_item)
            if not self.defer_vectormap:
                example = self.vectormap_pipeline(example, idx=idx)
            if self.filter_empty_gt and (
                example is None or ~(example["gt_labels_map"] != -1).any()
            ):
                return None

            seq_data.append(example)

        return seq_data

    def prepare_test_data(self, item):
        # example = super().__getitem__(item)
        # example = self.vectormap_pipeline(example)
        start = item
        end = item - self.queue_length
        seq_data = []

        for i in range(start, end, -1):
            idx = max(i, 0)
            example = super().__getitem__(idx)
            if not self.defer_vectormap:
                example = self.vectormap_pipeline(example, idx=idx)

            seq_data.append(example)
        return seq_data

        # return [example]

    def __getitem__(self, item):
        if isinstance(item, dict):
            idx = item["idx"]
            aug_cfg = item["aug"]
        else:
            idx = item
            aug_cfg = None
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx, aug_cfg)
            if data is None:
                idx = self._rand_another(idx)
                continue

            return data
