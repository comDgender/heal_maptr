"""Data loaders for sparse multitask det + MapTR + FlashOcc."""

import copy
import os

import torch
from common import (
    batch_size_per_gpu,
    bev_h_,
    bev_w_,
    data_rootdir,
    data_shape,
    dataloader_workers,
    enable_amp,
    fixed_ptsnum_per_gt_line,
    lidar_load_dim,
    lidar_train_lmdb,
    lidar_use_dim,
    lidar_val_lmdb,
    map_aux_seg_cfg,
    map_classes,
    map_point_cloud_range,
    meta_rootdir,
    num_lidar_sweeps,
    occ_train_lmdb,
    occ_val_lmdb,
    resize_aug,
    resize_shape,
    scale,
    use_lidar_gt,
)

from heal.data.collates.nusc_collates import collate_nuscenes

_vectorize_map_gt = dict(
    type="VectorizeMapGT",
    map_path=meta_rootdir,
    pc_range=map_point_cloud_range,
    map_classes=map_classes,
    fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
    padding_value=-10000,
    bev_size=(bev_h_, bev_w_),
    aux_seg=map_aux_seg_cfg,
    use_lidar_gt=use_lidar_gt,
)

train_transforms = [
    dict(
        type="MultiViewsImgResize",
        scales=tuple(x + scale for x in resize_aug),
    ),
    dict(type="MultiViewsImgCrop", size=data_shape[1:], random=True),
    dict(type="MultiViewsImgFlip"),
    dict(type="MultiViewsImgRotate", rot=(-5.4, 5.4)),
    dict(type="BevBBoxRotation", rotation_3d_range=(-0.3925, 0.3925)),
    dict(type="MultiViewsPhotoMetricDistortion", use_pil=False),
    dict(
        type="MultiViewsGridMask",
        use_h=True,
        use_w=True,
        rotate=1,
        offset=False,
        ratio=0.5,
        mode=1,
        prob=0.7,
    ),
    dict(
        type="MultiViewsImgTransformWrapper",
        transforms=[
            dict(type="PILToTensor"),
            dict(type="BgrToYuv444", rgb_input=True),
            dict(type="Normalize", mean=128, std=128),
        ],
    ),
    _vectorize_map_gt,
    dict(
        type="RasterizeMapPvSeg",
        aux_seg=map_aux_seg_cfg,
        use_lidar_gt=use_lidar_gt,
    ),
]

val_transforms = [
    dict(type="MultiViewsImgResize", size=resize_shape[1:]),
    dict(type="MultiViewsImgCrop", size=data_shape[1:]),
    _vectorize_map_gt,
    dict(
        type="MultiViewsImgTransformWrapper",
        transforms=[
            dict(type="PILToTensor"),
            dict(type="BgrToYuv444", rgb_input=True),
            dict(type="Normalize", mean=128, std=128),
        ],
    ),
]

_map_dataset_common = dict(
    map_path=meta_rootdir,
    pc_range=map_point_cloud_range,
    bev_size=(bev_h_, bev_w_),
    fixed_ptsnum_per_line=fixed_ptsnum_per_gt_line,
    padding_value=-10000,
    map_classes=map_classes,
    with_bev_bboxes=False,
    with_ego_bboxes=False,
    with_bev_mask=False,
    with_lidar_bboxes=True,
    use_lidar_gt=use_lidar_gt,
    aux_seg=map_aux_seg_cfg,
    num_split=2,
    filter_empty_gt=False,
    defer_vectormap=True,
)

_lidar_dataset_common = dict(
    type="NuscenesLidarDataset",
    num_sweeps=num_lidar_sweeps,
    load_dim=lidar_load_dim,
    use_dim=lidar_use_dim,
    filter_empty_gt=False,
    transforms=None,
)


def _multitask_dataset(
    base_lmdb_path,
    lidar_lmdb_path,
    occ_lmdb_path,
    transforms,
    test_mode=False,
):
    return dict(
        type="MergeDataset",
        join_key="sample_token",
        with_lidar_occ=True,
        transforms=None,  # transforms,
        datasets=[
            dict(
                type="NuscenesSparseMapDataset",
                data_path=base_lmdb_path,
                test_mode=test_mode,
                transforms=transforms,
                **_map_dataset_common,
            ),
            dict(
                **_lidar_dataset_common,
                data_path=lidar_lmdb_path,
                test_mode=test_mode,
            ),
            dict(
                type="NuscenesOccDataset",
                data_path=occ_lmdb_path,
                transforms=None,
            ),
        ],
    )


train_dataset = _multitask_dataset(
    os.path.join(data_rootdir, "train_lmdb"),
    lidar_train_lmdb,
    occ_train_lmdb,
    train_transforms,
    test_mode=False,
)
val_dataset = _multitask_dataset(
    os.path.join(data_rootdir, "val_lmdb"),
    lidar_val_lmdb,
    occ_val_lmdb,
    val_transforms,
    test_mode=True,
)

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=train_dataset,
    batch_sampler=dict(
        type="DistStreamBatchSampler",
        batch_size=batch_size_per_gpu,
        dataset=train_dataset,
        keep_consistent_seq_aug=True,
        skip_prob=0.0,
        sequence_flip_prob=0.0,
    ),
    num_workers=dataloader_workers,
    pin_memory=True,
    collate_fn=collate_nuscenes,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=val_dataset,
    num_workers=4,
    pin_memory=True,
    collate_fn=collate_nuscenes,
    batch_size=1,
    shuffle=False,
)

calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader["dataset"] = val_dataset
calibration_data_loader.pop("batch_sampler", None)
calibration_data_loader["batch_size"] = batch_size_per_gpu
calibration_data_loader["shuffle"] = False

int_infer_data_loader = copy.deepcopy(val_data_loader)


def loss_collector(outputs: dict):
    losses = []
    for _, loss in outputs.items():
        losses.append(loss)
    return losses


batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    loss_collector=loss_collector,
    enable_amp=enable_amp,
)

val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
)

calibration_batch_processor = copy.deepcopy(val_batch_processor)

# On-board BPU eval: batch_size=1, deterministic order. The HBM infer model
# (SparseMultitaskOEIrInfer, use_nv12=True) feeds raw images to the BPU
# pyramid input_source, which does YUV/NV12 on-chip, so drop BgrToYuv444 +
# Normalize that val transforms apply; keep only PILToTensor. The image
# transforms live on the nested NuscenesSparseMapDataset inside MergeDataset
# (dataset["datasets"][0]["transforms"]), so recurse to reach them.
hbm_data_loader = copy.deepcopy(val_data_loader)
hbm_data_loader["batch_size"] = 1
hbm_data_loader["shuffle"] = False


def _strip_yuv_normalize(node):
    if isinstance(node, dict):
        if node.get("type") == "MultiViewsImgTransformWrapper":
            node["transforms"] = [
                _sub
                for _sub in node.get("transforms", [])
                if _sub.get("type") not in ("BgrToYuv444", "Normalize")
            ]
        for _v in node.values():
            _strip_yuv_normalize(_v)
    elif isinstance(node, list):
        for _v in node:
            _strip_yuv_normalize(_v)


_strip_yuv_normalize(hbm_data_loader["dataset"])
hbm_batch_processor = copy.deepcopy(val_batch_processor)

# Flat single-sample dataset for tools/gen_demo.py + tools/inference.py demo
# generation. MergeDataset with transforms=None returns one merged sample dict
# per index (img/img_name/ego2img/lidar2img + lidar points); the infer.py
# gen_demo_data_saver writes those straight into demo/<task>/sample_idx_<i>.
# The nested NuscenesSparseMapDataset child still carries val_transforms
# (PILToTensor/BgrToYuv444/Normalize) which would turn imgs into tensors and
# break gen_demo_data_saver's PIL ``.save()``. Strip that child's transforms so
# demo samples keep raw PIL images; infer.py's process_inputs re-applies the
# needed image preprocessing for inference instead.
single_infer_dataset = copy.deepcopy(val_dataset)
single_infer_dataset["transforms"] = None
for _ds in single_infer_dataset["datasets"]:
    if _ds.get("type") == "NuscenesSparseMapDataset":
        _ds["transforms"] = None
