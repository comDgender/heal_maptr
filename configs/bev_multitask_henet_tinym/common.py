# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import os

from hat_engine.utils.config import ConfigVersion
from horizon_plugin_pytorch.march import March, set_march

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "bev_multitask_henet_tinym"
pretrain_ckpt_dir = "./tmp_pretrained_models/henet_tinym"

ckpt_dir = f"./tmp_models/{task_name}"

batch_size_per_gpu = 4
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
dataloader_workers = 4

# datadir settings
data_rootdir = "./tmp_data/heal_data/nuscenes/v1.0-trainval/"
meta_rootdir = "./tmp_data/heal_data/nuscenes/meta"
anchor_file = "./tmp_data/nuscenes/nuscenes_kmeans900.npy"

# Split lmdb layout: dynamic/static det + maptr + discrete_map stay in the
# base camera lmdb; occ GT is packed in a separate lmdb.
occ_train_lmdb = os.path.join(data_rootdir, "train_occ_lmdb")
occ_val_lmdb = os.path.join(data_rootdir, "val_occ_lmdb")

# anchor: {x, y, z, ln w,ln h, ln l,sin yaw, cos yaw, vx, vy, vz}
# gt: {x, y, z, ln w, ln h, ln l, yaw, vx, vy, cls}

num_machines = 2
num_steps_per_epoch = int(
    28130 // (len(device_ids) * batch_size_per_gpu * num_machines)
)

log_loss_show = 500

cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.NASH_M
set_march(march)
qat_mode = "fuse_bn"
convert_mode = "jit-strip"

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
bev_size = (51.2, 51.2, 0.8)
bev_h_ = 128
bev_w_ = 128

map_size = (20, 40, 0.8)
map_bev_h_ = 100
map_bev_w_ = 50

voxel_size = [0.2, 0.2, 8]
max_num_points = 20
max_voxels = (30000, 40000)

_dim_ = 256
_pos_dim_ = _dim_ // 2
_num_levels_ = 1
input_size = (512, 960)
use_lidar2img = True
num_decoder = 6
sparse_embed_dims = 256
sparse_num_groups = 8
sparse_num_levels = 1
sparse_drop_out = 0.1
num_single_frame_decoder = 0
use_adapter = True
with_track = True

add_dn_loss = True
if add_dn_loss:
    num_dn_groups = 5
    num_temp_dn_groups = 3
else:
    num_dn_groups = -1
    num_temp_dn_groups = -1

max_camoverlap_num = 2

lidar_input = False
camera_input = True

use_dynamic_det_head = True
use_static_objects_det_head = True
use_occ_head = True
use_discrete_map_head = True
use_maptr_head = True

dynamic_objects_class_names = (
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
)
dynamic_objects_num_classes = len(dynamic_objects_class_names)
dynamic_objects_detection_mapping = {
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.construction": "construction_vehicle",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}

static_objects_class_names = (
    "traffic_cone",
    "barrier",
)
static_objects_num_classes = len(static_objects_class_names)
static_objects_detection_mapping = {
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
}

occ3d_seg_class = [
    "others",
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
]
num_classes_occ = 18

discrete_map_class_names = ["stop_line"]
discrete_map_num_classes = len(discrete_map_class_names)
discrete_map_num_query = 100
discrete_map_bbox_size = 6
discrete_map_code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

hidden_dim = 64
num_vec_one2one = 50
num_vec_one2many = 300
fixed_ptsnum_per_gt_line = 20
fixed_ptsnum_per_pred_line = 20
head_embed_dims = 512
map_classes = ["divider", "ped_crossing", "boundary"]
num_map_classes = len(map_classes)
aux_seg_cfg = dict(
    use_aux_seg=True,
    bev_seg=True,
    pv_seg=False,
    seg_classes=1,
    feat_down_sample=32,
    pv_thickness=1,
)
eval_use_same_gt_sample_num_flag = True
map_point_cloud_range = [-20.0, -40.0, -5, 20.0, 40.0, 3.0]


def get_feature_map_size(point_cloud_range, voxel_size):
    import numpy as np

    point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
    voxel_size = np.array(voxel_size, dtype=np.float32)
    grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
    grid_size = np.round(grid_size).astype(np.int64)
    return grid_size


# --- 3-stage float training params (shared by model.py + train.py) ---
# stage1: dynamic + static + occ + discrete map
float_lr_stage1 = 1e-4
num_epochs_stage1 = 200
num_steps_stage1 = num_steps_per_epoch * num_epochs_stage1

# stage2: maptr only (frozen backbone/neck/view_transformer/fuse)
# Align with jiaqi float_maptr_freeze: 100 epochs → 43900 steps.
float_lr_stage2 = 2.5e-4
num_epochs_stage2 = 100
num_steps_stage2 = num_steps_per_epoch * num_epochs_stage2

# stage3: full tasks (align jiaqi float_joint: 2e-5 / 200 epochs → 87800 steps)
float_lr_stage3 = 2e-5
num_epochs_stage3 = 200
num_steps_stage3 = num_steps_per_epoch * num_epochs_stage3
