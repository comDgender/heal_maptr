"""Common settings for BevFusion PointPillars + HENet multisensor multitask NuScenes."""  # noqa: E501

import os

from hat_engine.utils.config import ConfigVersion
from horizon_plugin_pytorch.march import March

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "bevfusion_pointpillar_henet_multisensor_multitask"
lidar_task_name = "centerpoint_pointpillar"
camera_task_name = "bevformer_henet_camera_multitask_pretrained"

batch_size_per_gpu = 4
dataloader_workers = 4
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]

ckpt_dir = f"./tmp_models/{task_name}"
lidar_ckpt_dir = "./tmp_pretrained_models/centerpoint_pointpillar"
# Camera multitask pretrain checkpoint (same path as HAT source).
camera_ckpt_dir = f"/horizon-bucket/hat_data/users/rui.xu/heal_release_models/nash/{camera_task_name}"  # noqa: E501

data_rootdir = "./tmp_data/heal_data/nuscenes/v1.0-trainval/"
meta_rootdir = "./tmp_data/heal_data/nuscenes/meta"
lidar_train_lmdb = os.path.join(data_rootdir, "train_lidar_lmdb")
lidar_val_lmdb = os.path.join(data_rootdir, "val_lidar_lmdb")
occ_train_lmdb = os.path.join(data_rootdir, "train_occ_lmdb")
occ_val_lmdb = os.path.join(data_rootdir, "val_occ_lmdb")

log_loss_show = 200
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.NASH_M
norm_cfg = None
qat_mode = "fuse_bn"
convert_mode = "jit-strip"
enable_amp = True

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
bev_size = (51.2, 51.2, 0.8)
bev_size_occ = (40, 40)
voxel_size = [0.2, 0.2, 8]
max_num_points = 20
max_voxels = (30000, 40000)

_dim_ = 256
_pos_dim_ = _dim_ // 2
_num_levels_ = 1
bev_h_ = 128
bev_w_ = 128
input_size = (512, 960)
map_size = (15, 30, 0.15)
use_lidar2img = True
with_ego_occ = False
max_camoverlap_num = 2
bev_sparse_rate = 0.4

lidar_input = True
camera_input = True
use_occ_head = True
use_bev_head = True

num_lidar_sweeps = 9
lidar_load_dim = 5
lidar_use_dim = [0, 1, 2, 3, 4]

num_classes = 10
num_classes_occ = 18

class_names = (
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
)

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

float_epochs = 24
calibration_step = 100
val_interval = 1

float_lr = 4e-4
