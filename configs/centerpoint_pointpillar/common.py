"""Common settings for CenterPoint PointPillars NuScenes detection."""

import os

from hat_engine.utils.config import ConfigVersion
from horizon_plugin_pytorch.march import March, set_march

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "centerpoint_pointpillar"
batch_size_per_gpu = 4
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]

ckpt_dir = f"./tmp_models/{task_name}"
data_rootdir = "./tmp_data/heal_data/nuscenes/v1.0-trainval"
meta_rootdir = "./tmp_data/heal_data/nuscenes/meta"
lidar_train_lmdb = os.path.join(data_rootdir, "train_lidar_lmdb")
lidar_val_lmdb = os.path.join(data_rootdir, "val_lidar_lmdb")
log_loss_show = 200

cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.NASH_M
set_march(march)
norm_cfg = None
qat_mode = "fuse_bn"
convert_mode = "fx"
enable_vpu = True

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]
max_num_points = 20
max_voxels = (30000, 40000)

class_names = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

tasks = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=2, class_names=["truck", "construction_vehicle"]),
    dict(num_class=2, class_names=["bus", "trailer"]),
    dict(num_class=1, class_names=["barrier"]),
    dict(num_class=2, class_names=["motorcycle", "bicycle"]),
    dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
]

common_heads = dict(
    reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)
)
with_velocity = "vel" in common_heads

float_epochs = 20
qat_epochs = 10
calibration_step = 100
val_interval = 30
