"""Common settings for BevFormer HENet camera-only det+occ multitask pretrain."""  # noqa: E501

import os

from hat_engine.utils.config import ConfigVersion
from horizon_plugin_pytorch.march import March

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "bevformer_henet_camera_multitask"
pretrain_ckpt_dir = "./tmp_pretrained_models/henet_tinym"

batch_size_per_gpu = 1
dataloader_workers = 4
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]

ckpt_dir = f"./tmp_models/{task_name}"

data_rootdir = "./tmp_data/heal_data/nuscenes/v1.0-trainval/"
meta_rootdir = "./tmp_data/heal_data/nuscenes/meta"
occ_train_lmdb = os.path.join(data_rootdir, "train_occ_lmdb")
occ_val_lmdb = os.path.join(data_rootdir, "val_occ_lmdb")

log_loss_show = 200
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.NASH_M
norm_cfg = None
enable_amp = True

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
bev_size = (51.2, 51.2, 0.8)
bev_size_occ = (40, 40)

_dim_ = 256
_pos_dim_ = _dim_ // 2
_num_levels_ = 1
bev_h_ = 128
bev_w_ = 128
input_size = (512, 960)
map_size = (15, 30, 0.15)
use_lidar2img = True
max_camoverlap_num = 2

lidar_input = False
camera_input = True
use_occ_head = True
use_bev_head = True

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
val_interval = 1
float_lr = 4e-4
