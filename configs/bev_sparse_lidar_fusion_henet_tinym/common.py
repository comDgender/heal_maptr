"""Common settings for SparseBEV lidar fusion HENet-tiny NuScenes detection."""

import os

from hat_engine.utils.config import ConfigVersion
from horizon_plugin_pytorch.march import March

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "bev_sparse_lidar_fusion_henet_tinym"
pretrain_ckpt_dir = "./tmp_models/bev_sparse_henet_tinym"
num_classes = 10
batch_size_per_gpu = 4
dataloader_workers = 4
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = False
seed = None
log_rank_zero_only = True
march = March.NASH_M
convert_mode = "jit-strip"
qat_mode = "fuse_bn"
enable_amp = True

orig_shape = (3, 900, 1600)
resize_shape = (3, 396, 704)
data_shape = (3, 256, 704)
val_data_shape = (3, 256, 704)

bev_range = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
point_cloud_range = list(bev_range)
position_range = (-61.2, -61.2, -10.0, 61.2, 61.2, 10.0)

data_rootdir = "./tmp_data/heal_data/nuscenes/v1.0-trainval/"
meta_rootdir = "./tmp_data/heal_data/nuscenes/meta"
anchor_file = os.path.join(ckpt_dir, "nuscenes_kmeans900.npy")

num_epochs = 50
num_steps_per_epoch = int(28130 // (len(device_ids) * batch_size_per_gpu))
num_steps = num_steps_per_epoch * num_epochs
num_anchors = 384
temp_anchors = 128
embed_dims = 256
num_groups = 8
num_levels = 1
drop_out = 0.1
num_single_frame_decoder = 0
num_decoder = 6
num_depth_layers = 3
dense_depth_loss_weight = 0.2

max_num_points = 20
voxel_size = [0.2, 0.2, 8]
max_voxels = (30000, 40000)
lidar_level_idx = [1, 2]

lidar_train_lmdb = os.path.join(data_rootdir, "train_lidar_lmdb")
lidar_val_lmdb = os.path.join(data_rootdir, "val_lidar_lmdb")
num_lidar_sweeps = 9
lidar_load_dim = 5
lidar_use_dim = [0, 1, 2, 3, 4]

float_lr = 3e-4
qat_lr = 5e-6
calibration_step = 100
lossshow = 50

CLASSES = (
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
