"""Common settings for sparse MapTR HENet-tiny NuScenes."""

import os

import torch
from hat_engine.utils.config import ConfigVersion
from horizon_plugin_pytorch.march import March

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "maptr_sparse_henet_tinym"
batch_size_per_gpu = 4
dataloader_workers = 4
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
num_machines = 2
log_freq = 200
ckpt_dir = "./tmp_models/%s" % task_name

local_train = not os.path.exists("/running_package")
if local_train:
    device_ids = [0]
    log_freq = 1
    batch_size_per_gpu = 4

float_lr = 3e-4
float_backbone_lr = 1e-3
qat_lr = 1e-5
num_epochs = 100
num_steps_per_epoch = int(
    28130 // (len(device_ids) * batch_size_per_gpu * num_machines)
)
num_steps = num_steps_per_epoch * num_epochs
enable_amp = True
qat_enable_amp = False
enable_amp_dtype = torch.float16
convert_mode = "jit-strip"
qat_mode = "with_bn"
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.NASH_M

data_rootdir = "./tmp_data/heal_data/nuscenes/v1.0-trainval/"
meta_rootdir = "./tmp_data/heal_data/nuscenes/meta"
anchor_file = os.path.join(ckpt_dir, "kmeans_map_100.npy")

map_classes = ["divider", "ped_crossing", "boundary"]
fixed_ptsnum_per_gt_line = 20
fixed_ptsnum_per_pred_line = 20
num_anchor = 100
eval_use_same_gt_sample_num_flag = True
num_map_classes = len(map_classes)

aux_seg_cfg = dict(
    use_aux_seg=True,
    bev_seg=False,
    pv_seg=True,
    dense_depth=True,
    seg_classes=1,
    feat_down_sample=[(1, 8), (2, 16), (3, 32)],
    pv_thickness=1,
)

_dim_ = 256
num_groups = 8
drop_out = 0.1
num_single_frame_decoder = 1
num_decoder = 6
feat_indices = [2]
num_levels = len(feat_indices)
num_depth_layers = 3

# Lidar LMDB for dense depth GT (packed via nuscenes_lidar_packer.py).
lidar_train_lmdb = os.path.join(data_rootdir, "train_lidar_lmdb")
lidar_val_lmdb = os.path.join(data_rootdir, "val_lidar_lmdb")
# Precomputed map vector GT (see tools/datasets/nuscenes_map_packer.py).
map_train_lmdb = os.path.join(data_rootdir, "train_map_lmdb")
map_val_lmdb = os.path.join(data_rootdir, "val_map_lmdb")
num_lidar_sweeps = 0
lidar_load_dim = 5
lidar_use_dim = [0, 1, 2, 4]

use_lidar_gt = True
use_lidar2img = use_lidar_gt
if use_lidar_gt:
    point_cloud_range = [-15.0, -30.0, -10.0, 15.0, 30.0, 10.0]
    bev_h_ = 100
    bev_w_ = 50
    post_center_range = [-20, -35, -20, -35, 20, 35, 20, 35]
else:
    point_cloud_range = [-30.0, -15.0, -10.0, 30.0, 15.0, 10.0]
    bev_h_ = 50
    bev_w_ = 100
    post_center_range = [-35, -20, -35, -20, 35, 20, 35, 20]

pretrained_model_dir = "./tmp_pretrained_models/henet_tinym/float-checkpoint-best.pth.tar"  # noqa: E501

resize_shape = (3, 396, 704)
val_data_shape = (3, 256, 704)
val_interval = 20
calibration_step = 10

deploy_inputs = {
    "img": torch.randn((6, 3, 256, 704)),
    "projection_mat": torch.randn((6, 4, 4)),
}
