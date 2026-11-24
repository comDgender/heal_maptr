"""Common settings for MapTR HENet-tiny BEVFormer NuScenes."""

import os

import torch
from hat_engine.utils.config import ConfigVersion
from horizon_plugin_pytorch.march import March

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

enable_model_tracking = False
task_name = "maptr_henet_tinym_bevformer"
batch_size_per_gpu = 2
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
log_freq = 200
ckpt_dir = "./tmp_models/%s" % task_name

float_lr = 2.5e-4
float_epochs = 30
enable_amp = False
enable_amp_dtype = None
convert_mode = "jit-strip"
qat_mode = "with_bn"
calibration_step = 100

cudnn_benchmark = True
seed = 42
log_rank_zero_only = True
march = March.NASH_M

data_rootdir = "./tmp_data/heal_data/nuscenes/v1.0-trainval/"
meta_rootdir = "./tmp_data/heal_data/nuscenes/meta"
sd_map_path = "./tmp_data/heal_data/nuscenes/osm"
# Precomputed map vector GT (see tools/datasets/nuscenes_map_packer.py).
map_train_lmdb = os.path.join(data_rootdir, "train_map_lmdb")
map_val_lmdb = os.path.join(data_rootdir, "val_map_lmdb")

map_classes = ["divider", "ped_crossing", "boundary"]
aux_seg_cfg = dict(
    use_aux_seg=True,
    bev_seg=True,
    pv_seg=True,
    seg_classes=1,
    feat_down_sample=[(-1, 32)],
    pv_thickness=1,
)

_num_levels_ = 1
hidden_dim = 64
bev_embed_dims = 256
bev_sparse_rate = 0.4
max_camoverlap_num = 2
queue_length = 1
test_queue_length = 1
single_bev = True

head_embed_dims = 512
num_vec_one2one = 50
num_vec_one2many = 300
k_one2many = 6
fixed_ptsnum_per_gt_line = 20
fixed_ptsnum_per_pred_line = 20
eval_use_same_gt_sample_num_flag = True
num_map_classes = len(map_classes)

use_lidar_gt = False
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

resize_shape = (3, 450, 800)
val_data_shape = (3, 480, 800)
orig_shape = (3, 900, 1600)
val_interval = 30
ckpt_save_interval = 1


deploy_inputs = dict(
    img=torch.randn((6, 3, 480, 800)),
    osm_mask=torch.randn((1, 1, bev_h_, bev_w_)),
    queries_rebatch_grid=torch.randn(
        (6, int(bev_sparse_rate * bev_h_), bev_w_, 2)
    ),
    restore_bev_grid=torch.randn((1, max_camoverlap_num * bev_h_, bev_w_, 2)),
    reference_points_rebatch=torch.randn(
        (6, int(bev_sparse_rate * bev_h_ * bev_w_), 4, 2)
    ),
    bev_pillar_counts=torch.randn((1, bev_h_ * bev_w_, 1)),
)
