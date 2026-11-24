"""Common settings for sparse multitask det + MapTR + FlashOcc."""

import math
import os

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
from hat_engine.utils.config import ConfigVersion  # noqa: E402
from horizon_plugin_pytorch.march import March  # noqa: E402

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float_stage1")

task_name = "sparse_multitask_det_maptr_flashocc_henet_tinym"
batch_size_per_gpu = 4
dataloader_workers = 8
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
num_machines = 2
ckpt_dir = "./tmp_models/%s" % task_name
pretrain_ckpt_dir = "./tmp_pretrained_models/henet_tinym"

local_train = not os.path.exists("/running_package")
if local_train:
    device_ids = [0]
    batch_size_per_gpu = 4

cudnn_benchmark = False
seed = None
log_rank_zero_only = True
march = March.NASH_M
convert_mode = "jit-strip"
qat_mode = "fuse_bn"
enable_amp = False

orig_shape = (3, 900, 1600)
resize_shape = (3, 396, 704)
data_shape = (3, 256, 704)
val_data_shape = data_shape
scale = float(resize_shape[2]) / float(orig_shape[2])
resize_aug = (-0.06, 0.11)

data_rootdir = "./tmp_data/heal_data/nuscenes/v1.0-trainval/"
meta_rootdir = "./tmp_data/heal_data/nuscenes/meta"
det_anchor_file = os.path.join(ckpt_dir, "nuscenes_kmeans900.npy")
map_anchor_file = os.path.join(ckpt_dir, "kmeans_map_100.npy")

# Stage1: joint det + map + occ; stage2 trains map/occ only (backbone frozen,
# det absent) and is merged with stage1 afterward (det comes from stage1).
num_epochs = 200
num_steps_per_epoch = int(
    28130 // (len(device_ids) * batch_size_per_gpu * num_machines)
)
num_steps = num_steps_per_epoch * num_epochs
float_lr = 3e-4
float_stage2_lr = 1e-5
calibration_step = 50
calibration_num_steps = 100
val_interval = num_steps_per_epoch * 10
log_freq = 200 if not local_train else 1

embed_dims = 256
num_groups = 8
num_levels = 1
num_classes = 10
drop_out = 0.1
num_single_frame_decoder = 0
num_decoder = 6
num_depth_layers = 3
num_anchors = 384
temp_anchors = 128

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
map_point_cloud_range = [-15.0, -30.0, -10.0, 15.0, 30.0, 10.0]
map_post_center_range = [-20, -35, -20, -35, 20, 35, 20, 35]
bev_h_ = 100
bev_w_ = 50

map_classes = ["divider", "ped_crossing", "boundary"]
fixed_ptsnum_per_gt_line = 20
fixed_ptsnum_per_pred_line = 20
om_num_anchor = 100
num_map_classes = len(map_classes)
eval_use_same_gt_sample_num_flag = True

map_aux_seg_cfg = dict(
    use_aux_seg=True,
    bev_seg=False,
    pv_seg=True,
    dense_depth=False,
    seg_classes=1,
    feat_down_sample=[(1, 8), (2, 16), (3, 32)],
    pv_thickness=1,
    defer_pv_seg_rasterize=True,
)

enable_det_head = True
enable_om_head = True
enable_occ_head = True

# FlashOcc LSS settings (256x704 input).
occ_depth = 45
occ_grid_size = (128, 128)
view_shape = [data_shape[1] / 32, data_shape[2] / 32]
vt_input_hw = [int(view_shape[0]), int(view_shape[1])]
depthview_shape = [6 * occ_depth, view_shape[0] * view_shape[1]]
featview_shape = [view_shape[0] * 6, view_shape[1]]


def get_grid_quant_scale(grid_shape, view_shape):
    max_coord = max(*grid_shape, *view_shape)
    coord_bit_num = math.ceil(math.log(max_coord + 1, 2))
    coord_shift = 15 - coord_bit_num
    coord_shift = max(min(coord_shift, 8), 0)
    return 1.0 / (1 << coord_shift)


grid_quant_scale = get_grid_quant_scale(occ_grid_size, featview_shape)
depth_grid_quant_scale = get_grid_quant_scale(occ_grid_size, depthview_shape)
num_classes_occ = 18
occ_num_points = 10
occ_bev_size = (40, 40, 0.625)

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

lidar_train_lmdb = os.path.join(data_rootdir, "train_lidar_lmdb")
lidar_val_lmdb = os.path.join(data_rootdir, "val_lidar_lmdb")
map_train_lmdb = os.path.join(data_rootdir, "train_map_lmdb")
map_val_lmdb = os.path.join(data_rootdir, "val_map_lmdb")
occ_train_lmdb = os.path.join(data_rootdir, "train_occ_lmdb")
occ_val_lmdb = os.path.join(data_rootdir, "val_occ_lmdb")

num_lidar_sweeps = 9
lidar_load_dim = 5
lidar_use_dim = [0, 1, 2, 3, 4]
use_lidar_gt = True

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

loss_weights = dict(det=2.5, om=0.9, occ=9.0)


def get_deploy_input():
    return {
        "img": torch.randn((6, 3, 256, 704)),
        "projection_mat": torch.randn((6, 4, 4)),
        "cached_anchor": torch.randn((1, num_anchors, 11)),
        "cached_feature": torch.randn((1, num_anchors, embed_dims)),
        "points0": torch.randn((occ_num_points, 128, 128, 2)),
        "points1": torch.randn((occ_num_points, 128, 128, 2)),
    }


deploy_inputs = get_deploy_input()
