"""Common configuration for FlashOcc HENet LSS Occ3D NuScenes."""

import math
import os

from hat_engine.utils.config import ConfigVersion
from horizon_plugin_pytorch.march import March

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "flashocc_henet_lss_occ3d"
num_classes = 18
batch_size_per_gpu = 4
val_batch_size_per_gpu = 1
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.NASH_M
qat_mode = "fuse_bn"
convert_mode = "jit-strip"
enable_model_tracking = True

base_dataset_type = "NuscenesBevDataset"
occ_dataset_type = "NuscenesOccDataset"
data_rootdir = "./tmp_data/heal_data/nuscenes/v1.0-trainval/"

train_interval = 1
val_interval = 1
val_log_interval = 4
num_epochs = 150
step_log_freq = 100
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]

bn_kwargs = dict(eps=2e-5, momentum=0.1)
depth = 45
num_points = 10
bev_size = (40, 40, 0.625)
grid_size = (128, 128)

orig_shape = (3, 900, 1600)
data_shape = (3, 512, 960)
val_data_shape = (3, 512, 960)
resize_shape = (3, 540, 960)

view_shape = [data_shape[1] / 32, data_shape[2] / 32]
vt_input_hw = [int(view_shape[0]), int(view_shape[1])]
depthview_shape = [6 * depth, view_shape[0] * view_shape[1]]
featview_shape = [view_shape[0] * 6, view_shape[1]]


def get_grid_quant_scale(grid_shape, view_shape):
    max_coord = max(*grid_shape, *view_shape)
    coord_bit_num = math.ceil(math.log(max_coord + 1, 2))
    coord_shift = 15 - coord_bit_num
    coord_shift = max(min(coord_shift, 8), 0)
    grid_quant_scale = 1.0 / (1 << coord_shift)
    return grid_quant_scale


grid_quant_scale = get_grid_quant_scale(grid_size, featview_shape)
depth_quant_scale = get_grid_quant_scale(grid_size, depthview_shape)

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
