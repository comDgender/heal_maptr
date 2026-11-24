import math
import os

import numpy as np
from hat_engine.utils.config import ConfigVersion
from horizon_plugin_pytorch.march import March

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "detr3d_efficientnetb3"
num_classes = 10
batch_size_per_gpu = 1
dataloader_workers = 1
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = False
seed = None
log_rank_zero_only = True
bn_kwargs = {}
march = March.NASH_M
convert_mode = "jit-strip"
qat_mode = "fuse_bn"

num_query = 900
num_levels = 4
query_align = 256

orig_shape = (3, 900, 1600)
resize_shape = (3, 792, 1408)
data_shape = (3, 512, 1408)
val_data_shape = (3, 512, 1408)
vt_input_hw = [(32, 88), (16, 44), (8, 22), (4, 11)]

bev_range = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)

data_rootdir = "./tmp_data/heal_data/nuscenes/v1.0-trainval/"
meta_rootdir = "./tmp_data/heal_data/nuscenes/meta"


def get_grid_quant_scale(grid_shape, view_shape):
    max_coord = max(*grid_shape, *view_shape)
    coord_bit_num = math.ceil(math.log(max_coord + 1, 2))
    coord_shift = 15 - coord_bit_num
    coord_shift = max(min(coord_shift, 8), 0)
    grid_quant_scale = 1.0 / (1 << coord_shift)
    return grid_quant_scale


query_h = int(np.ceil(num_query / query_align))
query_w = query_align

grid_shape = (query_h, query_w)

view_scale = [8, 16, 32, 64]

grid_quant_scales = []
for scale in view_scale:
    view_shape = [data_shape[1] / scale, data_shape[2] / scale]
    grid_quant_scales.append(get_grid_quant_scale(grid_shape, view_shape))
