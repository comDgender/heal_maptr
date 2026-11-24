import os

from hat_engine.utils.config import ConfigVersion
from horizon_plugin_pytorch.march import March

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "petr_efficientnetb3"
num_classes = 10
batch_size_per_gpu = 2
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
query_align = 128

orig_shape = (3, 900, 1600)
resize_shape = (3, 792, 1408)
data_shape = (3, 512, 1408)
val_data_shape = (3, 512, 1408)

bev_range = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
position_range = (-61.2, -61.2, -10.0, 61.2, 61.2, 10.0)

vt_input_hw = (16, 44)
data_rootdir = "./tmp_data/heal_data/nuscenes/v1.0-trainval/"
meta_rootdir = "./tmp_data/heal_data/nuscenes/meta"
tensorboard_log_path = os.path.join(ckpt_dir, "tensorboard")
