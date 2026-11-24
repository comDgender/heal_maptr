import os

from hat_engine.utils.config import ConfigVersion
from horizon_plugin_pytorch.march import March

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "ganet_mixvargenet"
ckpt_dir = "./tmp_models/%s" % task_name

data_num_workers = 4
batch_size_per_gpu = 64
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]

train_data_path = "./tmp_data/heal_data/CULane/train_lmdb"
val_data_path = "./tmp_data/heal_data/CULane/test_lmdb"

cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.NASH_M

base_lr = 0.01
num_epochs = 240
bn_kwargs = {}
radius = 2
hid_dim = 32
attn_ratio = 4
