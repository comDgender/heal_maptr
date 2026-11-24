import os

from hat_engine.utils.config import ConfigVersion
from horizon_plugin_pytorch.march import March

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "motr_efficientnetb3"
ckpt_dir = "./tmp_models/%s" % task_name

num_classes = 1

train_batch_size_per_gpu = 1
test_batch_size_per_gpu = 1
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]

train_lmdb = "./tmp_data/heal_data/mot17/train_lmdb"
val_lmdb = "./tmp_data/heal_data/mot17/test_lmdb"
val_gt = "./tmp_data/heal_data/mot17/test_gt"

cudnn_benchmark = False
seed = None
log_rank_zero_only = True
bn_kwargs = {}
march = March.NASH_M
qat_mode = "fuse_bn"
convert_mode = "jit-strip"

img_shape = (800, 1422)
num_queries = 256
