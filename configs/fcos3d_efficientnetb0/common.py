import os

from hat_engine.utils.config import ConfigVersion
from horizon_plugin_pytorch.march import March

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "fcos3d_efficientnetb0"
ckpt_dir = "./tmp_models/%s" % task_name
meta_rootdir = "./tmp_data/heal_data/nuscenes/meta"

num_classes = 10
batch_size_per_gpu = 8
device_ids = [0, 1, 2, 3]
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
bn_kwargs = {}
march = March.NASH_M
