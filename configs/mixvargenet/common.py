import os

from hat_engine.utils.config import ConfigVersion
from horizon_plugin_pytorch.march import March

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "mixvargenet"
ckpt_dir = "./tmp_models/%s" % task_name

device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.NASH_M

num_classes = 1000
