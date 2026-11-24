import os

from hat_engine.utils.config import ConfigVersion
from horizon_plugin_pytorch.march import March, set_march

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "pointpillars"
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]

ckpt_dir = f"./tmp_models/{task_name}"
base_data_dir = "./tmp_data/heal_data/kitti3d/"

cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.NASH_M
set_march(march)
enable_vpu = True
class_names = ["Car"]
