import os

from hat_engine.utils.config import ConfigVersion
from horizon_plugin_pytorch.march import March

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "deform_detr_resnet50"
num_classes = 80
batch_size_per_gpu = 2
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = False
seed = 1
log_rank_zero_only = True
bn_kwargs = {}
march = March.NASH_M
qat_mode = "fuse_bn"
img_shape = (800, 1332)
convert_mode = "jit-strip"
num_queries = 900
with_box_refine = False
as_two_stage = False

data_path = "./tmp_data/heal_data/mscoco/"
ann_file = "./tmp_data/heal_data/mscoco/instances_val2017.json"
