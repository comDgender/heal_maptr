import os

from hat_engine.utils.config import ConfigVersion
from horizon_plugin_pytorch.march import March

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "bev_vtv2_resnet50"
batch_size_per_gpu = 2
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]

dataloader_workers = batch_size_per_gpu
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.NASH_M
convert_mode = "jit-strip"
enable_amp = False

orig_shape = (3, 900, 1600)
resize_shape = (3, 396, 704)
data_shape = (3, 256, 704)
val_data_shape = (3, 256, 704)
vt_input_hw = (16, 44)

bn_kwargs = dict(eps=2e-5, momentum=0.1)

weight_decay = 0.01
start_lr = 2e-4
train_epochs = 40
qat_lr = 2e-7
qat_train_epochs = 10

bev_size = (51.2, 51.2, 0.8)
grid_size = (128, 128)
map_size = (15, 30, 0.15)
task_map_size = (15, 30, 0.15)

data_rootdir = "./tmp_data/heal_data/nuscenes/v1.0-trainval/"
meta_rootdir = "./tmp_data/heal_data/nuscenes/meta"

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.1, 0.1, 0.2]

seg_classes_name = ["others", "divider", "ped_crossing", "Boundary"]
use_bce = False

if use_bce:
    seg_classes = 3
else:
    seg_classes = 3 + 1

depth = 60
num_points = 10
