import os

import numpy as np
from hat_engine.utils.config import ConfigVersion
from horizon_plugin_pytorch.march import March

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "unet_mobilenetv1"
num_classes = 19
batch_size_per_gpu = 4
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
dataloader_workers = batch_size_per_gpu
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.NASH_M
enable_amp = False

convert_mode = "jit-strip"

data_shape = (3, 1024, 2048)
train_scales = (4, 8, 16, 32, 64)
alpha = 0.25
bn_kwargs = dict(eps=2e-5, momentum=0.1)
output_with_bn = True

weight_decay = 5e-3
start_lr = 0.01
train_epochs = 600
qat_lr = 8e-5
qat_train_epochs = 30

data_path = "./tmp_data/heal_data/cityscapes/"
tensorboard_log_path = os.path.join(ckpt_dir, "tensorboard")

loss_weights = tuple(np.array((256, 128, 64, 32, 16)) / 19)
scale_factors = tuple(1 / np.array(train_scales))
