"""Common settings for BevFormer tiny ResNet50 NuScenes detection."""

import os

import torch
from hat_engine.utils.config import ConfigVersion
from horizon_plugin_pytorch.march import March

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

enable_model_tracking = True

task_name = "bevformer_tiny_resnet50_detection"
batch_size_per_gpu = 2
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
lossshow = 50
convert_mode = "jit-strip"
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = True
seed = None
log_rank_zero_only = True
march = March.NASH_M
data_rootdir = "./tmp_data/heal_data/nuscenes/v1.0-trainval/"
meta_rootdir = "./tmp_data/heal_data/nuscenes/meta"
map_size = (15, 30, 0.15)
CLASSES = (
    "car",
    "truck",
    "trailer",
    "bus",
    "construction_vehicle",
    "bicycle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "barrier",
)
bn_kwargs = {}
_dim_ = 256
_pos_dim_ = _dim_ // 2
_num_levels_ = 1
bev_h_ = 50
bev_w_ = 50
queue_length = 3  # each sequence contains `queue_length` frames.
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
bev_size = (51.2, 51.2, 2.048)
use_bce = False
num_classes = 10

resize_shape = (3, 450, 800)
val_data_shape = (3, 480, 800)
orig_shape = (3, 900, 1600)

dataloader_workers = 2
float_lr = 4e-4
float_epochs = 24
qat_lr = 1e-5
qat_epochs = 10
calibration_step = 100
val_interval = 4
ckpt_save_interval = 4


def reset_prev_frame_info(model):
    pose_key = model.view_transformer.global_pose_key
    model.view_transformer.prev_frame_info = {
        "prev_bev": None,
        "scene_token": None,
        pose_key: None,
    }
    return model


deploy_inputs = dict(
    img=torch.randn((6, 3, 480, 800)),
    prev_bev=torch.randn((1, 2500, 256)),
    prev_bev_ref=torch.randn((1, 50, 50, 2)),
    queries_rebatch_grid=torch.randn((6, 20, 32, 2)),
    restore_bev_grid=torch.randn((1, 100, 50, 2)),
    reference_points_rebatch=torch.randn((6, 640, 4, 2)),
    bev_pillar_counts=torch.randn((1, 2500, 1)),
)
