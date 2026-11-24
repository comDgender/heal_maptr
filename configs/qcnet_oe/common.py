import os

from hat_engine.utils.config import ConfigVersion
from horizon_plugin_pytorch.march import March, set_march

VERSION = ConfigVersion.v2
training_step = os.environ.get("HAT_TRAINING_STEP", "float")

task_name = "qcnet_oe"

batch_size_per_gpu = 4 * 4
dataloader_workers = 4
device_ids = [i for i in range(8)]  # noqa
ckpt_dir = "./tmp_models/%s" % task_name
cudnn_benchmark = False
seed = 2
log_rank_zero_only = True
bn_kwargs = {}
march = March.NASH_M
# march = March.NASH_P
set_march(march)
qat_mode = "fuse_bn"

data_rootdir = "./tmp_data/heal_data/argoverse2/"
input_dim = 2
hidden_dim = 128
ori_historical_sec = (
    5  # Length of the original historical trajectory in seconds）
)
ori_future_sec = 6  # Length of the predicted future trajectory in seconds
ori_sample_fre = (
    10  # Original sampling frequency, which is 10 steps per second
)
sample_fre = 2  # Frequency used for downsampling
num_historical_steps = ori_historical_sec * sample_fre
num_future_steps = ori_future_sec * sample_fre
num_freq_bands = 32
num_heads = 8
head_dim = 16
output_dim = 2
time_span = 2
dropout = 0.1
num_map_layers = 1
num_agent_layers = 1
num_modes = 6
num_recurrent_steps = 1
num_t2m_steps = 30 // ori_sample_fre * sample_fre  # times steps for decode

num_pl2a = 32  # pl2a gather
num_a2a = 36  # a2a gather

num_dec_layers = 1
split_rec_modules = True
reuse_agent_rembs = True  # Flag to reuse agent rembs for the decoder.

# quant_infer_cold_start: Indicates the compilation mode for inference.
# True for cold start streaming inference (requires one model compilation).
# False for hot start streaming inference (requires two model compilations).
# Applicable only for prediction or integer inference.
quant_infer_cold_start = True  # Set to False when training

# Shapes used to build deploy/streaming-inference example inputs.
B = 1
A = 30
pl = 80
pt = 50
HT = ori_historical_sec * sample_fre  # if sample_fre==2 init HT=10
