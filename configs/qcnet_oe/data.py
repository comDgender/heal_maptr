import copy
import os
from functools import partial

import torch
from common import (
    batch_size_per_gpu,
    data_rootdir,
    dataloader_workers,
    ori_future_sec,
    ori_historical_sec,
    ori_sample_fre,
    sample_fre,
)

from heal.data.collates.qc_collate import (
    collate_qc_argoverse2,  # Reduce the data sampling frequency
)

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Argoverse2PackedDataset",
        data_path=os.path.join(data_rootdir, "train_lmdb"),
        split="train",
        pack_type="lmdb",
        input_dim=2,
        transforms=None,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=dataloader_workers,
    pin_memory=True,
    collate_fn=partial(
        collate_qc_argoverse2,
        ori_historical_sec=ori_historical_sec,
        ori_future_sec=ori_future_sec,
        ori_sample_fre=ori_sample_fre,
        sample_fre=sample_fre,
        stage="train",
        add_noise=False,
        agent_num=30,
        pl_N=80,
        pt_N=None,
        pt_N_downsample_nums=50,
    ),
)


qat_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Argoverse2PackedDataset",
        data_path=os.path.join(data_rootdir, "train_lmdb"),
        split="train",
        pack_type="lmdb",
        input_dim=2,
        transforms=None,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=True,
    num_workers=dataloader_workers,
    pin_memory=True,
    collate_fn=partial(
        collate_qc_argoverse2,
        ori_historical_sec=ori_historical_sec,
        ori_future_sec=ori_future_sec,
        ori_sample_fre=ori_sample_fre,
        sample_fre=sample_fre,
        stage="train",
        add_noise=False,
        agent_num=30,
        pl_N=80,
        pt_N=None,
        pt_N_downsample_nums=50,
    ),
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Argoverse2PackedDataset",
        data_path=os.path.join(data_rootdir, "val_lmdb"),
        split="val",
        pack_type="lmdb",
        input_dim=2,
        transforms=None,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=dataloader_workers,
    pin_memory=True,
    collate_fn=partial(
        collate_qc_argoverse2,
        ori_historical_sec=ori_historical_sec,
        ori_future_sec=ori_future_sec,
        ori_sample_fre=ori_sample_fre,
        sample_fre=sample_fre,
        stage="val",
        add_noise=False,
        agent_num=30,
        pl_N=80,
        pt_N=50,
    ),
)


def loss_collector(outputs: dict):
    losses = []
    for _, loss in outputs.items():
        losses.append(loss)
    return losses


batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    loss_collector=loss_collector,
)

val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
)


# Note: The transforms of the dataset during calibration can be
# consistent with that during training or validation, or customized.
# Default used `val_batch_processor`.
calibration_data_loader = copy.deepcopy(qat_data_loader)
calibration_data_loader.pop("sampler")
calibration_example_data_loader = copy.deepcopy(calibration_data_loader)
calibration_example_data_loader["num_workers"] = 0
calibration_data_loader["batch_size"] = 1
calibration_batch_processor = copy.deepcopy(val_batch_processor)
calibration_step = 100


int_infer_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Argoverse2PackedDataset",
        data_path=os.path.join(data_rootdir, "val_lmdb"),
        split="val",
        pack_type="lmdb",
        input_dim=2,
        transforms=None,
    ),
    sampler=dict(type=torch.utils.data.DistributedSampler),
    batch_size=1,
    shuffle=False,
    num_workers=dataloader_workers,
    pin_memory=True,
    collate_fn=partial(
        collate_qc_argoverse2,
        ori_historical_sec=ori_historical_sec,
        ori_future_sec=ori_future_sec,
        ori_sample_fre=ori_sample_fre,
        sample_fre=sample_fre,
        stage="val",
        agent_num=30,
        pl_N=80,
        pt_N=50,
    ),
)
