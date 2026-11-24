import copy

import torch
from common import (
    test_batch_size_per_gpu,
    train_batch_size_per_gpu,
    train_lmdb,
    val_lmdb,
)
from torchvision.transforms import Compose

from heal.data.collates import collate_mot_seq

data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Mot17Dataset",
        data_path=train_lmdb,
        sampler_lengths=[2, 3, 4, 5],
        sample_mode="random_interval",
        sample_interval=10,
        sampler_steps=[100, 180, 240],
        transforms=[
            dict(
                type="SeqRandomFlip",
                px=0.5,
                py=0,
            ),
            dict(
                type="RandomSelectOne",
                transforms=[
                    dict(
                        type="SeqResize",
                        img_scale=[
                            (608, 1536),
                            (640, 1536),
                            (672, 1536),
                            (704, 1536),
                            (736, 1536),
                            (768, 1536),
                            (800, 1536),
                            (832, 1536),
                            (864, 1536),
                            (896, 1536),
                            (928, 1536),
                            (960, 1536),
                            (992, 1536),
                        ],
                        multiscale_mode="value",
                        keep_ratio=True,
                        rm_neg_coords=False,
                        divisor=2,
                    ),
                    dict(
                        type=Compose,
                        transforms=[
                            dict(
                                type="SeqResize",
                                img_scale=[
                                    (400, 9999999),
                                    (500, 9999999),
                                    (600, 9999999),
                                ],
                                multiscale_mode="value",
                                keep_ratio=True,
                                rm_neg_coords=False,
                                divisor=2,
                            ),
                            dict(
                                type="SeqRandomSizeCrop",
                                min_size=384,
                                max_size=600,
                                filter_area=False,
                                rm_neg_coords=False,
                            ),
                            dict(
                                type="SeqResize",
                                img_scale=[
                                    (608, 1536),
                                    (640, 1536),
                                    (672, 1536),
                                    (704, 1536),
                                    (736, 1536),
                                    (768, 1536),
                                    (800, 1536),
                                    (832, 1536),
                                    (864, 1536),
                                    (896, 1536),
                                    (928, 1536),
                                    (960, 1536),
                                    (992, 1536),
                                ],
                                multiscale_mode="value",
                                keep_ratio=True,
                                rm_neg_coords=False,
                                divisor=2,
                            ),
                        ],
                    ),
                ],
                p=1,
            ),
            dict(
                type="SeqToTensor",
                to_yuv=False,
            ),
            dict(type="SeqBgrToYuv444", rgb_input=True),
            dict(
                type="SeqNormalize",
                mean=128.0,
                std=128.0,
            ),
        ],
    ),
    sampler=dict(type="DistSetEpochDatasetSampler"),
    batch_size=train_batch_size_per_gpu,
    pin_memory=True,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_mot_seq,
)

val_data_loader = dict(
    type=torch.utils.data.DataLoader,
    dataset=dict(
        type="Mot17Dataset",
        data_path=val_lmdb,
        sampler_lengths=[1],
        sample_mode="fixed_interval",
        sample_interval=10,
        transforms=[
            dict(
                type="SeqResize",
                img_scale=(800, 1422),
                keep_ratio=False,
            ),
            dict(
                type="SeqToTensor",
                to_yuv=False,
            ),
            dict(type="SeqBgrToYuv444", rgb_input=True),
            dict(
                type="SeqNormalize",
                mean=128.0,
                std=128.0,
            ),
        ],
    ),
    batch_size=test_batch_size_per_gpu,
    pin_memory=True,
    shuffle=False,
    num_workers=2,
    collate_fn=collate_mot_seq,
)


def loss_collector(outputs: dict):
    losses = []
    for _, loss in outputs.items():
        mean_loss = sum(loss) / len(loss)
        losses.append(mean_loss)
    return losses


def update_loss(metrics, batch, model_outs):
    mean_losses = {}
    for loss_name, loss in model_outs.items():
        mean_loss = sum(loss) / len(loss)
        mean_losses[loss_name] = mean_loss
    for metric in metrics:
        metric.update(mean_losses)


batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=True,
    loss_collector=loss_collector,
)
val_batch_processor = dict(
    type="MultiBatchProcessor",
    need_grad_update=False,
)

calibration_data_loader = copy.deepcopy(data_loader)
calibration_data_loader["dataset"]["sampler_steps"] = [3, 5, 7]
calibration_data_loader.pop("sampler")  # Calibration do not support DDP or DP
calibration_data_loader["batch_size"] = train_batch_size_per_gpu * 2
calibration_data_loader["dataset"]["transforms"] = val_data_loader["dataset"][
    "transforms"
]

qat_train_dataloader = copy.deepcopy(data_loader)
qat_train_dataloader["dataset"]["sampler_steps"] = [3, 5, 7]

int_infer_data_loader = copy.deepcopy(val_data_loader)
int_infer_data_loader["sampler"] = None
int_infer_data_loader["batch_size"] = 1
int_infer_data_loader["shuffle"] = False

inference_dataset = dict(
    type="Mot17Dataset",
    data_path=val_lmdb,
    sampler_lengths=[2],
    sample_mode="fixed_interval",
    sample_interval=1,
    transforms=None,
)

hbir_data_loader = int_infer_data_loader

# HBM path: nv12_runtime expects uint8 RGB; strip BgrToYuv444 + Normalize.
hbm_data_loader = copy.deepcopy(int_infer_data_loader)
hbm_data_loader["dataset"]["transforms"] = [
    dict(
        type="SeqResize",
        img_scale=(800, 1422),
        keep_ratio=False,
    ),
    dict(
        type="SeqToTensor",
        to_yuv=False,
    ),
]
hbm_batch_processor = val_batch_processor


def get_example_input():
    """One real val batch for jit-strip trace.

    Motr.forward (non-compile) expects data['frame_data_list'], so use
    a real sequence batch from val_data_loader (collate_mot_seq output),
    not randn deploy_inputs which is for compile_motr deploy_model.
    """
    import pickle

    from hat_engine.registry import build_from_registry

    _ds = build_from_registry(val_data_loader["dataset"])
    _ds = pickle.loads(pickle.dumps(_ds))
    return collate_mot_seq([_ds[0]])


example_inputs = get_example_input()
