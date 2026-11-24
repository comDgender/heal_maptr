"""Predictor definitions for BevFormer HENet camera-only det+occ pretrain."""

import os

from callback import val_metric_updater, val_metrics
from common import ckpt_dir
from data import val_batch_processor, val_data_loader
from model import model

float_predictor = dict(
    type="Predictor",
    model=model,
    model_convert_pipeline=dict(
        type="ModelConvertPipeline",
        converters=[
            dict(
                type="LoadCheckpoint",
                checkpoint_path=os.path.join(
                    ckpt_dir, "float-checkpoint-last.pth.tar"
                ),
                allow_miss=True,
                ignore_extra=True,
                verbose=True,
            ),
        ],
    ),
    data_loader=[val_data_loader],
    batch_processor=val_batch_processor,
    device=None,
    metrics=val_metrics,
    callbacks=[val_metric_updater],
    log_interval=100,
)
