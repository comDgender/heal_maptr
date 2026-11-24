"""Visualize packed ImageNet classification samples."""

import argparse
import logging
import pickle

import numpy as np
import torch
from hat_engine.data.datasets.imagenet import ImageNet
from hat_engine.utils.logger import init_logger

from heal.data.datasets.imagenet.meta import Classes
from heal.visualize.cls import ClsViz

logger = logging.getLogger(__name__)

NUM_CLASSES = 1000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize packed ImageNet dataset."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Packed data path (e.g. tmp_data/heal_data/imagenet/val_lmdb/).",
    )
    parser.add_argument(
        "--viz-num",
        type=int,
        default=1,
        help="Max number of samples to visualize.",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Directory to save visualization PNGs.",
    )
    return parser.parse_args()


def _image_to_hwc(image):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    return image


def _label_to_logits(label):
    label = int(np.asarray(label).reshape(-1)[0])
    logits = torch.zeros(NUM_CLASSES)
    logits[label] = 1.0
    return logits


if __name__ == "__main__":
    args = parse_args()

    init_logger(".hat_logs/imagenet_viz")

    dataset = ImageNet(data_path=args.data_path)
    dataset = pickle.loads(pickle.dumps(dataset))
    viz = ClsViz(save_path=args.save_path)

    logger.info(
        "Visualizing up to %d samples from %s", args.viz_num, args.data_path
    )
    for i, data in enumerate(dataset):
        if i >= args.viz_num:
            break
        img = _image_to_hwc(data["img"])
        preds = _label_to_logits(data["labels"])
        viz(preds, {"img": img, "classes": Classes})
