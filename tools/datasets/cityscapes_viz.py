"""Visualize packed Cityscapes segmentation samples."""

import argparse
import logging
import os
import pickle

import matplotlib

if (
    os.environ.get("DISPLAY", "") == ""
    and os.environ.get("MPLBACKEND") is None
):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from hat_engine.data.transforms.common import PILToTensor
from hat_engine.utils.logger import init_logger

from heal.data.datasets.cityscapes import CITYSCAPES_LABLE_MAPPINGS, Cityscapes
from heal.data.transforms.segmentation import LabelRemap
from heal.visualize.seg import _cityscapes_colormap, semantic_segmentation
from heal.visualize.utils import plot_image

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize packed Cityscapes dataset."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Packed data path (e.g. tmp_data/cityscapes/train_lmdb).",
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


def _seg_to_numpy(gt_seg):
    if isinstance(gt_seg, torch.Tensor):
        gt_seg = gt_seg.detach().cpu().numpy()
    else:
        gt_seg = np.asarray(gt_seg)
    if gt_seg.ndim == 3:
        if gt_seg.shape[0] == 1:
            gt_seg = gt_seg[0]
        else:
            gt_seg = np.argmax(gt_seg, axis=0)
    gt_seg = gt_seg.squeeze().astype(np.int64)
    # uint8 LabelRemap encodes void (-1) as 255; use black in colormap.
    gt_seg = np.where(gt_seg == 255, 19, gt_seg)
    return gt_seg


def _show_sample(image, gt_seg, save_path, sample_idx):
    colors = _cityscapes_colormap.detach().cpu().numpy().tolist()
    blended = semantic_segmentation(gt_seg, colors, image=image, alpha=0.5)

    fig = plt.figure(figsize=(12, 6), facecolor="white")
    ax1 = fig.add_subplot(1, 2, 1)
    plot_image(image, ax=ax1)
    ax1.set_title("input")
    ax2 = fig.add_subplot(1, 2, 2)
    plot_image(blended, ax=ax2)
    ax2.set_title("gt_seg")

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        result_path = os.path.join(
            save_path, "cityscapes_gt_%d.png" % sample_idx
        )
        fig.savefig(
            result_path,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            pad_inches=0,
        )
        logger.info("Saved visualization to %s", result_path)
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()

    init_logger(".hat_logs/cityscapes_viz")

    dataset = Cityscapes(
        data_path=args.data_path,
        transforms=torchvision.transforms.Compose(
            [
                PILToTensor(),
                LabelRemap(mapping=CITYSCAPES_LABLE_MAPPINGS),
            ]
        ),
    )
    dataset = pickle.loads(pickle.dumps(dataset))

    logger.info(
        "Visualizing up to %d samples from %s", args.viz_num, args.data_path
    )
    for i, data in enumerate(dataset):
        if i >= args.viz_num:
            break
        image = _image_to_hwc(data["ori_img"])
        gt_seg = _seg_to_numpy(data["gt_seg"])
        _show_sample(
            image,
            gt_seg,
            save_path=args.save_path,
            sample_idx=i,
        )
