"""Visualize packed MSCOCO detection samples."""

import argparse
import logging
import pickle

import torch
from hat_engine.utils.logger import init_logger

from heal.data.datasets.mscoco import COCO_CLASS_NAMES, Coco
from heal.visualize.det import DetViz

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize packed mscoco dataset."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Packed data path (e.g. tmp_data/mscoco/train_lmdb).",
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


if __name__ == "__main__":
    args = parse_args()

    init_logger(".hat_logs/mscoco_viz")

    dataset = Coco(data_path=args.data_path)
    dataset = pickle.loads(pickle.dumps(dataset))
    viz = DetViz(
        is_plot=args.save_path is not None,
        save_path=args.save_path,
        score_thresh=0.0,
        class_names=COCO_CLASS_NAMES,
    )

    logger.info(
        "Visualizing up to %d samples from %s", args.viz_num, args.data_path
    )
    for i, data in enumerate(dataset):
        if i >= args.viz_num:
            break
        img = data["img"]
        gt_bboxes = torch.from_numpy(data["gt_bboxes"])
        gt_classes = torch.from_numpy(data["gt_classes"])
        gt_score = torch.ones_like(gt_classes)
        gt_labels = torch.cat(
            (
                gt_bboxes,
                gt_score.unsqueeze(-1),
                gt_classes.unsqueeze(-1),
            ),
            -1,
        )
        viz({"pred_bboxes": [gt_labels]}, {"img": img, "ori_img": img})
