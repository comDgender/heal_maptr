"""Visualize packed MOT17 LMDB samples.

Run from HEAL root with PYTHONPATH set:

    PYTHONPATH=. python3 tools/datasets/mot17_viz.py \
        --data-path /horizon-bucket/hat_data/data/pack_data/\
heal_data/mot17/test_lmdb \
        --viz-num 20 \
        --save-dir ./viz_out/mot17

    PYTHONPATH=. python3 tools/datasets/mot17_viz.py \
        --data-path .../train_lmdb \
        --plot
"""

import argparse
import os
import pickle

import torch
from hat_engine.utils.logger import init_logger

from heal.data.datasets.mot17_dataset import Mot17Dataset
from heal.visualize.det import DetViz


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize MOT17 dataset.")
    parser.add_argument(
        "--data-path",
        required=True,
        help="Packed LMDB path (e.g. .../mot17/test_lmdb).",
    )
    parser.add_argument(
        "--viz-num",
        type=int,
        default=50,
        help="Max number of samples to visualize.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show plots interactively (requires display).",
    )
    parser.add_argument(
        "--save-dir",
        default="WORKSPACE/mot17_viz",
        help="Directory to save visualization PNGs.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save PNGs; use with --plot.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.no_save and not args.plot:
        raise SystemExit(
            "Use --save-dir (default) or --plot; --no-save needs --plot."
        )

    init_logger(".hat_logs/mot17_viz")
    save_dir = None if args.no_save else args.save_dir

    # pickle trick to make dataset multi-process-safe (matches culane_viz
    # pattern).
    dataset = Mot17Dataset(data_path=args.data_path)
    dataset = pickle.loads(pickle.dumps(dataset))

    # DetViz saves only when is_plot=True (its save branch is nested under
    # is_plot). Force is_plot=True so --save-dir works without --plot.
    viz = DetViz(is_plot=True, save_path=save_dir)

    num = 0
    for i, data in enumerate(dataset):
        frame_data = data["frame_data_list"][0]
        img = frame_data["img"]
        gt_bboxes = torch.from_numpy(frame_data["gt_bboxes"])
        gt_classes = torch.from_numpy(frame_data["gt_classes"])
        gt_score = torch.ones_like(gt_classes)
        gt_labels = torch.cat(
            (
                gt_bboxes,
                gt_score.unsqueeze(-1),
                gt_classes.unsqueeze(-1),
            ),
            -1,
        )
        viz(
            gt_labels,
            {"img": img, "ori_img": img, "img_name": frame_data["img_name"]},
        )
        num += 1
        if i >= args.viz_num - 1:
            break

    if save_dir is not None and num > 0:
        print(
            "Saved %d visualization(s) to %s"
            % (num, os.path.abspath(save_dir))
        )
    elif args.plot and num > 0:
        print("Displayed %d sample(s)." % num)
    elif num == 0:
        print("No samples found; no images written.")
