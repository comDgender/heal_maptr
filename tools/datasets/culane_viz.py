"""Visualize packed CuLane LMDB samples.

Run from HEAL root with PYTHONPATH set:

    PYTHONPATH=. python3 tools/datasets/culane_viz.py \
        --data-path /horizon-bucket/hat_data/data/pack_data/\
heal_data/CULane/test_lmdb \
        --viz-num 20 \
        --save-dir ./viz_out/culane

    PYTHONPATH=. python3 tools/datasets/culane_viz.py \
        --data-path .../train_lmdb \
        --plot
"""

import argparse
import os
import pickle

from hat_engine.utils.logger import init_logger

from heal.data.datasets.culane_dataset import CuLaneDataset
from heal.visualize.lane_lines import LanelineViz


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize CuLane dataset.")
    parser.add_argument(
        "--data-path",
        required=True,
        help="Packed LMDB path (e.g. .../CULane/test_lmdb).",
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
        default="WORKSPACE/culane_viz",
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

    init_logger(".hat_logs/culane_viz")
    save_dir = None if args.no_save else args.save_dir

    dataset = CuLaneDataset(data_path=args.data_path, to_rgb=True)
    dataset = pickle.loads(pickle.dumps(dataset))
    viz = LanelineViz(is_plot=True if args.plot else False, save_path=save_dir)

    num = 0
    for i, data in enumerate(dataset):
        img = data["ori_img"]
        gt_lines = data["ori_gt_lines"]
        print(data["image_name"])
        viz(gt_lines, {"ori_img": img, "image_name": data["image_name"]})
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
