# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import argparse
import os
import pickle

import numpy as np
import torch
from hat_engine.utils.logger import init_logger

from heal.data.datasets.nuscenes.nuscenes_lidar import NuscenesLidarDataset
from heal.visualize.lidar_det import lidar_det_visualize


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize packed NuScenes lidar LMDB samples."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Packed lidar LMDB path "
        "(e.g. tmp_data/heal_data/nuscenes/v1.0-trainval/val_lidar_lmdb).",
    )
    parser.add_argument(
        "--viz-num",
        type=int,
        default=10,
        help="Max number of samples to visualize.",
    )
    parser.add_argument(
        "--save-dir",
        default="WORKSPACE/nuscenes_lidar_viz",
        help="Directory to save visualization PNGs.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show plots interactively (requires display).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save PNGs; use with --plot.",
    )
    parser.add_argument(
        "--num-sweeps",
        type=int,
        default=0,
        help="Number of past lidar sweeps to load (default: 0 = keyframe "
        "only, fastest and enough for a pack-verification spot check).",
    )
    parser.add_argument(
        "--with-velocity",
        action="store_true",
        default=True,
        help="Keep velocity columns in GT boxes (default True; matches "
        "the pack).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.no_save and not args.plot:
        raise SystemExit(
            "Use --save-dir (default) or --plot; --no-save needs --plot."
        )

    init_logger(".hat_logs/nuscenes_lidar_viz")
    save_dir = None if args.no_save else args.save_dir

    # test_mode=True => deterministic sweep selection and skip the random
    # empty-GT resample loop in NuscenesLidarDataset.__getitem__.
    dataset = NuscenesLidarDataset(
        data_path=args.data_path,
        num_sweeps=args.num_sweeps,
        with_velocity=args.with_velocity,
        test_mode=True,
        filter_empty_gt=False,
    )
    dataset = pickle.loads(pickle.dumps(dataset))

    num = 0
    for i, data in enumerate(dataset):
        points = data["lidar"]["points"]
        annotations = data["lidar"]["annotations"]
        boxes = np.asarray(annotations["boxes"], dtype=np.float32).copy()
        labels = torch.from_numpy(
            np.asarray(annotations["labels"], dtype=np.int64)
        )

        # Convert (N, 9) (x,y,z,l,w,h,vx,vy,yaw) -> viz layout: swap
        # cols 3/4 to (w,l) so the BEV footprint matches; reverse=True makes
        # get_boxes read the yaw from the last column (index -1).
        if boxes.shape[1] >= 5:
            boxes[:, [3, 4]] = boxes[:, [4, 3]]
        scores = torch.ones(boxes.shape[0], dtype=torch.float32)
        gt_data = (torch.from_numpy(boxes), labels, scores)

        # Per-sample save_name so samples don't overwrite each other (the
        # viz function uses a fixed name by default).
        save_name = f"nusc_lidar_gt_{i}.png"

        lidar_det_visualize(
            points,
            gt_data,
            reverse=True,
            score_thresh=0.0,
            is_plot=True,
            save_path=save_dir,
            save_name=save_name,
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


if __name__ == "__main__":
    main()
