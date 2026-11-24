# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import argparse
import os

from hat_engine.utils.logger import init_logger

from heal.data.datasets.nuscenes.nuscenes_lidar import NuscenesLidarPacker


def parse_args():
    parser = argparse.ArgumentParser(description="Pack NuScenes lidar data.")
    parser.add_argument(
        "--src-data-dir",
        required=True,
        help="NuScenes root (contains samples/, sweeps/, ...).",
    )
    parser.add_argument(
        "--pack-type",
        required=True,
        help="Pack format, e.g. lmdb.",
    )
    parser.add_argument(
        "--target-data-dir",
        default="",
        help="Output directory; default: same as src.",
    )
    parser.add_argument(
        "--split-name",
        default="val",
        choices=["train", "val"],
        help="Split to pack.",
    )
    parser.add_argument(
        "--version",
        default="v1.0-mini",
        help="NuScenes version.",
    )
    parser.add_argument(
        "--max-sweeps",
        type=int,
        default=10,
        help="Max historical sweeps stored per sample.",
    )
    parser.add_argument(
        "--load-dim",
        type=int,
        default=5,
        help="Raw point dimension in .pcd.bin files.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=20,
        help="Workers for parallel packing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    init_logger(".hat_logs/nuscenes_lidar_packer")

    src_dir = os.path.expanduser(args.src_data_dir)
    target_dir = args.target_data_dir or src_dir
    pack_path = os.path.join(
        target_dir, "%s_lidar_%s" % (args.split_name, args.pack_type)
    )

    print("Loading NuScenes lidar from %s" % src_dir)
    print("Writing lidar pack to %s" % pack_path)

    packer = NuscenesLidarPacker(
        args.version,
        src_dir,
        pack_path,
        args.split_name,
        args.num_workers,
        args.pack_type,
        max_sweeps=args.max_sweeps,
        load_dim=args.load_dim,
    )
    packer()
