# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import argparse
import os

from hat_engine.utils.logger import init_logger

from heal.data.datasets.navsim.navsim_dataset_packer import NavsimPacker
from heal.data.datasets.navsim.navsim_metric_cache_dataset import (
    NavsimMetricCachePacker,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Pack NavSim dataset.")
    parser.add_argument(
        "--src-data-dir",
        required=True,
        help="The directory that contains unpacked image files.",
    )
    parser.add_argument(
        "--pack-type",
        required=True,
        help="The pack data type for result of packer",
    )
    parser.add_argument(
        "--target-data-dir",
        default="",
        help="The directory for result of packer",
    )
    parser.add_argument(
        "--split-name", default="trainval", help="The split to pack."
    )
    parser.add_argument(
        "--num-workers",
        default=20,
        help="The number of workers to load image.",
    )
    parser.add_argument(
        "--pack-metric-cache",
        action="store_true",
        help="If True, pack metric cache instead of the raw dataset.",
    )
    parser.add_argument(
        "--sync-data-dir",
        default="",
        help="Optional synthetic data directory.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to pack.",
    )
    parser.add_argument(
        "--map-root",
        default="",
        help="Map root directory.",
    )
    parser.add_argument(
        "--split-cfg-root",
        default="",
        help="Scene filter config root directory.",
    )
    parser.add_argument(
        "--synth-data-path",
        default="",
        help="Optional synthetic data path for NavSim raw packing.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    init_logger(".hat_logs/navsim_packer")
    directory = os.path.expanduser(args.src_data_dir)
    print(f"Loading dataset from {directory}")

    if args.target_data_dir == "":
        args.target_data_dir = args.src_data_dir

    pack_path = os.path.join(
        args.target_data_dir,
        f"{args.split_name}_{args.pack_type}",
    )

    if args.pack_metric_cache:
        pack_path = os.path.join(
            args.target_data_dir,
            f"metric_cache_{args.split_name}_{args.pack_type}",
        )
        packer = NavsimMetricCachePacker(
            directory,
            target_data_dir=pack_path,
            split_name=args.split_name,
            num_workers=int(args.num_workers),
            pack_type=args.pack_type,
            map_root=args.map_root,
            split_cfg_root=args.split_cfg_root or None,
        )
    else:
        packer = NavsimPacker(
            directory,
            pack_path,
            args.split_name,
            args.split_cfg_root,
            num_workers=int(args.num_workers),
            synth_data_path=args.synth_data_path or None,
            pack_type=args.pack_type,
        )

    packer()
