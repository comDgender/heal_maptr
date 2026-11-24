# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import argparse
import os

from hat_engine.utils.logger import init_logger

from heal.data.datasets.nuscenes.nuscenes_occ import NuscenesOccPacker


def parse_args():
    parser = argparse.ArgumentParser(description="Pack NuScenes occ GT.")
    parser.add_argument(
        "--src-data-dir",
        required=True,
        help="NuScenes root (contains samples/, sweeps/, occ3d/ ...).",
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
        help="Split to pack (must match the base pack).",
    )
    parser.add_argument(
        "--version",
        default="v1.0-mini",
        help="NuScenes version (must match the base pack).",
    )
    parser.add_argument(
        "--occ-gt-dir",
        default="occ3d/gts",
        help="Dir under src holding <scene>/<token>/labels.npz occ labels.",
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
    init_logger(".hat_logs/nuscenes_occ_packer")

    src_dir = os.path.expanduser(args.src_data_dir)
    target_dir = args.target_data_dir or src_dir
    pack_path = os.path.join(
        target_dir, "%s_occ_%s" % (args.split_name, args.pack_type)
    )

    print("Loading NuScenes occ from %s" % src_dir)
    print("Writing occ pack to %s" % pack_path)

    packer = NuscenesOccPacker(
        args.version,
        src_dir,
        pack_path,
        args.split_name,
        args.num_workers,
        args.pack_type,
        occ_gt_dir=args.occ_gt_dir,
    )
    packer()
