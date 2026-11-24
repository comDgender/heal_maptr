"""Pack ImageNet raw data into LMDB for classification."""

import argparse
import os

from hat_engine.utils.logger import init_logger

from heal.data.datasets.imagenet import ImageNetPacker


def parse_args():
    parser = argparse.ArgumentParser(description="Pack ImageNet dataset.")
    parser.add_argument(
        "--src-data-dir",
        required=True,
        help="ImageNet root (contains train/, val/, etc.).",
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
        default="train",
        choices=["train", "val"],
        help="Split to pack.",
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
    init_logger(".hat_logs/imagenet_packer")

    src_dir = os.path.expanduser(args.src_data_dir)
    target_dir = args.target_data_dir or src_dir
    pack_path = os.path.join(
        target_dir, "%s_%s" % (args.split_name, args.pack_type)
    )

    print("Loading ImageNet from %s" % src_dir)
    print("Writing pack to %s" % pack_path)

    packer = ImageNetPacker(
        src_dir,
        pack_path,
        args.split_name,
        args.num_workers,
        args.pack_type,
    )
    packer()
