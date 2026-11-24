"""Pack CuLane raw data into LMDB for ganet training.

Output: {target_data_dir}/{split_name}_{pack_type}  (train_lmdb / test_lmdb)
"""

import argparse
import os

from hat_engine.utils.logger import init_logger

from heal.data.datasets.culane_dataset import CuLanePacker


def parse_args():
    parser = argparse.ArgumentParser(description="Pack CuLane dataset.")
    parser.add_argument(
        "--src-data-dir",
        required=True,
        help="Dir with unpacked CuLane data (list/, driver_*/, *.lines.txt).",
    )
    parser.add_argument(
        "--pack-type",
        required=True,
        help="Target pack type (e.g. lmdb).",
    )
    parser.add_argument(
        "--target-data-dir",
        default="",
        help="Output dir; default same as src.",
    )
    parser.add_argument(
        "--split-name",
        default="train",
        choices=["train", "test"],
        help="Split to pack.",
    )
    parser.add_argument(
        "--num-workers",
        default=20,
        type=int,
        help="Num workers to load image.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    init_logger(".hat_logs/culane_packer")
    directory = os.path.expanduser(args.src_data_dir)
    print("Loading dataset from %s" % directory)

    if args.target_data_dir == "":
        args.target_data_dir = args.src_data_dir

    pack_path = os.path.join(
        args.target_data_dir,
        "%s_%s" % (args.split_name, args.pack_type),
    )

    packer = CuLanePacker(
        directory,
        pack_path,
        args.split_name,
        args.num_workers,
        args.pack_type,
    )
    packer()
