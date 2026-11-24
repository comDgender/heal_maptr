"""Pack official MOT17 raw data into LMDB for motr training.

Reads the official MOT17 layout (``--src-data-dir`` = the MOT17 root with
``train/`` and ``test/``). Only the ``train/*SDP`` sequences are used because
official ``test/`` has no public gt; each train SDP sequence is split in half
in memory (first half -> train split, second half -> test split with frame
ids renumbered to start at 1). No intermediate ``split_data`` /
``labels_with_ids`` files are written — the split and label generation that
HAT's ``gen_mot_data.py`` used to do are folded into the packer.

Output:
  - {target_data_dir}/{split_name}_{pack_type}  (train_lmdb / test_lmdb)
  - {target_data_dir}/test_gt/{seq}/gt/gt.txt   (only for --split-name test;
    renumbered in memory from the original train/<seq>/gt/gt.txt second half,
    for MotMetric evaluation, since motmetrics only reads raw gt.txt text)
"""

import argparse
import os

from hat_engine.utils.logger import init_logger

from heal.data.datasets.mot17_dataset import Mot17Packer


def parse_args():
    parser = argparse.ArgumentParser(description="Pack MOT17 dataset.")
    parser.add_argument(
        "--src-data-dir",
        required=True,
        help="MOT17 root containing train/ and test/ (official MOT17 "
        "layout). Only train/*SDP sequences are used; official test/ has no "
        "public gt, so the packer splits each train SDP sequence in half "
        "(first half -> train split, second half -> test split).",
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

    init_logger(".hat_logs/mot17_packer")
    directory = os.path.expanduser(args.src_data_dir)
    print("Loading dataset from %s" % directory)

    if args.target_data_dir == "":
        args.target_data_dir = args.src_data_dir

    pack_path = os.path.join(
        args.target_data_dir,
        "%s_%s" % (args.split_name, args.pack_type),
    )

    packer = Mot17Packer(
        directory,
        pack_path,
        args.split_name,
        args.num_workers,
        args.pack_type,
    )
    packer()
