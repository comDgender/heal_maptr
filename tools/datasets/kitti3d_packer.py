"""Build kitti3d.

除了把 KITTI 3D 原始数据打包成 lmdb，本脚本还可一键生成训练/打包所需的
中间产物（info pkl / velodyne_reduced / gt_database / dbinfos）。这些中间产物
由 ``--prepare-data`` 触发，与 split 无关，只需在首次准备数据时执行一次。
"""

import argparse
import os
import shutil

from hat_engine.utils.logger import init_logger

from heal.data.datasets.kitti3d import (
    Kitti3DDetectionPacker,
    create_groundtruth_database,
    create_kitti_info_file,
    create_reduced_point_cloud,
)

# create 三步产出的、与 split 无关的中间产物。这些文件与原始数据强耦合
# （velodyne_reduced 与 velodyne 同树、info pkl 记录相对路径），生成在 src 下；
# 为保证 config 读取的 target 目录含完整数据，需把它们复制到 target 目录。
_KITTI_INTERMEDIATE_FILES = [
    "kitti3d_infos_train.pkl",
    "kitti3d_infos_val.pkl",
    "kitti3d_infos_test.pkl",
    "kitti3d_infos_trainval.pkl",
    "kitti3d_dbinfos_train.pkl",
]
_KITTI_INTERMEDIATE_DIRS = [
    "kitti3d_gt_database",
]


def prepare_data(src_dir, target_dir):
    """Generate KITTI 3D intermediate files under src_dir; copy to target_dir.

    Runs the three ``create_*`` helpers in order against ``src_dir`` (the raw
    KITTI root containing ImageSets/ training/ testing/), then copies the
    resulting info pkls / gt_database / dbinfos to ``target_dir`` so the latter
    holds a complete, self-contained dataset for config consumption.
    """
    print("Preparing KITTI 3D intermediate files under %s" % src_dir)
    create_kitti_info_file(src_dir)
    create_reduced_point_cloud(src_dir)
    create_groundtruth_database(src_dir)

    os.makedirs(target_dir, exist_ok=True)
    for name in _KITTI_INTERMEDIATE_FILES:
        src_file = os.path.join(src_dir, name)
        if os.path.isfile(src_file):
            shutil.copy2(src_file, os.path.join(target_dir, name))
            print("Copied %s -> %s" % (src_file, target_dir))
    for name in _KITTI_INTERMEDIATE_DIRS:
        src_dir_item = os.path.join(src_dir, name)
        if os.path.isdir(src_dir_item):
            dst_dir_item = os.path.join(target_dir, name)
            if os.path.isdir(dst_dir_item):
                shutil.rmtree(dst_dir_item)
            shutil.copytree(src_dir_item, dst_dir_item)
            print("Copied %s/ -> %s" % (name, target_dir))
    print("KITTI 3D intermediate files ready under %s" % target_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Pack kitti3d dataset.")
    parser.add_argument(
        "--src-data-dir",
        required=True,
        help="The directory that contains unpacked image files.",
    )
    parser.add_argument(
        "--pack-type",
        default="",
        help="The target pack type for result of packer (e.g. lmdb).",
    )
    parser.add_argument(
        "--target-data-dir",
        default="",
        help="The directory for result of packer",
    )
    parser.add_argument(
        "--split-name", default="train", help="The split to pack."
    )
    parser.add_argument(
        "--num-workers",
        default=20,
        help="The number of workers to load image.",
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Generate intermediate files (info pkl / velodyne_reduced / "
        "gt_database / dbinfos) under src-data-dir and copy them to "
        "target-data-dir. Split-independent; run once before packing. "
        "When set, only data preparation is performed (no lmdb packing).",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    init_logger(".hat_logs/kitti3d_packer")
    directory = os.path.expanduser(args.src_data_dir)
    print("Loading dataset from %s" % directory)

    if args.target_data_dir == "":
        args.target_data_dir = args.src_data_dir
    target_dir = os.path.expanduser(args.target_data_dir)

    if args.prepare_data:
        # Only prepare intermediate files; do not pack lmdb in this run.
        prepare_data(directory, target_dir)
        raise SystemExit(0)

    split_name = args.split_name

    pack_path = os.path.join(
        target_dir,
        "%s_%s" % (split_name, args.pack_type),
    )

    packer = Kitti3DDetectionPacker(
        directory,
        pack_path,
        args.split_name,
        int(args.num_workers),
        args.pack_type,
        None,
    )
    packer()
