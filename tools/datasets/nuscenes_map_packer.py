# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import argparse
import os

from hat_engine.utils.logger import init_logger

from heal.data.datasets.nuscenes.nuscenes_map_gt import NuscenesMapGtPacker


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pack NuScenes map vector GT."
    )
    parser.add_argument(
        "--src-data-dir",
        required=True,
        help="NuScenes root (contains samples/, maps/, ...).",
    )
    parser.add_argument(
        "--map-path",
        required=True,
        help="NuScenes map expansion root (meta directory).",
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
        "--pc-range",
        type=float,
        nargs=6,
        required=True,
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
        help="Point cloud / map patch range, same as training config.",
    )
    parser.add_argument(
        "--bev-size",
        type=int,
        nargs=2,
        default=[100, 50],
        metavar=("H", "W"),
        help="BEV canvas size (bev_h, bev_w).",
    )
    parser.add_argument(
        "--fixed-ptsnum-per-line",
        type=int,
        default=20,
        help="Fixed points per map line instance.",
    )
    parser.add_argument(
        "--padding-value",
        type=int,
        default=-10000,
        help="Padding value for map line sampling.",
    )
    parser.add_argument(
        "--map-classes",
        nargs="+",
        default=["divider", "ped_crossing", "boundary"],
        help="Map classes to pack.",
    )
    parser.add_argument(
        "--use-ego-gt",
        action="store_true",
        help="Use ego pose instead of lidar pose for map GT (default: lidar).",
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
    init_logger(".hat_logs/nuscenes_map_packer")

    src_dir = os.path.expanduser(args.src_data_dir)
    map_path = os.path.expanduser(args.map_path)
    target_dir = args.target_data_dir or src_dir
    pack_path = os.path.join(
        target_dir, "%s_map_%s" % (args.split_name, args.pack_type)
    )

    print("Loading NuScenes map GT from %s" % src_dir)
    print("Map expansion path: %s" % map_path)
    print("Writing map GT pack to %s" % pack_path)

    packer = NuscenesMapGtPacker(
        args.version,
        src_dir,
        pack_path,
        args.split_name,
        args.num_workers,
        args.pack_type,
        map_path=map_path,
        pc_range=args.pc_range,
        map_classes=args.map_classes,
        fixed_ptsnum_per_line=args.fixed_ptsnum_per_line,
        bev_size=tuple(args.bev_size),
        padding_value=args.padding_value,
        use_lidar_gt=not args.use_ego_gt,
    )
    packer()
