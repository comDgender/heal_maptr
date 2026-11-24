# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import argparse
import os

from hat_engine.utils.logger import init_logger

from heal.data.datasets.navsim.diffusion_planner_packer import (
    DiffusionPlannerNavsimPacker,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pack navsim dataset for diffusion planner."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the packed (lmdb) navsim scenes.",
    )
    parser.add_argument(
        "--original-sensor-path",
        default=None,
        help="Legacy sensor_blobs path; unused (NavsimDataset is pack-only).",
    )
    parser.add_argument(
        "--split",
        default="navtrain",
        help="Dataset split name (default: navtrain).",
    )
    parser.add_argument(
        "--split-cfg-root",
        default=None,
        help="Root directory for navsim split config files.",
    )
    parser.add_argument(
        "--map-root",
        default=None,
        help="Path to nuplan maps directory.",
    )
    parser.add_argument(
        "--pack-type",
        default="lmdb",
        help="The pack data type for result of packer.",
    )
    parser.add_argument(
        "--target-data-dir",
        default="",
        help="The directory for result of packer.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=20,
        help="The number of workers to load data.",
    )
    parser.add_argument(
        "--agent-num",
        type=int,
        default=32,
        help="Max number of neighbor agents.",
    )
    parser.add_argument(
        "--model-time-len",
        type=int,
        default=4,
        help="Temporal length for agent features at 2Hz.",
    )
    parser.add_argument(
        "--static-objects-num",
        type=int,
        default=5,
        help="Max number of static objects.",
    )
    parser.add_argument(
        "--max-ped-bike",
        type=int,
        default=10,
        help="Max pedestrian/bicycle agents.",
    )
    parser.add_argument(
        "--lane-num",
        type=int,
        default=70,
        help="Max number of lane elements.",
    )
    parser.add_argument(
        "--lane-len",
        type=int,
        default=20,
        help="Number of points per lane element.",
    )
    parser.add_argument(
        "--route-num",
        type=int,
        default=25,
        help="Max route lane elements.",
    )
    parser.add_argument(
        "--route-len",
        type=int,
        default=20,
        help="Points per route lane element.",
    )
    parser.add_argument(
        "--predicted-neighbor-num",
        type=int,
        default=10,
        help="Number of neighbor agents for future GT.",
    )
    parser.add_argument(
        "--num-future-poses",
        type=int,
        default=8,
        help="Number of future time steps at 2Hz.",
    )
    parser.add_argument(
        "--raw-frequency",
        type=float,
        default=0.5,
        help="The frequency of the original data.",
    )
    parser.add_argument(
        "--pad-frequency",
        type=float,
        default=0.5,
        help="The frequency of the processed data.",
    )
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "test"],
        help="Train or test; also used as the output subdirectory name.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=100.0,
        help="The scope of Agent.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to pack (default: all).",
    )
    parser.add_argument(
        "--agents-future-frames",
        type=int,
        default=8,
        help="The number of frames in the future trajectory of the agent.",
    )
    parser.add_argument(
        "--no-lane-boundary",
        dest="add_lane_boundary",
        action="store_false",
        help="Disable lane boundary extraction (on by default; keep it on "
        "for configs with use_lane_boundary=True).",
    )
    parser.add_argument(
        "--map-min-length",
        type=float,
        default=5.0,
        help="The shortest length of the retained lane boundary.",
    )
    parser.add_argument(
        "--pc-range",
        type=float,
        nargs=6,
        default=None,
        help="Point cloud range, e.g. -51.2 -51.2 -4.0 102.4 51.2 4.0",
    )
    parser.add_argument(
        "--map-pc-range",
        type=float,
        nargs=6,
        default=None,
        help="Map point cloud range, e.g. -25.6 -51.2 -4.0 102.4 51.2 4.0",
    )
    parser.add_argument(
        "--map-version",
        default="nuplan-maps-v1.0",
        help="The version of nuplan map.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    init_logger(".hat_logs/diffusion_planner_navsim_packer")
    print("Loading dataset from %s" % args.data_path)

    target_data_dir = args.target_data_dir or "./"
    pack_path = os.path.join(target_data_dir, args.mode)

    packer = DiffusionPlannerNavsimPacker(
        data_path=args.data_path,
        split=args.split,
        split_cfg_root=args.split_cfg_root,
        map_root=args.map_root,
        num_workers=args.num_workers,
        pack_type=args.pack_type,
        target_data_dir=pack_path,
        agent_num=args.agent_num,
        model_time_len=args.model_time_len,
        static_objects_num=args.static_objects_num,
        max_ped_bike=args.max_ped_bike,
        lane_num=args.lane_num,
        lane_len=args.lane_len,
        route_num=args.route_num,
        route_len=args.route_len,
        predicted_neighbor_num=args.predicted_neighbor_num,
        num_future_poses=args.num_future_poses,
        raw_frequency=args.raw_frequency,
        pad_frequency=args.pad_frequency,
        mode=args.mode,
        radius=args.radius,
        agents_future_frames=args.agents_future_frames,
        num_samples=args.num_samples,
        add_lane_boundary=args.add_lane_boundary,
        map_min_length=args.map_min_length,
        pc_range=args.pc_range,
        map_pc_range=args.map_pc_range,
        map_version=args.map_version,
    )
    packer()
