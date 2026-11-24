# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import argparse
import logging
import os
import pickle

from hat_engine.utils.logger import init_logger

import heal  # noqa: F401
from heal.data.datasets.nuscenes.nuscenes_occ import NuscenesOccDataset
from heal.visualize.occ import OccViz

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize packed NuScenes occ3d GT."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Packed occ LMDB path (e.g. tmp_data/nuscenes/val_occ_lmdb/).",
    )
    parser.add_argument(
        "--viz-num",
        type=int,
        default=1,
        help="Max number of samples to visualize.",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Directory to save visualization PNGs.",
    )
    parser.add_argument(
        "--vis-3d",
        action="store_true",
        help="Also render the 3D occ view (requires mayavi + offscreen X).",
    )
    parser.add_argument(
        "--vcs-range",
        type=float,
        nargs=6,
        default=[-40.0, -40.0, -1.0, 40.0, 40.0, 5.4],
        metavar=("X0", "Y0", "Z0", "X1", "Y1", "Z1"),
        help="VCS range passed to OccViz for 3D rendering.",
    )
    return parser.parse_args()


def _build_dataset(data_path):
    return NuscenesOccDataset(data_path=data_path)


if __name__ == "__main__":
    args = parse_args()
    init_logger(".hat_logs/nuscenes_occ_viz", level=logging.INFO)

    dataset = _build_dataset(args.data_path)
    dataset = pickle.loads(pickle.dumps(dataset))
    viz = OccViz(
        vcs_range=tuple(args.vcs_range),
        vis_bev_2d=True,
        vis_occ_3d=args.vis_3d,
    )

    logger.info(
        "Visualizing up to %d samples from %s", args.viz_num, args.data_path
    )

    num_saved = 0
    for i, data in enumerate(dataset):
        voxel_semantics = data["gt_occ_info"]["voxel_semantics"]
        sample_save_path = None
        if args.save_path is not None:
            sample_save_path = os.path.join(args.save_path, f"occ_gt_{i}")
        viz(voxel_semantics, save_path=sample_save_path)
        num_saved += 1
        if i >= int(args.viz_num) - 1:
            break

    if args.save_path is not None and num_saved > 0:
        abs_dir = os.path.abspath(args.save_path)
        print(f"Saved {num_saved} visualization(s) to {abs_dir}")
    elif num_saved == 0:
        print("No samples found in occ pack; no images written.")
