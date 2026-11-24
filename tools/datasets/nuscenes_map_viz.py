# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import argparse
import os
import pickle

import msgpack
import numpy as np
from hat_engine.utils.logger import init_logger
from hat_engine.utils.pack_type import get_packtype_from_path
from matplotlib import pyplot as plt

from heal.data.datasets.nuscenes.nuscenes_map_gt import NuscenesMapGtDataset
from heal.visualize.nuscenes_map import NuscenesMapViz


def _read_meta(data_path):
    """Read the packer-written ``__meta__`` from the LMDB (pc_range etc.)."""
    pack_type = get_packtype_from_path(data_path)
    pack_file = pack_type(data_path, writable=False)
    pack_file.open()
    raw = pack_file.read("__meta__")
    pack_file.close()
    if raw is None:
        return None
    return msgpack.unpackb(raw, raw=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize packed NuScenes map-vector GT LMDB samples."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Packed map GT LMDB path "
        "(e.g. tmp_data/heal_data/nuscenes/v1.0-trainval/val_map_lmdb).",
    )
    parser.add_argument(
        "--viz-num",
        type=int,
        default=10,
        help="Max number of samples to visualize.",
    )
    parser.add_argument(
        "--save-dir",
        default="WORKSPACE/nuscenes_map_viz",
        help="Directory to save visualization PNGs.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show plots interactively (requires display).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save PNGs; use with --plot.",
    )
    parser.add_argument(
        "--pc-range",
        type=float,
        nargs=6,
        default=None,
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
        help="Point cloud range; auto-read from the LMDB __meta__ when "
        "unset. Override only if the LMDB has no __meta__.",
    )
    parser.add_argument(
        "--fixed-ptsnum-per-line",
        type=int,
        default=None,
        help="Points per map line; auto-read from the LMDB __meta__ when "
        "unset (default 20).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.no_save and not args.plot:
        raise SystemExit(
            "Use --save-dir (default) or --plot; --no-save needs --plot."
        )

    init_logger(".hat_logs/nuscenes_map_viz")
    save_dir = None if args.no_save else args.save_dir

    # Resolve pc_range / fixed_ptsnum from the LMDB __meta__ unless overridden.
    meta = _read_meta(args.data_path)
    if args.pc_range is not None:
        pc_range = list(args.pc_range)
    elif meta is not None and "pc_range" in meta:
        pc_range = list(meta["pc_range"])
    else:
        raise SystemExit(
            "LMDB has no __meta__ with pc_range; pass --pc-range explicitly."
        )
    fixed_ptsnum = args.fixed_ptsnum_per_line
    if fixed_ptsnum is None:
        fixed_ptsnum = meta.get("fixed_ptsnum_per_line", 20) if meta else 20

    dataset = NuscenesMapGtDataset(
        data_path=args.data_path,
        pc_range=pc_range,
        fixed_ptsnum_per_line=fixed_ptsnum,
    )
    dataset = pickle.loads(pickle.dumps(dataset))

    # We reuse NuscenesMapViz only for its draw_ego_lines() plotting method
    # (pc_range xlim/ylim + per-class polyline colors + optional ego car),
    # NOT its __call__, which builds a fixed subplots(1,3)/figsize(16,24)
    # layout meant for multi-camera + map inference. With imgs=[] that leaves
    # 2 empty axes and a distorted tall figure. Instead we draw our own
    # single-panel figure with an aspect ratio matching pc_range.
    viz = NuscenesMapViz(pc_range=tuple(pc_range), is_plot=True)

    # pc_range span -> figure aspect ratio (so the map isn't stretched).
    x_span = pc_range[3] - pc_range[0]
    y_span = pc_range[4] - pc_range[1]
    fig_w = 8.0
    fig_h = fig_w * (y_span / x_span) if x_span > 0 else 8.0
    # Clamp so very wide/tall ranges still get a usable figure.
    fig_h = max(4.0, min(fig_h, 16.0))

    num = 0
    for i, data in enumerate(dataset):
        gt_instances = data["gt_instances"]
        labels = list(data["gt_labels_map"])
        # Each LineString's coords is an (M, 2) XY array; feed the GT
        # polylines to draw_ego_lines (same method __call__ uses for preds).
        pts_3d = [
            np.asarray(line.coords) for line in gt_instances.instance_list
        ]

        fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
        viz.draw_ego_lines(ax, pts_3d, labels, "GT_MAP")
        ax.set_aspect("equal", adjustable="box")

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            out_png = os.path.join(save_dir, f"nusc_map_gt_{i}.png")
            fig.savefig(out_png, bbox_inches="tight")
            plt.close(fig)
        elif args.plot:
            plt.show()
        num += 1
        if i >= args.viz_num - 1:
            break

    if save_dir is not None and num > 0:
        print(
            "Saved %d visualization(s) to %s"
            % (num, os.path.abspath(save_dir))
        )
    elif args.plot and num > 0:
        print("Displayed %d sample(s)." % num)
    elif num == 0:
        print("No samples found; no images written.")


if __name__ == "__main__":
    main()
