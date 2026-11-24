"""Visualize packed Argoverse 2 LMDB samples (pack-verification).

Reuses the inference-side visualizer ``Argoverse2Viz`` (which already knows
how to draw packed map + historical/future GT trajectories) by calling it
with ``preds=None`` / ``map_path=None``, i.e. ground-truth only — no model
needed. Use it to spot-check that an LMDB produced by
``argoverse2_packer.py`` is correct (agent tracks + packed map aligned).

``Argoverse2Viz.__call__`` writes a *fixed* ``traj_pred.png`` and does not
let you name it, so each sample is saved under its own sub-directory
(``{save_dir}/sample_{i}/traj_pred.png``) to avoid overwriting.

Run from the HEAL root with PYTHONPATH set (``--save-dir`` must be a
locally writable path; ``tmp_data/heal_data`` is a read-only bucket):

    PYTHONPATH=. python3 tools/datasets/argoverse2_viz.py \
        --data-path ./tmp_data/heal_data/argoverse2/val_lmdb \
        --viz-num 10 \
        --save-dir ./viz_out/argoverse2

    PYTHONPATH=. python3 tools/datasets/argoverse2_viz.py \
        --data-path .../val_lmdb --plot --no-save

Defaults (num_historical_steps=10, num_future_steps=12, input_dim=2) match
the qcnet_oe task config; override via flags for other settings.
"""

import argparse
import os
import pickle

from hat_engine.utils.logger import init_logger
from matplotlib import pyplot as plt

from heal.data.datasets.argoverse2_dataset import Argoverse2PackedDataset
from heal.visualize.argoverse2 import Argoverse2Viz


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize packed Argoverse 2 LMDB samples."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Packed LMDB path (e.g. tmp_data/heal_data/argoverse2/val_lmdb).",
    )
    parser.add_argument(
        "--viz-num",
        type=int,
        default=10,
        help="Max number of samples to visualize.",
    )
    parser.add_argument(
        "--save-dir",
        default="WORKSPACE/argoverse2_viz",
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
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to read (default: val).",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=2,
        help="Input dimension for argoverse2 decode (default: 2).",
    )
    parser.add_argument(
        "--num-historical-steps",
        type=int,
        default=10,
        help="Historical steps; splits hist/future in viz (default: 10).",
    )
    parser.add_argument(
        "--num-future-steps",
        type=int,
        default=12,
        help="Future steps for argoverse2 decode (default: 12).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.no_save and not args.plot:
        raise SystemExit(
            "Use --save-dir (default) or --plot; --no-save needs --plot."
        )

    init_logger(".hat_logs/argoverse2_viz")
    save_dir = None if args.no_save else args.save_dir

    # pickle trick to make the lmdb handle multi-process-safe (matches the
    # mot17_viz / culane_viz pattern).
    dataset = Argoverse2PackedDataset(
        data_path=args.data_path,
        split=args.split,
        input_dim=args.input_dim,
        num_historical_steps=args.num_historical_steps,
        num_future_steps=args.num_future_steps,
    )
    dataset = pickle.loads(pickle.dumps(dataset))

    # Argoverse2Viz's save branch is nested under is_plot, so force
    # is_plot=True to make --save-dir work without --plot (see mot17_viz).
    viz = Argoverse2Viz(
        num_historical_steps=args.num_historical_steps,
        is_plot=True,
    )

    num = 0
    for i, data in enumerate(dataset):
        # preds=None -> draw ground truth only; map_path=None -> use the
        # packed map already in `data` (plot_packed_map), no av2 map file.
        # Argoverse2Viz.__call__ saves to a *fixed* "{save_path}/traj_pred.png"
        # and does not close its figure, so per sample we give it a unique
        # sub-directory (otherwise multiple samples overwrite the same file)
        # and close the figure afterwards to avoid leaking across samples.
        sample_save = (
            os.path.join(save_dir, f"sample_{i}")
            if save_dir is not None
            else None
        )
        viz(data, None, map_path=None, save_path=sample_save)
        plt.close("all")
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
