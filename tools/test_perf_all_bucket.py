# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
r"""Measure the bucket's release-hbdk hbm via test_perf_all_25's driver.

The bucket stores ``<model>[/qim]/<nash_x>/compile/model.hbm``. Each file is
symlinked to ``tmp_models_bucket/<entry>/compile/model_<nash-x>.hbm`` -- the
name ``test_perf_all.py`` expects -- and the inner script is pointed there
with ``--opts hbir_compiler.out_path``, so locally compiled hbm under
``tmp_models/`` are untouched. Only (model, march) pairs that exist in the
bucket are measured.

This adapter only pulls bucket hbm into the layout the driver wants and drives
it; summary + version collection live in the driver.

    PYTHONPATH=. python3 tools/test_perf_all_bucket.py --list
    PYTHONPATH=. python3 tools/test_perf_all_bucket.py --measure all
    PYTHONPATH=. python3 tools/test_perf_all_bucket.py --only resnet50
"""

import argparse
import logging
import os

from tools.test_perf_all_25 import (
    BOARDS,
    MARCH_ORDER,
    MODELS,
    collect_versions,
    run_one,
    write_summary,
)

logger = logging.getLogger("test_perf_all_bucket")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUCKET = "/horizon-bucket/hat_data/users/rui.xu/heal_release_hbm/tmp_models"
MIRROR = os.path.join(REPO_ROOT, "tmp_models_bucket")
OUT_ROOT = os.path.join(REPO_ROOT, "perf_results_bucket")

# Batch entry -> bucket subdir, where they differ: a subgraph is stored under
# its parent model's dir. Bucket dirs use the plain entry name (no dataset
# suffix) once the 2026-09 rename lands; until then _nuscenes models resolve
# via --bucket to a renamed copy or after the rename.
BUCKET_SUBDIR = {
    "motr_efficientnetb3_qim": "motr_efficientnetb3/qim",
}


def bucket_hbm(entry, march):
    """Path of the bucket-compiled hbm for one (entry, march)."""
    sub = BUCKET_SUBDIR.get(entry, entry)
    return os.path.join(
        BUCKET, sub, march.replace("-", "_"), "compile", "model.hbm"
    )


def link_hbm(entry, march):
    """Symlink the bucket hbm into the layout test_perf_all.py expects."""
    dst = os.path.join(MIRROR, entry, "compile", "model_%s.hbm" % march)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(bucket_hbm(entry, march), dst)
    return dst


def available_pairs(entries, marches):
    """(entry, march) pairs the bucket actually has an hbm for."""
    return [
        (e, m)
        for e in entries
        for m in marches
        if os.path.exists(bucket_hbm(e, m))
    ]


def parse_args():
    p = argparse.ArgumentParser(
        description="Measure bucket-compiled hbm on J6M/J6P/J6B."
    )
    p.add_argument("--measure", default="all")
    p.add_argument("--perf-time", default="1")
    p.add_argument(
        "--fps-thread-num",
        type=int,
        default=None,
        help="Override the FPS-case thread count (default 8). Lower for "
        "board/model combos that OOM at 8 threads, e.g. J6B heavy -> 4.",
    )
    p.add_argument("--only", default=None)
    p.add_argument("--boards", default=None)
    p.add_argument("--bucket", default=BUCKET)
    p.add_argument(
        "--list",
        action="store_true",
        help="Print the bucket availability matrix and exit.",
    )
    p.add_argument("--no-clean-board-hbm", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    args.skip_compile = True
    args.no_board = False
    args.out_root = "perf_results_bucket"
    return args


def main():
    logging.basicConfig(
        format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO
    )
    args = parse_args()
    global BUCKET
    BUCKET = args.bucket

    entries = args.only.split(",") if args.only else MODELS
    entries = [e.strip() for e in entries if e.strip()]
    unknown = [e for e in entries if e not in MODELS]
    if unknown:
        raise SystemExit("unknown model(s): %s" % unknown)
    marches = args.boards.split(",") if args.boards else MARCH_ORDER
    marches = [m.strip() for m in marches if m.strip()]
    bad = [m for m in marches if m not in BOARDS]
    if bad:
        raise SystemExit("unknown board(s): %s" % bad)

    pairs = available_pairs(entries, marches)
    if args.list:
        for e in entries:
            have = [m for m in marches if (e, m) in pairs]
            print("%-58s %s" % (e, " ".join(have) or "-"))
        print("%d pairs" % len(pairs))
        return

    out_root = os.path.join(REPO_ROOT, args.out_root)
    versions = collect_versions(marches)
    fails = []
    for i, (entry, march) in enumerate(pairs, 1):
        logger.info(
            "########## [%d/%d] %s :: %s ##########",
            i,
            len(pairs),
            entry,
            march,
        )
        dst = link_hbm(entry, march)
        logger.info("linked %s -> %s", dst, bucket_hbm(entry, march))
        extra = [
            "--opts",
            "hbir_compiler.out_path",
            os.path.join(os.path.dirname(dst), "model.hbm"),
        ]
        try:
            rc = run_one(
                entry,
                march,
                BOARDS[march],
                args,
                out_root,
                extra=extra,
            )
            if rc != 0:
                fails.append((entry, march, rc))
        except Exception as e:  # noqa: B902
            logger.error("[%s/%s] outer error: %s", entry, march, e)
            fails.append((entry, march, str(e)))

    rows = write_summary(entries, marches, out_root, versions)
    bad_rows = [k for k, r in rows.items() if r["status"] != "ok"]
    logger.info("=" * 60)
    logger.info(
        "DONE: %d pairs, %d non-zero rc, %d non-ok rows",
        len(pairs),
        len(fails),
        len(bad_rows),
    )
    for entry, march, why in fails:
        logger.info("  FAIL %s / %s : %s", entry, march, why)
    for k in bad_rows:
        logger.info("  NOT-OK %s / %s : %s", k[0], k[1], rows[k]["status"])


if __name__ == "__main__":
    main()
