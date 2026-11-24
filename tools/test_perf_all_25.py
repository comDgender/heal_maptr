# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
r"""Batch FPS/latency/peak-DDR measurement for all HEAL models across 3 boards.

Drives ``tools/test_perf_all.py`` once per (model, board): different chips live
on different physical boards that differ in ip, hrt location, *and* ssh auth
(J6M/J6P are key-less via an installed pubkey; J6B is QNX with an empty
password, which ``--password None`` cannot express in a single shared call), so
each board gets its own resolved settings and the inner script is invoked per
board.

Per the user's flow:

  * the **local** compiled hbm
    (``tmp_models/<task>/compile/model_<march>.hbm``)
    is kept -- never deleted;
  * the **board-side** uploaded hbm is deleted right after each model so the
    small partitions (esp. J6B's 507 MB /userdata, 232 MB free) do not fill up;
  * the **board-side** result logs (profiler.log / monitor.log) are left in
    place (not cleaned) for later inspection.

MOTR ships two independent subgraphs and is therefore measured twice: the
main body (``deploy.py``, pyramid image input) and QIM (``deploy_qim.py``,
all-DDR inputs). They compile to different hbm files under different
directories, so each is its own batch entry -- see SUBGRAPH_MODELS.

Run from the HEAL root with the env that has hbm_infer (HAT_new):

    PYTHONPATH=. python3 tools/test_perf_all_25.py --measure all
    PYTHONPATH=. python3 tools/test_perf_all_25.py --measure all \
        --only efficientnet --boards nash-b   # smoke test one
"""

import argparse
import csv
import glob
import logging
import os
import posixpath
import re
import shlex
import subprocess
import sys
import time

logger = logging.getLogger("test_perf_all_25")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INNER = os.path.join(REPO_ROOT, "tools", "test_perf_all.py")

SUMMARY_HEADER = "task,march,fps,latency_ms,bpu_ms,ucp_ms,peak_ddr_mib,status"

# 24 single-graph models whose deploy reads a single ckpt present in the
# bucket. Names == task_name == config dir name.
MODELS = [
    "bevformer_tiny_resnet50_detection",
    "bevfusion_pointpillar_henet_multisensor_multitask",
    "bev_sparse_henet_tinym",
    "bev_sparse_lidar_fusion_henet_tinym",
    "bev_vtv2_resnet50",
    "centerpoint_pointpillar",
    "deform_detr_resnet50",
    "detr3d_efficientnetb3",
    "efficientnet",
    "fcos3d_efficientnetb0",
    "fcos_efficientnetb3",
    "flashocc_henet_lss_occ3d",
    "ganet_mixvargenet",
    "henet_tinye",
    "henet_tinym",
    "maptr_henet_tinym_bevformer",
    "maptr_sparse_henet_tinym",
    "mixvargenet",
    "petr_efficientnetb3",
    "pointpillars",
    "qcnet_oe",
    "resnet50",
    "sparse_multitask_det_maptr_flashocc_henet_tinym",
    "unet_mobilenetv1",
]

# Multi-subgraph models: one batch entry per subgraph. The entry name is a
# label only -- what actually differs is the config path, the *task* and the
# csv location. ``task`` is what test_perf_all.py derives itself (task_name of
# the config, else the exporter's model_name) and is what names the board-side
# upload dir <board_dir>/<task>/<march>/, so the cleaner must be given the same
# value. ``out_rel`` is the per-entry csv path relative to the results root
# (see entry_out); it nests each subgraph under its parent model's dir.
SUBGRAPH_MODELS = {
    "motr_efficientnetb3": {
        "config": "configs/motr_efficientnetb3",
        "task": "motr_efficientnetb3",
        "out_rel": "motr_efficientnetb3/perf_all.csv",
    },
    "motr_efficientnetb3_qim": {
        "config": "configs/motr_efficientnetb3/deploy_qim.py",
        "task": "motr_efficientnetb3_qim",
        "out_rel": "motr_efficientnetb3/qim/perf_all.csv",
    },
}

MODELS = MODELS + list(SUBGRAPH_MODELS)


def resolve_entry(name):
    """Map a batch entry name to its (config path, task name, csv rel)."""
    entry = SUBGRAPH_MODELS.get(name)
    if entry is not None:
        return entry["config"], entry["task"], entry["out_rel"]
    return os.path.join("configs", name), name, name + "/perf_all.csv"


def entry_out(name, out_root):
    """Absolute per-entry csv path under ``out_root``."""
    _, _, out_rel = resolve_entry(name)
    return os.path.join(out_root, out_rel)


# hrt tools are the uploaded runtime pack, one copy per board under
# /userdata/heal_perf (202609 boards: M=20260422, P=20260821, B=20260909,
# all release builds; M/B/P share the same aarch64 pack layout). J6M/J6P
# auth by installed pubkey (password None); J6B is QNX with an empty
# password (RSA1024 host key, plain ssh cannot connect).
NASH_PACK = "/userdata/heal_perf/output_shared_J6_aarch64"
BOARDS = {
    "nash-m": {
        "ip": "10.103.43.138",
        "hrt_bin": NASH_PACK + "/aarch64/bin",
        "hrt_lib": NASH_PACK + "/aarch64/lib:/usr/hobot/lib",
        "board_dir": "/userdata/heal_perf/heal_hbm",
        "result_dir": "/userdata/heal_perf/heal_result",
        "monitor_bin": NASH_PACK + "/aarch64/hrt_ucp_monitor",
        "password": None,
    },
    "nash-p": {
        "ip": "10.103.122.126",
        "hrt_bin": NASH_PACK + "/aarch64/bin",
        "hrt_lib": NASH_PACK + "/aarch64/lib:/usr/hobot/lib",
        "board_dir": "/userdata/heal_perf/heal_hbm",
        "result_dir": "/userdata/heal_perf/heal_result",
        "monitor_bin": NASH_PACK + "/aarch64/hrt_ucp_monitor",
        "password": None,
    },
    "nash-b": {
        "ip": "10.103.42.214",
        "hrt_bin": NASH_PACK + "/aarch64/bin",
        "hrt_lib": NASH_PACK + "/aarch64/lib:/usr/hobot/lib",
        "board_dir": "/userdata/heal_perf/heal_hbm",
        "result_dir": "/userdata/heal_perf/heal_result",
        "monitor_bin": NASH_PACK + "/aarch64/hrt_ucp_monitor",
        "password": "",
    },
}

MARCH_ORDER = ["nash-m", "nash-p", "nash-b"]

# FPS-thread overrides for (board, model) combos that OOM at the default
# 8 threads on the small-memory J6B: fewer threads = fewer stacked BPU
# instances. Not in this table -> the default 8 (or --fps-thread-num).
FPS_THREAD_OVERRIDES = {
    ("nash-b", "sparse_multitask_det_maptr_flashocc_henet_tinym"): 2,
    ("nash-b", "maptr_sparse_henet_tinym"): 4,
}


def _board_version(creds, bin_path, lib_path=None):
    """SSH ``<bin> --version`` and return the matched ``x.y.z`` or '?'."""
    export = ""
    if lib_path:
        export = "export LD_LIBRARY_PATH=%s:$LD_LIBRARY_PATH; " % lib_path
    cmd = "%s%s --version 2>&1" % (export, bin_path)
    try:
        out = ssh_board(creds, cmd)
    except Exception as e:  # noqa: B902
        logger.warning("version query failed %s: %s", bin_path, e)
        return "?"
    m = re.search(r"(\d+\.\d+\.\d+)", out or "")
    return m.group(1) if m else "?"


def collect_versions(marches):
    """Query hrt_model_exec + hrt_ucp_monitor version per board.

    Returns {march: (exec_ver, monitor_ver)}.
    """
    versions = {}
    for march in marches:
        b = BOARDS[march]
        lib = b["hrt_lib"].split(":")[0]
        exec_v = _board_version(
            b,
            posixpath.join(b["hrt_bin"], "hrt_model_exec"),
            lib,
        )
        mon_v = _board_version(b, b["monitor_bin"], lib)
        versions[march] = (exec_v, mon_v)
        logger.info(
            "[%s] hrt_model_exec=%s hrt_ucp_monitor=%s",
            march,
            exec_v,
            mon_v,
        )
    return versions


def write_summary(entries, marches, out_root, versions):
    """Collapse the per-entry csvs into <out_root>/summary_all.csv.

    Leading ``#``-prefixed lines record the board hrt tool versions so a
    reader can confirm the runtime stack at a glance. Last row per
    (task, march) wins (reruns do not stack duplicates).
    """
    rows = {}
    for path in glob.glob(
        os.path.join(out_root, "**", "perf_all.csv"), recursive=True
    ):
        with open(path) as f:
            for r in csv.DictReader(f):
                rows[(r["task"], r["march"])] = r
    out = os.path.join(out_root, "summary_all.csv")
    os.makedirs(out_root, exist_ok=True)
    keys = sorted(rows, key=lambda k: (k[0], MARCH_ORDER.index(k[1])))
    with open(out, "w") as f:
        for march in marches:
            exec_v, mon_v = versions.get(march, ("?", "?"))
            f.write(
                "# %s: hrt_model_exec=%s hrt_ucp_monitor=%s\n"
                % (march, exec_v, mon_v)
            )
        f.write(SUMMARY_HEADER + "\n")
        for k in keys:
            f.write(
                ",".join(rows[k][c] for c in SUMMARY_HEADER.split(",")) + "\n"
            )
    logger.info("summary: %s (%d rows)", out, len(keys))
    return rows


def parse_args():
    p = argparse.ArgumentParser(
        description="Batch measure all HEAL models on J6M/J6P/J6B."
    )
    p.add_argument(
        "--measure",
        default="all",
        choices=["fps_latency", "memory", "all"],
        help="Passed to test_perf_all.py. all = fps+latency+DDR.",
    )
    p.add_argument("--perf-time", default="1", help="perf_time per case.")
    p.add_argument(
        "--fps-thread-num",
        type=int,
        default=None,
        help="Override the FPS-case thread count (default 8). Lower for "
        "board/model combos that OOM at 8 threads, e.g. J6B heavy -> 4.",
    )
    p.add_argument(
        "--only",
        default=None,
        help="Comma-separated subset of models to run (default: all 24).",
    )
    p.add_argument(
        "--boards",
        default=None,
        help="Comma-separated subset of marches (default: all 3).",
    )
    p.add_argument(
        "--skip-compile",
        action="store_true",
        help="Reuse existing local model_<march>.hbm (passed through).",
    )
    p.add_argument(
        "--no-board",
        action="store_true",
        help="Compile only, do not touch any board (smoke the build).",
    )
    p.add_argument(
        "--no-clean-board-hbm",
        action="store_true",
        help="Do NOT delete the board-side hbm after each model "
        "(default: delete it to save board space).",
    )
    p.add_argument(
        "--out-root",
        default="perf_results",
        help="Root dir for per-entry csvs and summary_all.csv.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be run, do not execute.",
    )
    return p.parse_args()


def ssh_board(creds, cmd):
    """Run a command on a board via paramiko (mirrors hbm_infer creds)."""
    from hbm_infer.utils import remote_execute

    return remote_execute(
        cmd, creds["ip"], "root", creds["password"], 22, max_attempts=3
    )


def clean_board_hbm(creds, board_dir, task, march):
    """Delete the uploaded hbm for this (task, march) to free board space.

    Result logs under <result_dir>/<task>_<march>/ are left in place per the
    user (not cleaned here).
    """
    remote_dir = posixpath.join(board_dir, task, march)
    try:
        out = ssh_board(
            creds, "rm -f %s/*.hbm 2>/dev/null; echo cleaned" % remote_dir
        )
        logger.info(
            "[%s/%s] board hbm cleaned: %s",
            task,
            march,
            (out or "").strip(),
        )
    except Exception as e:  # noqa: B902
        logger.warning("[%s/%s] board hbm clean failed: %s", task, march, e)


def build_inner_cmd(cfg, march, b, args, out_root, out=None, extra=None):
    """Assemble the test_perf_all.py invocation for one (model, board)."""
    config, _, _ = resolve_entry(cfg)
    out = out or entry_out(cfg, out_root)
    # Per-(board, model) FPS-thread override; an explicit --fps-thread-num
    # from the caller wins over the table.
    override = FPS_THREAD_OVERRIDES.get((march, cfg))
    if override is not None and args.fps_thread_num is None:
        args = argparse.Namespace(**{**vars(args), "fps_thread_num": override})
        logger.info("[%s/%s] fps thread override -> %d", cfg, march, override)
    pw = b["password"]
    # --password None cannot be expressed on a single shared line for J6B;
    # omit the flag entirely when the board is key-less (default None), and
    # pass an explicit empty string for the empty-password board.
    if pw is None:
        pw_args = []
    else:
        pw_args = ["--password", pw]
    cmd = [
        sys.executable,
        INNER,
        "-c",
        config,
        "--march",
        march,
        "--measure",
        args.measure,
        "--perf-time",
        args.perf_time,
        "--board",
        b["ip"],
        "--hrt-bin",
        b["hrt_bin"],
        "--hrt-lib",
        b["hrt_lib"],
        "--board-dir",
        b["board_dir"],
        "--result-dir",
        b["result_dir"],
        "--monitor-bin",
        b["monitor_bin"],
    ] + pw_args
    if out:
        cmd += ["--out", out]
    if args.fps_thread_num is not None:
        cmd += ["--fps-thread-num", str(args.fps_thread_num)]
    if args.skip_compile:
        cmd.append("--skip-compile")
    if args.no_board:
        cmd.append("--no-board")
    # last: a trailing --opts in extra is argparse.REMAINDER and eats the rest.
    if extra:
        cmd += extra
    return cmd


def run_one(cfg, march, b, args, out_root, out=None, extra=None):
    """Compile + measure one model on one board; then clean board hbm."""
    cmd = build_inner_cmd(cfg, march, b, args, out_root, out=out, extra=extra)
    pretty = " ".join(shlex.quote(c) for c in cmd)
    logger.info("=" * 28 + " %s / %s " % (cfg, march) + "=" * 28)
    logger.info("CMD: %s", pretty)
    if args.dry_run:
        return 0
    t0 = time.monotonic()
    rc = subprocess.call(
        cmd, cwd=REPO_ROOT, env={**os.environ, "PYTHONPATH": "."}
    )
    dt = time.monotonic() - t0
    logger.info("[%s/%s] inner rc=%d (%.0fs)", cfg, march, rc, dt)
    if not args.no_board and not args.no_clean_board_hbm:
        _, task, _ = resolve_entry(cfg)
        clean_board_hbm(b, b["board_dir"], task, march)
    return rc


def main():
    logging.basicConfig(
        format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO
    )
    args = parse_args()

    models = args.only.split(",") if args.only else MODELS
    models = [m.strip() for m in models if m.strip()]
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        raise SystemExit("unknown model(s): %s (known: see MODELS)" % unknown)

    marches = args.boards.split(",") if args.boards else MARCH_ORDER
    marches = [m.strip() for m in marches if m.strip()]
    bad = [m for m in marches if m not in BOARDS]
    if bad:
        raise SystemExit(
            "unknown board(s): %s (known: %s)" % (bad, list(BOARDS))
        )

    # Order: outer = model, inner = board. So all boards of model N finish
    # before model N+1 starts, and a broken model does not strand a board's
    # later models behind a long compile of the next one. Local hbm is kept
    # across boards (same march family compiled once per march, reused via
    # --skip-compile only if the user asks).
    out_root = os.path.join(REPO_ROOT, args.out_root)
    versions = {} if args.no_board else collect_versions(marches)
    total = len(models) * len(marches)
    done = 0
    fails = []
    for cfg in models:
        for march in marches:
            done += 1
            logger.info(
                "########## [%d/%d] %s :: %s ##########",
                done,
                total,
                cfg,
                march,
            )
            b = BOARDS[march]
            try:
                rc = run_one(cfg, march, b, args, out_root)
                if rc != 0:
                    fails.append((cfg, march, rc))
            except Exception as e:  # noqa: B902
                logger.error("[%s/%s] outer error: %s", cfg, march, e)
                fails.append((cfg, march, str(e)))

    rows = write_summary(models, marches, out_root, versions)
    bad_rows = [k for k, r in rows.items() if r["status"] != "ok"]
    logger.info("=" * 60)
    logger.info(
        "DONE: %d/%d ok, %d failed, %d non-ok rows",
        total - len(fails),
        total,
        len(fails),
        len(bad_rows),
    )
    for cfg, march, why in fails:
        logger.info("  FAIL %s / %s : %s", cfg, march, why)
    for k in bad_rows:
        logger.info("  NOT-OK %s / %s : %s", k[0], k[1], rows[k]["status"])


if __name__ == "__main__":
    main()
