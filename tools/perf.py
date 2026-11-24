# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
import argparse
import json
import logging
import os
import posixpath
import re
import traceback

logger = logging.getLogger("perf")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHIP_TO_MARCH = {"j6b": "nash-b", "j6m": "nash-m", "j6p": "nash-p"}

# Board defaults (2026-09 930 boards, see tools/test_perf_all_25.py): J6M/J6P
# auth by installed pubkey; J6B is QNX with an empty password.
NASH_PACK = "/userdata/heal_perf/output_shared_J6_aarch64"
BOARDS = {
    "j6b": {
        "ip": "10.103.42.214",
        "hrt_bin": NASH_PACK + "/aarch64/bin",
        "hrt_lib": NASH_PACK + "/aarch64/lib:/usr/hobot/lib",
        "board_dir": "/userdata/heal_perf/heal_hbm",
        "result_dir": "/userdata/heal_perf/heal_result",
        "monitor_bin": NASH_PACK + "/aarch64/hrt_ucp_monitor",
        "password": "",
    },
    "j6m": {
        "ip": "10.103.43.138",
        "hrt_bin": NASH_PACK + "/aarch64/bin",
        "hrt_lib": NASH_PACK + "/aarch64/lib:/usr/hobot/lib",
        "board_dir": "/userdata/heal_perf/heal_hbm",
        "result_dir": "/userdata/heal_perf/heal_result",
        "monitor_bin": NASH_PACK + "/aarch64/hrt_ucp_monitor",
        "password": None,
    },
    "j6p": {
        "ip": "10.103.122.126",
        "hrt_bin": NASH_PACK + "/aarch64/bin",
        "hrt_lib": NASH_PACK + "/aarch64/lib:/usr/hobot/lib",
        "board_dir": "/userdata/heal_perf/heal_hbm",
        "result_dir": "/userdata/heal_perf/heal_result",
        "monitor_bin": NASH_PACK + "/aarch64/hrt_ucp_monitor",
        "password": None,
    },
}

CASES = {
    "fps": {"thread_num": 8, "core_id": 0},
    "latency": {"thread_num": 1, "core_id": 1},
}

_UNIT_TO_MIB = {"K": 1.0 / 1024.0, "M": 1.0, "G": 1024.0}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure board-side FPS / latency / peak DDR of a "
        "precompiled hbm (no export/compile). Example:\n"
        "  PYTHONPATH=. python3 tools/perf.py "
        "--hbm tmp_models/efficientnet/compile/model_nash-m.hbm --chip j6m",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Optional config file or directory; deploy.py is read for the "
        "task name (lower priority than --task, higher than the hbm stem).",
    )
    parser.add_argument(
        "--hbm",
        type=str,
        required=True,
        help="Local path of the compiled .hbm to upload and measure.",
    )
    parser.add_argument(
        "--chip",
        type=str,
        required=True,
        help="Comma-separated chips, one or more of j6b,j6m,j6p. The chip "
        "selects the march (for naming) and the default board settings.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task name for board dirs and csv. Defaults to task_name / "
        "hbir_exporter.model_name of -c (if given), else the hbm file stem "
        "with a trailing '_<march>' stripped (model_nash-m.hbm -> model).",
    )
    parser.add_argument(
        "--board",
        type=str,
        default=None,
        help="Board ip override (applies to every selected chip).",
    )
    parser.add_argument("--username", type=str, default="root")
    parser.add_argument("--password", type=str, default=None)
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument(
        "--measure",
        type=str,
        default="all",
        choices=["fps_latency", "memory", "all"],
        help="fps_latency: fps (multi-thread) + latency (single-thread), "
        "no DDR. memory: latency + peak DDR (single-thread under "
        "hrt_ucp_monitor). all: fps run plus the memory run.",
    )
    parser.add_argument(
        "--perf-time",
        type=str,
        default="1",
        help="perf_time for hrt_model_exec (seconds per case).",
    )
    parser.add_argument(
        "--fps-thread-num",
        type=int,
        default=None,
        help="Override the FPS-case thread count (default 8). Lower it for "
        "boards/models that OOM at 8 threads, e.g. j6b heavy models -> 4.",
    )
    parser.add_argument(
        "--hrt-bin",
        type=str,
        default=None,
        help="Directory holding hrt_model_exec on the board.",
    )
    parser.add_argument(
        "--hrt-lib",
        type=str,
        default=None,
        help="Colon-separated LD_LIBRARY_PATH for the board. "
        "Defaults to <hrt-bin>/lib.",
    )
    parser.add_argument(
        "--board-dir",
        type=str,
        default=None,
        help="Board-side hbm root: hbm goes to <root>/<task>/<march>/.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Board-side result root: profiler.log/monitor.log go to "
        "<root>/<task>_<march>/.",
    )
    parser.add_argument(
        "--monitor-bin",
        type=str,
        default=None,
        help="Path to hrt_ucp_monitor on the board.",
    )
    parser.add_argument(
        "--monitor-time",
        type=str,
        default=None,
        help="Sampling window for hrt_ucp_monitor, seconds. Defaults to "
        "perf_time + 2.5.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Summary csv path. " "Defaults to ./perf_results/<task>/perf.csv",
    )
    return parser.parse_args()


def parse_profiler_log(text):
    """Parse ``profiler.log`` JSON into fps / latency / bpu / ucp."""
    res = {"fps": None, "latency": None, "bpu": None, "ucp": None}
    if not text:
        return res
    blocks = []
    for chunk in text.split("***"):
        brace = chunk.find("{")
        if brace < 0:
            continue
        try:
            blocks.append(json.loads(chunk[brace:]))
        except json.JSONDecodeError:
            continue
    if not blocks:
        return res
    perf = blocks[0].get("perf_result", {})
    res["fps"] = perf.get("FPS")
    res["latency"] = perf.get("average_latency")
    if len(blocks) > 1:
        proc = blocks[1].get("processor_latency", {})
        b = proc.get("BPU_inference_time_cost", {})
        c = proc.get("CPU_inference_time_cost", {})
        t = blocks[1].get("task_latency", {}).get("TaskRunningTime", {})
        res["bpu"] = b.get("avg_time")
        task = t.get("avg_time")
        bpu, cpu = b.get("avg_time"), c.get("avg_time")
        if task is not None and bpu is not None and cpu is not None:
            res["ucp"] = max(task - bpu - cpu, 0.0)
    return res


def parse_monitor_mem(text, proc_name="hrt_model_exec"):
    """Return peak HBMEM (MiB) of ``proc_name`` over the monitor log."""
    if not text:
        return None
    pat = re.compile(
        r"^\|\s*\d+\s+" + re.escape(proc_name) + r"\s+([0-9.]+)([KMG]?)\s+"
        r"([0-9.]+)([KMG]?)\s*\|"
    )
    peak = None
    for line in text.splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        val, unit = float(m.group(3)), m.group(4)
        mib = val * _UNIT_TO_MIB.get(unit, 1.0)
        if peak is None or mib > peak:
            peak = mib
    return peak


def _ssh(host, args, password):
    """Return a (exec, upload) pair bound to the board creds."""
    from hbm_infer.utils import remote_execute, remote_upload

    creds = (host, args.username, password, args.ssh_port)

    def exec_cmd(cmd, attempts=5):
        return remote_execute(cmd, *creds, max_attempts=attempts)

    def upload(local, remote):
        remote_upload(local, remote, *creds)

    return exec_cmd, upload


def _upload_hbm(exec_cmd, upload, hbm_path, board_dir, task, march, host):
    """Scp the hbm to <board_dir>/<task>/<march>/. Returns remote path."""
    remote_dir = posixpath.join(board_dir, task, march)
    exec_cmd("mkdir -p %s" % remote_dir)
    remote_hbm = posixpath.join(remote_dir, "%s_%s.hbm" % (task, march))
    logger.info("[%s] upload -> %s:%s", march, host, remote_hbm)
    upload(hbm_path, remote_hbm)
    return remote_hbm


def _run_case(
    exec_cmd,
    case,
    hbm,
    hrt_bin,
    hrt_lib,
    perf_time,
    prof_dir,
    monitor_bin=None,
    monitor_time=None,
    monitor_log=None,
    march="",
):
    """Run one hrt_model_exec perf case on the board."""
    c = CASES[case]
    mkdir = "mkdir -p %s" % prof_dir
    # Absolute binary path: a `cd ... && nohup ... &` chain backgrounds the
    # whole `&`-left side in a subshell, so the foreground command after `&`
    # would run in the original CWD and miss ./hrt_model_exec.
    hrt_cmd = (
        "%s perf --model_file=%s --thread_num=%d --core_id=%d --perf_time=%s "
        "--profile_path=%s"
        % (
            posixpath.join(hrt_bin, "hrt_model_exec"),
            hbm,
            c["thread_num"],
            c["core_id"],
            perf_time,
            prof_dir,
        )
    )
    hrt_log = posixpath.join(prof_dir, "%s.log" % case)
    if monitor_bin:
        cmd = (
            "export LD_LIBRARY_PATH=%s:$LD_LIBRARY_PATH; %s; "
            "if [ ! -f %s ]; then echo NO_MONITOR; fi; "
            "nohup %s -b -t %s -d 200 > %s 2>&1 & MPID=$!; sleep 0.8; "
            "%s > %s 2>&1; wait $MPID 2>/dev/null; echo PERF_DONE"
        ) % (
            hrt_lib,
            mkdir,
            monitor_bin,
            monitor_bin,
            monitor_time,
            monitor_log,
            hrt_cmd,
            hrt_log,
        )
    else:
        cmd = (
            "export LD_LIBRARY_PATH=%s:$LD_LIBRARY_PATH; %s; "
            "%s > %s 2>&1; echo PERF_DONE"
        ) % (hrt_lib, mkdir, hrt_cmd, hrt_log)
    logger.info("[%s] board: %s", march or case, cmd)
    # Single attempt: a retry costs another full perf run.
    return exec_cmd(cmd, attempts=1)


def run_hrt_perf(exec_cmd, case, hbm, brd, perf_time, march):
    """Run a no-monitor perf case and parse profiler.log -> metrics dict."""
    prof_dir = posixpath.join(brd["result_dir_full"], case)
    hrt_log = posixpath.join(prof_dir, "%s.log" % case)
    _run_case(
        exec_cmd,
        case,
        hbm,
        brd["hrt_bin"],
        brd["hrt_lib"],
        perf_time,
        prof_dir,
        march=march,
    )
    prof_txt = exec_cmd(
        "cat %s 2>/dev/null || true" % posixpath.join(prof_dir, "profiler.log")
    )
    metrics = parse_profiler_log(prof_txt)
    if metrics["fps"] is None and metrics["latency"] is None:
        logger.warning(
            "[%s] %s: profiler.log empty. hrt log tail:\n%s",
            march,
            case,
            exec_cmd("tail -n 20 %s 2>/dev/null || true" % hrt_log),
        )
    return metrics


def run_board_memory(exec_cmd, hbm, brd, perf_time, monitor_time, march):
    """Single-threaded perf under monitor sampling -> latency + peak DDR."""
    prof_dir = posixpath.join(brd["result_dir_full"], "memory")
    hrt_log = posixpath.join(prof_dir, "latency.log")
    monitor_log = posixpath.join(prof_dir, "monitor.log")
    out = _run_case(
        exec_cmd,
        "latency",
        hbm,
        brd["hrt_bin"],
        brd["hrt_lib"],
        perf_time,
        prof_dir,
        monitor_bin=brd["monitor_bin"],
        monitor_time=monitor_time,
        monitor_log=monitor_log,
        march=march,
    )
    if "NO_MONITOR" in (out or ""):
        logger.warning(
            "[%s] monitor %s not found on board; DDR skipped",
            march,
            brd["monitor_bin"],
        )
    prof_txt = exec_cmd(
        "cat %s 2>/dev/null || true" % posixpath.join(prof_dir, "profiler.log")
    )
    metrics = parse_profiler_log(prof_txt)
    mon_txt = exec_cmd("cat %s 2>/dev/null || true" % monitor_log)
    peak = parse_monitor_mem(mon_txt)
    if metrics["latency"] is None:
        logger.warning(
            "[%s] memory: profiler.log empty. hrt log tail:\n%s",
            march,
            exec_cmd("tail -n 20 %s 2>/dev/null || true" % hrt_log),
        )
    if peak is None and "NO_MONITOR" not in (out or ""):
        logger.warning(
            "[%s] memory: no hrt_model_exec row in monitor log. tail:\n%s",
            march,
            exec_cmd("tail -n 12 %s 2>/dev/null || true" % monitor_log),
        )
    return {
        "latency": metrics["latency"],
        "bpu": metrics["bpu"],
        "ucp": metrics["ucp"],
        "peak_ddr_mib": peak,
    }


def resolve_task(config, hbm_path):
    """Task name: deploy.py of ``config`` if given, else the hbm file stem."""
    if config is not None:
        import heal  # noqa: F401
        from heal.utils.config import (
            load_config_with_optional_base,
            resolve_config_file,
        )

        cfg = load_config_with_optional_base(
            resolve_config_file(config, default_filename="deploy.py")
        )
        return cfg.get("task_name", None) or cfg.hbir_exporter.get(
            "model_name", "model"
        )
    stem = os.path.splitext(os.path.basename(hbm_path))[0]
    for march in ("nash-b", "nash-m", "nash-p"):
        suffix = "_" + march
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def resolve_board(chip, args):
    """Board settings for one chip: BOARDS defaults, CLI flags override."""
    d = dict(BOARDS[chip])
    for key in (
        "hrt_bin",
        "hrt_lib",
        "board_dir",
        "result_dir",
        "monitor_bin",
    ):
        v = getattr(args, key)
        if v is not None:
            d[key] = v
    if args.board is not None:
        d["ip"] = args.board
    if args.password is not None:
        d["password"] = args.password
    if d["hrt_lib"] is None:
        d["hrt_lib"] = posixpath.join(d["hrt_bin"], "lib")
    return d


def run_one_chip(chip, hbm_path, task, args, brd):
    """Upload the hbm and measure it on one board."""
    march = CHIP_TO_MARCH[chip]
    row = {"chip": chip, "march": march, "task": task, "status": "ok"}
    exec_cmd, upload = _ssh(brd["ip"], args, brd["password"])
    remote_hbm = _upload_hbm(
        exec_cmd, upload, hbm_path, brd["board_dir"], task, march, brd["ip"]
    )
    brd["result_dir_full"] = posixpath.join(
        brd["result_dir"], "%s_%s" % (task, march)
    )
    exec_cmd("mkdir -p %s" % brd["result_dir_full"])

    monitor_time = args.monitor_time or str(int(float(args.perf_time) + 2.5))
    if args.measure == "fps_latency":
        fps = run_hrt_perf(
            exec_cmd, "fps", remote_hbm, brd, args.perf_time, march
        )
        lat = run_hrt_perf(
            exec_cmd, "latency", remote_hbm, brd, args.perf_time, march
        )
        row.update(
            fps=fps["fps"],
            latency=lat["latency"],
            bpu=lat["bpu"],
            ucp=lat["ucp"],
            peak_ddr_mib=None,
        )
    elif args.measure == "memory":
        mem = run_board_memory(
            exec_cmd, remote_hbm, brd, args.perf_time, monitor_time, march
        )
        row.update(
            fps=None,
            latency=mem["latency"],
            bpu=mem["bpu"],
            ucp=mem["ucp"],
            peak_ddr_mib=mem["peak_ddr_mib"],
        )
    else:  # all
        fps = run_hrt_perf(
            exec_cmd, "fps", remote_hbm, brd, args.perf_time, march
        )
        mem = run_board_memory(
            exec_cmd, remote_hbm, brd, args.perf_time, monitor_time, march
        )
        row.update(
            fps=fps["fps"],
            latency=mem["latency"],
            bpu=mem["bpu"],
            ucp=mem["ucp"],
            peak_ddr_mib=mem["peak_ddr_mib"],
        )
    return row


def fmt(value):
    """Render a measurement, keeping '-' for anything that failed to parse."""
    return "-" if value is None else "%.2f" % value


def print_table(rows):
    head = (
        "chip",
        "FPS",
        "latency(ms)",
        "bpu(ms)",
        "ucp(ms)",
        "DDR(MiB)",
        "status",
    )
    widths = [max(len(h), 12) for h in head]
    line = "  ".join(h.ljust(w) for h, w in zip(head, widths))
    print("\n" + line)
    print("-" * len(line))
    for r in rows:
        cells = (
            r["chip"],
            fmt(r.get("fps")),
            fmt(r.get("latency")),
            fmt(r.get("bpu")),
            fmt(r.get("ucp")),
            fmt(r.get("peak_ddr_mib")),
            r.get("status", ""),
        )
        print("  ".join(str(c).ljust(w) for c, w in zip(cells, widths)))
    print()


def write_out(rows, args, task):
    if not rows:
        return
    out = args.out or os.path.join(REPO_ROOT, "perf_results", task, "perf.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    exists = os.path.exists(out)
    with open(out, "a") as f:
        if not exists:
            f.write(
                "task,chip,march,fps,latency_ms,bpu_ms,ucp_ms,"
                "peak_ddr_mib,status\n"
            )
        for r in rows:
            f.write(
                "%s,%s,%s,%s,%s,%s,%s,%s,%s\n"
                % (
                    r.get("task", task),
                    r["chip"],
                    r["march"],
                    fmt(r.get("fps")),
                    fmt(r.get("latency")),
                    fmt(r.get("bpu")),
                    fmt(r.get("ucp")),
                    fmt(r.get("peak_ddr_mib")),
                    r.get("status", ""),
                )
            )
    logger.info("summary written to %s", out)


def main():
    args = parse_args()
    if args.fps_thread_num is not None:
        CASES["fps"]["thread_num"] = args.fps_thread_num
    if not os.path.exists(args.hbm):
        raise FileNotFoundError("--hbm %s does not exist" % args.hbm)
    hbm_path = os.path.abspath(args.hbm)
    task = args.task or resolve_task(args.config, hbm_path)

    chips = [c.strip() for c in args.chip.split(",") if c.strip()]
    bad = [c for c in chips if c not in CHIP_TO_MARCH]
    if bad:
        raise SystemExit(
            "unknown chip(s): %s (known: %s)" % (bad, list(CHIP_TO_MARCH))
        )

    rows = []
    for chip in chips:
        logger.info("=" * 30 + " chip %s " % chip + "=" * 30)
        try:
            rows.append(
                run_one_chip(
                    chip, hbm_path, task, args, resolve_board(chip, args)
                )
            )
        except Exception:  # noqa: B902
            logger.error("[%s] failed:\n%s", chip, traceback.format_exc())
            rows.append(
                {
                    "chip": chip,
                    "march": CHIP_TO_MARCH[chip],
                    "task": task,
                    "status": "ERROR",
                }
            )
    print_table(rows)
    write_out(rows, args, task)
    return rows


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)-15s %(levelname)s %(message)s",
        level=logging.INFO,
    )
    main()
