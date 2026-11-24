# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
r"""One-key board-side FPS / latency / peak-DDR measurement for a HEAL model.

Unlike ``tools/test_perf.py`` (which drives the board through the external
``test_perf_oe_qat.sh``), this script talks to ``hrt_model_exec`` directly over
ssh and parses its ``profiler.log`` JSON -- no board-side sh is needed. Peak
DDR is measured with ``hrt_ucp_monitor`` (compile-time ``hbm_perf`` DDR is
meaningless per the user): it is sampled during the *latency* run only, since
latency is single-threaded (``thread_num=1``) and single-threaded DDR is the
per-instance peak (multi-threaded fps would stack several instances and
overstate it). So one single-threaded perf run yields latency + peak DDR; fps
gets its own multi-threaded run.

For every requested march this script

  1. loads ``deploy.py`` and overrides ``hbir_exporter.march`` /
     ``hbir_compiler.march`` **in memory** -- ``common.py`` is never touched;
  2. exports the hbir and compiles a march-specific ``model_<march>.hbm``
     (every config emits ``model.hbm``, and the model name is derived from
     ``basename $hbm .hbm``, so they would collide);
  3. uploads the hbm, then -- depending on ``--measure`` -- runs on the board:
       - fps_latency: multi-threaded fps run + single-threaded latency run,
         both via ``hrt_model_exec perf``, parsing ``profiler.log``;
       - memory: one single-threaded run under ``hrt_ucp_monitor`` sampling ->
         latency (from profiler.log) + peak DDR (max HBMEM of the run);
       - all: the fps run plus the memory run (hbm uploaded once).

Different chips are different physical boards, and boards differ in more
than ip: the hrt location, which partition has free space (e.g. J6M's /map
is full -> results go to /userdata; J6P's /userdata is full -> results go to
/map), and where hrt_ucp_monitor lives. So every board-facing option takes
``march@value`` (comma-separated, mixable with a bare value shared by all
marches); a march without an explicit ``march@`` entry falls back to the bare
value, then the option default.

Run from the HEAL root with PYTHONPATH set (use the env that has hbm_infer).
The hrt tools are the uploaded runtime pack (hrt_model_exec 3.15.8); boards
below are the 2026-09 930 boards (pack at /userdata/heal_perf, J6B is QNX
with an empty password):

    PKG=/userdata/heal_perf/output_shared_J6_aarch64/aarch64
    PYTHONPATH=. python3 tools/test_perf_all.py \\
        -c configs/efficientnet --march nash-m --skip-compile \\
        --measure all --board 10.103.43.138 \\
        --hrt-bin $PKG/bin --hrt-lib $PKG/lib:/usr/hobot/lib \\
        --board-dir  /userdata/heal_perf/heal_hbm \\
        --result-dir /userdata/heal_perf/heal_result \\
        --monitor-bin $PKG/hrt_ucp_monitor

    # compile only (no board needed), keep the per-march compile reports
    PYTHONPATH=. python3 tools/test_perf_all.py \\
        -c configs/efficientnet --march nash-m --no-board --save-json
"""

import argparse
import ast
import json
import logging
import multiprocessing
import os
import posixpath
import re
import shutil
import traceback

import horizon_plugin_pytorch as horizon  # noqa: F401
from hat_engine.registry import RegistryContext, build_from_registry
from hat_engine.utils.setup_env import setup_args_env
from horizon_plugin_pytorch.march import March

import heal  # noqa: F401
from heal.utils.config import (
    load_config_with_optional_base,
    resolve_config_file,
)

logger = logging.getLogger("test_perf_all")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VALID_MARCHES = sorted(
    getattr(March, _a)
    for _a in dir(March)
    if not _a.startswith("_") and isinstance(getattr(March, _a), str)
)

# Where hrt_ucp_monitor lives per march (boards differ chip-to-chip).
DEFAULT_MONITOR_BIN = {
    "nash-m": "/userdata/heal_perf_test/hrt_ucp_monitor",
    "nash-p": "/map/heal_perf_test/hrt_ucp_monitor",
}

# hrt_model_exec perf cases -- thread/core chosen to match test_perf_oe_qat.sh.
CASES = {
    "fps": {"thread_num": 8, "core_id": 0},
    "latency": {"thread_num": 1, "core_id": 1},
}

_UNIT_TO_MIB = {"K": 1.0 / 1024.0, "M": 1.0, "G": 1024.0}


def parse_args():
    """Parse CLI arguments and passthrough env options."""
    parser = argparse.ArgumentParser(
        description="Measure board-side FPS / latency / peak DDR of a HEAL "
        "model without relying on a board-side sh."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Config file path, or config directory (uses deploy.py).",
    )
    parser.add_argument(
        "--march",
        type=str,
        default=None,
        help="Comma separated marches, e.g. 'nash-e,nash-m'. "
        "Defaults to hbir_compiler.march of the config.",
    )
    parser.add_argument(
        "--board",
        type=str,
        default=None,
        help="Board ip, or a per-march map since different chips live on "
        "different boards: 'nash-m@10.103.43.99,nash-p@10.64.61.25'. "
        "A bare ip is shared by every march. Defaults to "
        "hbm_model.ir_model.host of predict.py (only the default march).",
    )
    parser.add_argument("--username", type=str, default="root")
    parser.add_argument("--password", type=str, default=None)
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument(
        "--no-board",
        action="store_true",
        help="Only export and compile, skip the board measurement.",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Reuse an existing compile/model_<march>.hbm.",
    )
    parser.add_argument(
        "--measure",
        type=str,
        default="fps_latency",
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
        "boards/models that OOM at 8 threads, e.g. J6B heavy models -> 4.",
    )
    parser.add_argument(
        "--hrt-bin",
        type=str,
        default="/userdata/.horizon",
        help="Directory holding hrt_model_exec on the board (cd into it "
        "then run ./hrt_model_exec). Accepts 'march@dir'.",
    )
    parser.add_argument(
        "--hrt-lib",
        type=str,
        default=None,
        help="Colon-separated LD_LIBRARY_PATH for the board, e.g. "
        "'/map/lib:/usr/hobot/lib'. Defaults to <hrt-bin>/lib. Accepts "
        "'march@libs'.",
    )
    parser.add_argument(
        "--board-dir",
        type=str,
        default="/userdata/heal_hbm",
        help="Board-side hbm root: hbm goes to <root>/<task>/<march>/. "
        "Accepts 'march@root'.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="/userdata/.horizon/heal_perf",
        help="Board-side result root: profiler.log/monitor.log go to "
        "<root>/<task>_<march>/. Accepts 'march@root'.",
    )
    parser.add_argument(
        "--monitor-bin",
        type=str,
        default=None,
        help="Path to hrt_ucp_monitor on the board. Accepts 'march@path'. "
        "Defaults to a per-march guess (nash-m -> /userdata/heal_perf_test, "
        "nash-p -> /map/heal_perf_test).",
    )
    parser.add_argument(
        "--monitor-time",
        type=str,
        default=None,
        help="Sampling window for hrt_ucp_monitor, seconds. Defaults to "
        "perf_time + 3.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Summary csv path. "
        "Defaults to ./perf_results/<task>/perf_all.csv",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Copy the per-march compile report to perf_results/<task>/ "
        "as <task>_<march>.json/.html.",
    )
    parser.add_argument(
        "--opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--opts-overwrite",
        type=ast.literal_eval,
        default=True,
        help="True or False, default True, "
        "Weather to overwrite existing (keys, values) in configs",
    )
    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args


# ----- config / compile (reused from test_perf.py, unchanged in spirit) -----


def config_dir_of(config):
    """Return the directory holding common.py/model.py for a config arg."""
    return config if os.path.isdir(config) else os.path.dirname(config)


def parse_march_map(value, label="value"):
    """Parse a per-march option into ``{march_or_None: value}``.

    Boards differ chip-to-chip (hrt location, result dir, ip), so every
    board-facing option takes the same form, comma-separated and mixable:

      ``/map``                       -- one value shared by every march
      ``nash-m@/map``                 -- value for one march only
      ``nash-m@10.103.43.99,nash-p@10.64.61.25``
                                     -- different chips need different values

    ``None`` key holds a bare value applied to any march without an explicit
    ``march@`` entry.
    """
    result = {}
    if not value:
        return result
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "@" in item:
            m, v = item.split("@", 1)
            result[m.strip()] = v.strip()
        else:
            result[None] = item
    return result


def pick(march_map, march, default):
    """Resolve a per-march option: march@ entry -> bare entry -> default."""
    return march_map.get(march) or march_map.get(None) or default


def resolve_board_host(config):
    """Read hbm_model.ir_model.host from the config's predict.py."""
    try:
        cfg = load_config_with_optional_base(
            resolve_config_file(
                config_dir_of(config), default_filename="predict.py"
            )
        )
        host = cfg.hbm_model["ir_model"]["host"]
    except Exception as e:  # noqa: B902
        logger.warning("cannot read board host from predict.py: %s", e)
        return None
    if not host or any(c.isalpha() for c in str(host)):
        logger.warning("board host looks like a placeholder: %s", host)
        return None
    return str(host)


def build_hbm(config, march, args):
    """Export + compile a march-specific hbm. Returns (task_name, hbm_path)."""
    cfg = load_config_with_optional_base(
        resolve_config_file(config, default_filename="deploy.py")
    )
    if args.opts:
        cfg.merge_from_list_or_dict(args.opts, overwrite=args.opts_overwrite)
    cfg.merge_from_list_or_dict(
        ["hbir_exporter.march", march, "hbir_compiler.march", march],
        overwrite=True,
    )

    exporter = cfg.hbir_exporter
    compiler = cfg.hbir_compiler
    task = cfg.get("task_name", None) or exporter.get("model_name", "model")
    # deploy.py does not carry with_check, but HbirExporter requires it --
    # export_hbir.py fills it in on the tool side too (export_hbir.py:73).
    exporter["with_check"] = exporter.get("with_check", False)

    exporter["save_path"] = os.path.abspath(exporter["save_path"])
    compiler["model_path"] = os.path.abspath(compiler["model_path"])
    compile_dir = os.path.dirname(os.path.abspath(compiler["out_path"]))
    hbm_path = os.path.join(compile_dir, "model_%s.hbm" % march)
    compiler["out_path"] = hbm_path

    if args.skip_compile:
        if not os.path.exists(hbm_path):
            raise FileNotFoundError(
                "--skip-compile but %s does not exist" % hbm_path
            )
        logger.info("[%s] reuse %s", march, hbm_path)
        return task, hbm_path

    logger.info("[%s] exporting hbir -> %s", march, compiler["model_path"])
    with RegistryContext():
        hbir_exporter = build_from_registry(exporter)
        hbir_exporter(False)

    json_dir = os.path.join(exporter["save_path"], "perf_json", march)
    os.makedirs(json_dir, exist_ok=True)
    logger.info("[%s] compiling -> %s (report: %s)", march, hbm_path, json_dir)
    cwd = os.getcwd()
    os.chdir(json_dir)
    try:
        with RegistryContext():
            hbir_compiler = build_from_registry(compiler)
            hbir_compiler()
    finally:
        os.chdir(cwd)

    if args.save_json:
        save_report(json_dir, task, march)
    return task, hbm_path


def _build_worker(config, march, args, queue):
    """Child-process entry for :func:`build_hbm_isolated`."""
    logging.basicConfig(
        format="%(asctime)-15s %(levelname)s %(message)s",
        level=logging.INFO,
    )
    try:
        queue.put(("ok", build_hbm(config, march, args)))
    except Exception:  # noqa: B902
        queue.put(("err", traceback.format_exc()))


def build_hbm_isolated(config, march, args):
    """Compile one march in its own process.

    hbdk4 keeps the default target as process-global state -- exporting a
    second march in the same process dies with "error: resetting default
    target currently not allowed", so every march gets a fresh interpreter.
    """
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_build_worker, args=(config, march, args, queue))
    proc.start()
    proc.join()
    if queue.empty():
        raise RuntimeError(
            "compile process for %s died with exit code %s"
            % (march, proc.exitcode)
        )
    status, payload = queue.get()
    if status == "err":
        raise RuntimeError("compile for %s failed:\n%s" % (march, payload))
    return payload


def save_report(json_dir, task, march):
    """Copy the compile report out, renamed by march so nothing overwrites.

    hbm_perf writes ``<model>.json`` + ``<model>.html`` into the CWD, next to
    hbdk's own ``model-codegen-info.json`` -- only the html/json pair sharing
    a stem is the perf report we want.
    """
    dst_dir = os.path.join(REPO_ROOT, "perf_results", task)
    os.makedirs(dst_dir, exist_ok=True)
    names = os.listdir(json_dir)
    stems = {os.path.splitext(n)[0] for n in names if n.endswith(".html")}
    for name in sorted(names):
        stem, ext = os.path.splitext(name)
        if stem not in stems or ext not in (".json", ".html"):
            continue
        dst = os.path.join(dst_dir, "%s_%s%s" % (task, march, ext))
        shutil.copyfile(os.path.join(json_dir, name), dst)
        logger.info("[%s] report saved: %s", march, dst)


# ----- board measurement: direct hrt_model_exec + monitor (no sh) -----------


def parse_profiler_log(text):
    """Parse ``profiler.log`` JSON into fps / latency / bpu / ucp.

    The file is ``ucp_version: ...`` on the first line, then two JSON blocks
    separated by ``***``: block 0 holds ``perf_result`` (fps, latency), block
    1 holds ``BPU_inference_time_cost`` / ``TaskRunningTime`` /
    ``CPU_inference_time_cost`` (bpu, ucp). Returns a dict; missing fields are
    None. ucp = TaskRunningTime - BPU - CPU (clamped to >= 0), i.e. the sh
    ``get_ucp`` logic.
    """
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
        # Block 1 nests the per-stage times under separate objects:
        #   processor_latency.{BPU,CPU}_inference_time_cost,
        #   task_latency.TaskRunningTime.
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
    """Return peak HBMEM (MiB) of ``proc_name`` over the monitor log.

    Each sampled frame has a ``Process Mem Info`` block with a row per process:
    ``| <pid>  <proc>  <RSS>  <HBMEM> |`` where RSS/HBMEM carry a K/M/G suffix
    (MiB-ish). HBMEM is the BPU model memory == peak DDR. We take the max
    across all frames. Returns None if no matching row was seen.
    """
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


def _ssh(host, args):
    """Return a (exec, upload) pair bound to the board creds."""
    from hbm_infer.utils import remote_execute, remote_upload

    creds = (host, args.username, args.password, args.ssh_port)

    def exec_cmd(cmd, attempts=5):
        return remote_execute(cmd, *creds, max_attempts=attempts)

    def upload(local, remote):
        remote_upload(local, remote, *creds)

    return exec_cmd, upload


def _upload_hbm(exec_cmd, upload, hbm_path, board_dir, task, march, host):
    """Scp the hbm to <board_dir>/<task>/<march>/. Returns remote hbm path."""
    model_name = "%s_%s" % (task, march)
    remote_dir = posixpath.join(board_dir, task, march)
    exec_cmd("mkdir -p %s" % remote_dir)
    remote_hbm = posixpath.join(remote_dir, model_name + ".hbm")
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
    """Run one hrt_model_exec perf case on the board.

    ``case`` is ``fps`` or ``latency`` (selects thread/core). When
    ``monitor_bin`` is given (latency only), the run is wrapped in a
    ``hrt_ucp_monitor`` sampling window (started before, stopped after).

    Returns the raw stdout of the ssh command (mainly a trailing marker);
    the caller reads profiler.log / monitor.log back separately.
    """
    c = CASES[case]
    mkdir = "mkdir -p %s" % prof_dir
    # Absolute path: a `cd ... && nohup ... &` chain has a shell precedence
    # trap where `&` backgrounds the whole `&&` chain, so the `cd` lands in a
    # subshell and the foreground command after `&` runs in the original CWD
    # and cannot find `./hrt_model_exec`. Using the absolute binary path
    # sidesteps
    # CWD entirely, and the monitor branch is `;`-joined (not `&&`) so the `&`
    # only backgrounds the nohup'd monitor.
    hrt_bin_abs = posixpath.join(hrt_bin, "hrt_model_exec")
    hrt_cmd = (
        "%s perf --model_file=%s --thread_num=%d --core_id=%d --perf_time=%s "
        "--profile_path=%s"
        % (
            hrt_bin_abs,
            hbm,
            c["thread_num"],
            c["core_id"],
            perf_time,
            prof_dir,
        )
    )
    hrt_log = posixpath.join(prof_dir, "%s.log" % case)
    if monitor_bin:
        # Wrap: start monitor in the background, run the single-thread perf,
        # then wait for monitor to finish so the sampling window covers the
        # whole run. NO_MONITOR short-circuits to keep the exit code 0 while
        # still letting the perf run proceed (so latency is still measured).
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
    result_dir = brd["result_dir_full"]
    prof_dir = posixpath.join(result_dir, case)
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
    """Single-threaded perf under monitor sampling -> latency + peak DDR.

    Returns dict(latency, peak_ddr_mib). latency comes from profiler.log,
    peak DDR from the monitor log's max HBMEM. If the monitor binary is
    missing the run still produces latency but DDR is None.
    """
    result_dir = brd["result_dir_full"]
    prof_dir = posixpath.join(result_dir, "memory")
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


def run_one_march(config, march, host, args, brd):
    """Compile for one march and measure it on the board.

    ``brd`` is this march's resolved board settings (see test_perf_all).
    """
    row = {"march": march, "status": "ok"}
    if args.skip_compile:
        task, hbm_path = build_hbm(config, march, args)
    else:
        task, hbm_path = build_hbm_isolated(config, march, args)
    row["task"] = task
    row["hbm"] = hbm_path
    if args.no_board or host is None:
        row["status"] = "compiled (no board)"
        return row

    exec_cmd, upload = _ssh(host, args)
    remote_hbm = _upload_hbm(
        exec_cmd, upload, hbm_path, brd["board_dir"], task, march, host
    )
    exec_cmd("mkdir -p %s" % brd["result_dir_full"])

    measure = args.measure
    monitor_time = args.monitor_time or str(int(float(args.perf_time) + 2.5))
    if measure == "fps_latency":
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
    elif measure == "memory":
        mem = run_board_memory(
            exec_cmd,
            remote_hbm,
            brd,
            args.perf_time,
            monitor_time,
            march,
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
            exec_cmd,
            remote_hbm,
            brd,
            args.perf_time,
            monitor_time,
            march,
        )
        row.update(
            fps=fps["fps"],
            latency=mem["latency"],
            bpu=mem["bpu"],
            ucp=mem["ucp"],
            peak_ddr_mib=mem["peak_ddr_mib"],
        )
    return row


# ----- output -------------------------------------------------------------


def fmt(value):
    """Render a measurement, keeping '-' for anything that failed to parse."""
    return "-" if value is None else "%.2f" % value


def print_table(rows):
    """Print the per-march summary the readme tables are filled from."""
    head = (
        "march",
        "FPS",
        "latency(ms)",
        "bpu(ms)",
        "ucp(ms)",
        "DDR(MiB)",
        "status",
    )
    widths = [max(len(head[i]), 12) for i in range(len(head))]
    line = "  ".join(h.ljust(w) for h, w in zip(head, widths))
    print("\n" + line)
    print("-" * len(line))
    for r in rows:
        cells = (
            r["march"],
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
    """Append the summary rows to the output csv, creating it if needed."""
    if not rows:
        return
    out = args.out or os.path.join(
        REPO_ROOT, "perf_results", task, "perf_all.csv"
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    exists = os.path.exists(out)
    with open(out, "a") as f:
        if not exists:
            f.write(
                "task,march,fps,latency_ms,bpu_ms,ucp_ms,peak_ddr_mib,status\n"
            )
        for r in rows:
            f.write(
                "%s,%s,%s,%s,%s,%s,%s,%s\n"
                % (
                    r.get("task", task),
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


# ----- entry --------------------------------------------------------------


def test_perf_all(args):
    """Run compile + board measurement for every requested march."""
    if args.fps_thread_num is not None:
        CASES["fps"]["thread_num"] = args.fps_thread_num
    cfg = load_config_with_optional_base(
        resolve_config_file(args.config, default_filename="deploy.py")
    )
    task = cfg.get("task_name", None) or cfg.hbir_exporter.get(
        "model_name", "model"
    )
    if args.march:
        marches = [m.strip() for m in args.march.split(",") if m.strip()]
    else:
        marches = [str(cfg.hbir_compiler["march"])]
    for march in marches:
        if march not in VALID_MARCHES:
            raise ValueError(
                "unknown march %r, expected one of %s"
                % (march, ", ".join(VALID_MARCHES))
            )

    board_map = parse_march_map(args.board)
    cfg_host = None if args.no_board else resolve_board_host(args.config)
    if not args.no_board and not board_map and cfg_host is None:
        logger.warning(
            "no usable board ip, pass --board explicitly; "
            "only compiling this time"
        )

    hrt_bin_map = parse_march_map(args.hrt_bin)
    hrt_lib_map = parse_march_map(args.hrt_lib)
    board_dir_map = parse_march_map(args.board_dir)
    result_dir_map = parse_march_map(args.result_dir)
    monitor_map = parse_march_map(args.monitor_bin)

    def host_for(march):
        if args.no_board:
            return None
        return board_map.get(march) or board_map.get(None) or cfg_host

    def brd_for(march):
        hrt_bin = pick(hrt_bin_map, march, args.hrt_bin)
        return {
            "hrt_bin": hrt_bin,
            "hrt_lib": pick(hrt_lib_map, march, args.hrt_lib)
            or posixpath.join(hrt_bin, "lib"),
            "board_dir": pick(board_dir_map, march, args.board_dir),
            "result_dir": pick(result_dir_map, march, args.result_dir),
            "result_dir_full": posixpath.join(
                pick(result_dir_map, march, args.result_dir),
                "%s_%s" % (task, march),
            ),
            "monitor_bin": pick(monitor_map, march, args.monitor_bin)
            or DEFAULT_MONITOR_BIN.get(march),
        }

    rows = []
    for march in marches:
        logger.info("=" * 30 + " march %s " % march + "=" * 30)
        try:
            rows.append(
                run_one_march(
                    args.config, march, host_for(march), args, brd_for(march)
                )
            )
        except Exception:  # noqa: B902
            logger.error("[%s] failed:\n%s", march, traceback.format_exc())
            rows.append({"march": march, "task": task, "status": "ERROR"})
    print_table(rows)
    write_out(rows, args, task)
    return rows


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)-15s %(levelname)s %(message)s",
        level=logging.INFO,
    )
    known, args_env = parse_args()
    if args_env:
        setup_args_env(args_env)
    test_perf_all(known)
