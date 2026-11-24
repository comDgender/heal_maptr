r"""Per-layer quantization sensitivity analysis entry point.

This tool expects the user's config (typically
``configs/<task>/quant_analysis.py``) to define a ``quant_analysis_solver``
dict with at least a ``type`` key (e.g. ``type="QuantAnalysis"``) plus the
usual ``model`` / ``dataloader`` / ``baseline_model_convert_pipeline`` /
``analysis_model_convert_pipeline`` fields. The solver dict is consumed by
``hat_engine.registry.build_from_registry`` after sibling ``common.py`` is
merged in by ``heal.utils.config.load_config_with_optional_base``.

Example:
    python3 tools/quant_analysis.py -c configs/qcnet_oe
    python3 tools/quant_analysis.py -c configs/qcnet_oe \\
        --opts quant_analysis_solver.num_steps 1
"""

import argparse
import ast
import copy
import logging
import os
from typing import List, Sequence, Union

import hat_engine.engine.ddp_trainer
import horizon_plugin_pytorch as horizon
import torch
from hat_engine.engine.launcher import get_launcher
from hat_engine.registry import RegistryContext, build_from_registry
from hat_engine.utils.config import Config
from hat_engine.utils.distributed import get_dist_info
from hat_engine.utils.logger import (
    LOG_DIR,
    DisableLogger,
    ExperimentLogger,
    MSGColor,
    format_msg,
    init_rank_logger,
    rank_zero_info,
    rank_zero_warn,
)
from hat_engine.utils.seed import seed_training
from hat_engine.utils.setup_env import setup_args_env, setup_hat_env

import heal  # noqa: F401
from heal.utils.config import (
    load_config_with_optional_base,
    resolve_config_file,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)-15s %(levelname)s %(message)s",
    level=logging.INFO,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="config file path",
    )
    parser.add_argument(
        "--device-ids",
        "-ids",
        type=str,
        required=False,
        default=None,
        help="GPU device ids like '0,1,2,3', "
        "will override `device_ids` in config",
    )
    parser.add_argument(
        "--dist-url",
        type=str,
        default="auto",
        help="dist url for init process, such as tcp://localhost:8000",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="NCCL",
        choices=["NCCL", "GLOO"],
        help="dist url for init process",
    )
    parser.add_argument(
        "--launcher",
        type=str,
        choices=["torch", "mpi"],
        default=None,
        help="job launcher for multi machines",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=logging.WARNING,
        help="Set the logging level for other rank except rank0",
    )
    parser.add_argument(
        "--enable-tracking",
        action="store_true",
        default=False,
        help="export HAT_ENABLE_MODEL_TRACKING=1, which enable aidi tracking",
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


def quant_analysis_entrance(
    device: Union[None, int, Sequence[int]],
    cfg_file: str,
    cfg_opts: List = None,
    cfg_opts_overwrite: bool = True,
):
    if isinstance(cfg_file, Config):
        cfg = cfg_file
        cfg_file = cfg._filename
    else:
        # Avoid repeated fromfile causing different behaviors in same process
        # Use HEAL's loader so sibling common.py (march, etc.) is merged in.
        cfg = load_config_with_optional_base(
            resolve_config_file(cfg_file, default_filename="quant_analysis.py")
        )
        if cfg_opts is not None:
            cfg.merge_from_list_or_dict(cfg_opts, overwrite=cfg_opts_overwrite)

    rank, _ = get_dist_info()
    disable_logger = rank != 0 and cfg.get("log_rank_zero_only", False)

    # 1. init logger
    logger = init_rank_logger(
        rank,
        save_dir=cfg.get("log_dir", LOG_DIR),
        cfg_file=cfg_file,
        step="analysis",
        prefix="quant-analysis-",
    )

    if disable_logger:
        logger.info(
            format_msg(
                f"Logger of rank {rank} has been disable, turn off "
                "`log_rank_zero_only` in config if you don't want this.",
                MSGColor.GREEN,
            )
        )

    torch.backends.cudnn.benchmark = cfg.get("cudnn_benchmark", False)
    if cfg.get("seed", None) is not None:
        seed_training(cfg.seed)

    quant_analysis_solver = cfg.quant_analysis_solver

    rank_zero_info("=" * 50 + "BEGIN QUANT ANALYSIS" + "=" * 50)

    if "march" not in cfg:
        rank_zero_warn(
            format_msg(
                f"Please make sure the march is provided in configs. "
                f"Defaultly use {horizon.march.March.BAYES}",
                MSGColor.RED,
            )
        )
    horizon.march.set_march(cfg.get("march", horizon.march.March.BAYES))

    if quant_analysis_solver.get("out_dir", None) is None:
        save_path = "."
        if "ckpt_dir" in cfg:
            save_path = cfg.ckpt_dir
        else:
            logger.warning(
                "No ckpt_dir found in config file. Analysis results will be "
                + "saved in ./quant_analysis"
            )
        save_path = os.path.join(save_path, "quant_analysis")
        os.makedirs(save_path, exist_ok=True)
        quant_analysis_solver["out_dir"] = save_path
    # Predict / train apply val_batch_processor (YUV / Normalize). The
    # analysis dataloader is the raw val loader; without those transforms
    # uint8 images hit conv as ByteTensor.
    if quant_analysis_solver.get("batch_transforms") is None:
        for pred_key in ("float_predictor", "calibration_predictor"):
            pred = cfg.get(pred_key)
            if not isinstance(pred, dict):
                continue
            proc = pred.get("batch_processor")
            if isinstance(proc, dict) and proc.get("batch_transforms"):
                quant_analysis_solver["batch_transforms"] = copy.deepcopy(
                    proc["batch_transforms"]
                )
                break

    with DisableLogger(disable_logger), RegistryContext():
        quant_analysis_solver["device_id"] = device
        quant_analysis = build_from_registry(quant_analysis_solver)
        quant_analysis()

    rank_zero_info("=" * 50 + "END QUANT ANALYSIS" + "=" * 50)


def quant_analysis(
    config: str,
    device_ids: str = None,
    dist_url: str = "auto",
    backend: str = "NCCL",
    launcher: str = None,
    args_env: list = None,
    enable_tracking: bool = False,
    opts: List = None,
    opts_overwrite: bool = True,
):
    if args_env:
        setup_args_env(args_env)
    stage = "quant_analysis"
    setup_hat_env(stage, enable_tracking=enable_tracking)

    config_info = load_config_with_optional_base(
        resolve_config_file(config, default_filename="quant_analysis.py")
    )
    if opts is not None:
        config_info.merge_from_list_or_dict(opts, overwrite=opts_overwrite)

    if device_ids is not None:
        ids = list(map(int, device_ids.split(",")))
    else:
        ids = config_info.device_ids
    str_ids = list(map(str, ids))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str_ids)
    num_processes = len(ids)
    if ids is None or ids == -1:
        assert backend == "GLOO", f"backend should be GLOO, but get {backend}"
        num_processes = 1

    quant_analysis_solver = config_info.get("quant_analysis_solver", None)
    if quant_analysis_solver is None:
        raise RuntimeError("Please set quant_analysis_solver in config file.")
    if not isinstance(quant_analysis_solver, dict):
        raise TypeError(
            "quant_analysis_solver must be a dict, "
            f"got {type(quant_analysis_solver).__name__}."
        )
    if "type" not in quant_analysis_solver:
        raise KeyError(
            "quant_analysis_solver is missing required 'type' key (e.g. "
            "type='QuantAnalysis'). See the docstring of tools/quant_analysis.py"  # noqa: E501
            " for the expected solver structure."
        )

    experiment_logger = ExperimentLogger(enable_tracking=enable_tracking)
    experiment_logger.init_group("quant_analysis")

    launch = get_launcher("distributed_data_parallel_trainer")
    if launch is hat_engine.engine.ddp_trainer.launch and launcher is not None:
        config_info = config_info
    else:
        config_info = config

    launch(
        quant_analysis_entrance,
        ids,
        dist_url=dist_url,
        dist_launcher=launcher,
        num_processes=num_processes,
        backend=backend,
        args=(config_info, opts, opts_overwrite),
    )


if __name__ == "__main__":
    try:
        args, args_env = parse_args()

        quant_analysis(
            config=args.config,
            device_ids=args.device_ids,
            dist_url=args.dist_url,
            backend=args.backend,
            launcher=args.launcher,
            args_env=args_env,
            enable_tracking=args.enable_tracking,
            opts=args.opts,
            opts_overwrite=args.opts_overwrite,
        )
    except Exception as e:
        ExperimentLogger.log_exception(
            exception=e,
            prefix="quant analysis failed",
            enable_tracking=args.enable_tracking,
        )
        raise e
