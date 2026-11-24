"""train tools."""

import argparse
import ast
import gc
import logging
import os
import pprint
import warnings
from typing import List, Sequence, Union

import hat_engine
import horizon_plugin_pytorch as horizon
import torch.backends.cudnn as cudnn
from hat_engine.engine import build_launcher
from hat_engine.registry import RegistryContext, build_from_registry
from hat_engine.utils.config import Config, ConfigVersion, ParseAction
from hat_engine.utils.deterministic import set_deterministic_level
from hat_engine.utils.distributed import get_dist_info, rank_zero_only
from hat_engine.utils.errors import ExceptionWithNodeInfo, record
from hat_engine.utils.logger import (
    LOG_DIR,
    DisableLogger,
    ExperimentLogger,
    MSGColor,
    format_msg,
    init_rank_logger,
    rank_zero_info,
)
from hat_engine.utils.package_check import package_check
from hat_engine.utils.seed import seed_training
from hat_engine.utils.setup_env import setup_args_env, setup_hat_env
from hat_engine.utils.thread_init import init_num_threads
from hat_engine.utils.timer import TimerContextManager
from hat_engine.utils.traceback_monitor import setup_hat_flight_record

import heal  # noqa: F401
from heal.utils.config import (
    load_config_with_optional_base,
    refresh_example_inputs,
    resolve_config_file,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    """Parse CLI arguments for stage-based training."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        "-s",
        type=str,
        required=True,
        help=(
            "the training stage, you should define "
            "{stage}_trainer in your config"
        ),
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="train config file path, or config directory (uses train.py)",
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
        "--launcher",
        type=str,
        choices=["torch", "mpi"],
        default=None,
        help="job launcher for multi machines",
    )
    parser.add_argument(
        "--pipeline-test",
        action="store_true",
        default=False,
        help="export HAT_PIPELINE_TEST=1, which used in config",
    )
    parser.add_argument(
        "--opts",
        help="modify config options using the command-line",
        default=None,
        nargs="+",
        action=ParseAction,
    )
    parser.add_argument(
        "--opts-overwrite",
        type=ast.literal_eval,
        default=True,
        help="True or False, default True, "
        "Weather to overwrite existing (keys, values) in configs",
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

    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args


def train_entrance(
    device: Union[None, int, Sequence[int]],
    cfg_file: str,
    cfg_opts: List,
    cfg_opts_overwrite: bool,
    stage: str,
    level: int = logging.WARNING,
):
    """Training entrance function for launcher.

    Args:
        device: run on cpu (if None), or gpu (list of gpu ids)
        cfg_file: Config file used to build log file name.
        cfg_opts: Custom config options from command-line.
        cfg_opts_overwrite: Weather to overwrite existing {k: v} in configs.
        stage: Current training stage used to build log file name.
        level: logging level for other rank except rank0.
    """

    with TimerContextManager("Config fromfile in rank"):
        if isinstance(cfg_file, Config):
            cfg = cfg_file
            cfg_file = cfg._filename
        else:
            cfg_file = resolve_config_file(
                cfg_file, default_filename="train.py"
            )
            # Avoid repeated fromfile causing
            # different behaviors in same process
            cfg = load_config_with_optional_base(cfg_file)
            if cfg_opts is not None:
                cfg.merge_from_list_or_dict(
                    cfg_opts, overwrite=cfg_opts_overwrite
                )
            refresh_example_inputs(cfg)

    rank, world_size = get_dist_info()
    disable_logger = rank != 0 and cfg.get("log_rank_zero_only", False)

    # 1. init logger
    logger = init_rank_logger(
        rank,
        save_dir=cfg.get("log_dir", LOG_DIR),
        cfg_file=cfg_file,
        step=stage,
        prefix="train-",
        filter_warning=cfg.get("filter_warning", False),
    )

    if disable_logger:
        logger.info(
            format_msg(
                f"Logger of rank {rank} has been disable, turn off "
                "`log_rank_zero_only` in config if you don't want this.",
                MSGColor.GREEN,
            )
        )
    with TimerContextManager("Config log or save"):
        if "redirect_config_logging_path" in cfg:
            if cfg["redirect_config_logging_path"] is not None and rank == 0:
                with open(cfg["redirect_config_logging_path"], "w") as fid:
                    fid.write(pprint.pformat(cfg._cfg_dict))
                rank_zero_info(
                    "save config logging output to %s"
                    % cfg["redirect_config_logging_path"]
                )
        else:
            rank_zero_info(pprint.pformat(cfg._cfg_dict))

    rank_zero_info("=" * 50 + "BEGIN %s STAGE" % stage.upper() + "=" * 50)

    # 2. init num threads
    init_num_threads()

    # 3. seed training
    set_deterministic_level(level=cfg.get("deterministic_level", None))
    cudnn.benchmark = cfg.cudnn_benchmark
    if cfg.seed is not None:
        train_seed = int(cfg.seed)
        if cfg.get("seed_with_rank", False):
            train_seed = train_seed + rank
        seed_training(train_seed)

    if "march" not in cfg:
        rank_zero_only(
            logger.warning(
                format_msg(
                    f"Please make sure the march is provided in configs. "
                    f"Defaultly use {horizon.march.March.BAYES}",
                    MSGColor.RED,
                )
            )
        )
    horizon.march.set_march(cfg.get("march", horizon.march.March.BAYES))

    # 4. start monitoring
    setup_hat_flight_record(rank=rank)

    # 5. build and run trainer
    with DisableLogger(disable_logger, level), RegistryContext():
        trainer = getattr(cfg, f"{stage}_trainer")
        trainer["device"] = device
        with TimerContextManager("Trainer init"):
            trainer = build_from_registry(trainer)

        # del cfg for saving memory
        del cfg
        gc.collect()

        trainer.fit()

    rank_zero_info("=" * 50 + "END %s STAGE" % stage.upper() + "=" * 50)


@record(record_prefix="train")
def train(
    stage: str,
    config: str,
    device_ids: str = None,
    dist_url: str = "auto",
    launcher: str = None,
    pipeline_test: bool = False,
    opts: list = None,
    opts_overwrite: bool = None,
    args_env: list = None,
    level: int = logging.WARNING,
    enable_tracking: bool = False,
):
    """Training  function.

    Args:
        stage: The training stage, you should define {stage}_trainer
               in your config.
        config: Config file path.
        device_ids: GPU device ids like '0,1,2,3', will override
                    `device_ids` in config.
        dist_url: Dist url for init process, such as tcp://localhost:8000.
        launcher: Job launcher for multi machines.
        pipeline_test: export HAT_PIPELINE_TEST=1, which used in config.
        opts: modify config options.
        opts_overwrite: True or False, default True, weather to overwrite
                        existing (keys, values) in configs.
        args_env: The args will be set in env.
        level: logging level for other rank except rank0.
        enable_tracking: Whether to enable aidi tracking.
    """
    if args_env:
        setup_args_env(args_env)
    setup_hat_env(
        stage,
        pipeline_test,
        enable_tracking,
    )

    package_check()

    with TimerContextManager("Config fromfile in main"):
        config = resolve_config_file(config, default_filename="train.py")
        config_info = load_config_with_optional_base(config)
        # check config version
        config_version = config_info.get("VERSION", None)
        if config_version is not None:
            assert (
                config_version == ConfigVersion.v2
            ), "{} only support config with version 2, not version {}".format(
                os.path.basename(__file__), config_version.value
            )
        else:
            warnings.warn(
                "VERSION will must set in config in the future."
                "You can refer to examples/classification/resnet18.py."
            )

        if opts is not None:
            config_info.merge_from_list_or_dict(opts, overwrite=opts_overwrite)
        refresh_example_inputs(config_info)

    os.environ["HAT_MONITOR_RANK_ZERO"] = str(
        int(config_info.get("log_rank_zero_only", False))
    )

    if device_ids is not None:
        ids = list(map(int, device_ids.split(",")))
    else:
        ids = config_info.device_ids

    assert hasattr(
        config_info, f"{stage}_trainer"
    ), f"There are not {stage}_trainer in config"
    trainer_config = getattr(config_info, f"{stage}_trainer")

    experiment_logger = ExperimentLogger(enable_tracking=enable_tracking)
    experiment_logger.init_group(f"training-group-{stage}")
    experiment_logger.log_config(config_info)

    launch = build_launcher(trainer_config)
    if launch is hat_engine.engine.ddp_trainer.launch and launcher is not None:
        config_info = config_info
    else:
        config_info = config

    launch(
        train_entrance,
        ids,
        dist_url=dist_url,
        dist_launcher=launcher,
        args=(
            config_info,
            opts,
            opts_overwrite,
            stage,
            level,
        ),
    )


if __name__ == "__main__":

    try:
        args, args_env = parse_args()
        train(
            stage=args.stage,
            config=args.config,
            device_ids=args.device_ids,
            dist_url=args.dist_url,
            launcher=args.launcher,
            pipeline_test=args.pipeline_test,
            opts=args.opts,
            opts_overwrite=args.opts_overwrite,
            level=args.level,
            args_env=args_env,
            enable_tracking=args.enable_tracking,
        )

    except Exception as e:
        ExperimentLogger.log_exception(
            exception=e,
            prefix="train failed",
            enable_tracking=args.enable_tracking,
        )
        raise ExceptionWithNodeInfo(e)
