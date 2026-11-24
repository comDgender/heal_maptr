import argparse
import ast
import inspect
import logging
from functools import partial

import horizon_plugin_pytorch as horizon
import torchvision
from hat_engine.models.ir_modules.hbm_module import close_hbm_servers
from hat_engine.registry import RegistryContext, build_from_registry
from hat_engine.utils.config import ParseAction
from hat_engine.utils.logger import init_logger

import heal  # noqa: F401
from heal.utils.config import (
    load_config_with_optional_base,
    refresh_example_inputs,
    resolve_config_file,
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse CLI arguments for stage-based inference."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        "-s",
        type=str,
        required=True,
        help=(
            "the inference stage, you should define "
            "{stage}_infer in your config"
        ),
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="inference config file path, or config directory (uses infer.py)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="override visualize_processor save path.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="max number of samples to run. Default: all samples.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="override data_path in stage infer config.",
    )
    parser.add_argument(
        "--device-ids",
        "-ids",
        type=str,
        required=False,
        default=None,
        help="GPU device ids like '0'"
        "will override `device_ids` in infer_cfg."
        "Current only support single device.",
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
        help="Whether to overwrite existing (keys, values) in configs",
    )
    return parser.parse_args()


def modify_infer_cfg(infer_cfg, args):
    """Apply CLI overrides to the selected stage inference config."""

    if args.data_path is not None:
        infer_cfg["data_path"] = args.data_path

    if "device" not in infer_cfg["infer"]:
        infer_cfg["infer"]["device"] = None

    if args.device_ids is not None:
        ids = list(map(int, args.device_ids.split(",")))
        infer_cfg["infer"]["device"] = ids

    if (
        args.save_path is not None
        and "visualize_processors" in infer_cfg["infer"]
    ):
        visualize_processors = infer_cfg["infer"]["visualize_processors"]
        if isinstance(visualize_processors, dict):
            visualize_processors["save_path"] = args.save_path
        elif isinstance(visualize_processors, list):
            for processor in visualize_processors:
                processor["save_path"] = args.save_path

    return infer_cfg


def build_transforms(transforms):
    """Build transform pipeline from config and compose if needed."""
    transforms = build_from_registry(transforms)
    if isinstance(transforms, list):
        return torchvision.transforms.Compose(transforms)
    return transforms


def bind_pre_processors(pre_processors, transforms):
    """Bind ``transforms`` to pre-processors that declare this parameter."""
    processors = pre_processors
    if not isinstance(processors, list):
        processors = [processors]

    bound_processors = []
    for processor in processors:
        signature = inspect.signature(processor)
        if "transforms" in signature.parameters:
            processor = partial(processor, transforms=transforms)
        bound_processors.append(processor)
    return bound_processors


if __name__ == "__main__":
    args = parse_args()

    cfg = load_config_with_optional_base(
        resolve_config_file(args.config, default_filename="infer.py")
    )
    if args.opts is not None:
        cfg.merge_from_list_or_dict(args.opts, overwrite=args.opts_overwrite)
    refresh_example_inputs(cfg)
    init_logger(f".hat_logs/{cfg.task_name}_{args.stage}_inference")
    horizon.march.set_march(cfg.get("march"))

    infer_key = f"{args.stage}_infer"
    assert hasattr(
        cfg, infer_key
    ), f"you should define {infer_key} in the config file"
    infer_cfg = cfg[infer_key]

    # modify infer_cfg with args
    infer_cfg = modify_infer_cfg(infer_cfg, args)

    data_loader = infer_cfg.get("data_loader", None)
    assert data_loader is not None, "data_loader is not defined in infer_cfg"
    # build data_loader
    data_loader = build_from_registry(data_loader)
    # bind pre_processors with transforms if needed
    transforms = infer_cfg.get("transforms", None)
    pre_processors = infer_cfg["infer"].get("pre_processors", None)

    if pre_processors is not None and transforms is not None:
        transforms = build_transforms(transforms)
        pre_processors = build_from_registry(pre_processors)
        pre_processors = bind_pre_processors(pre_processors, transforms)
        infer_cfg["infer"]["pre_processors"] = pre_processors

    infer = None
    try:
        with RegistryContext():
            infer = build_from_registry(infer_cfg["infer"])
            data_list = data_loader(infer_cfg["data_path"])
            if args.max_samples is not None:
                data_list = list(data_list)[: max(args.max_samples, 0)]
            for data in data_list:
                if isinstance(data, dict):
                    data["stage"] = args.stage
                infer.forward(data)
            viz_cfg = infer_cfg["infer"].get("visualize_processors")
            if isinstance(viz_cfg, dict) and viz_cfg.get("save_path"):
                logger.info(
                    "Visualization output directory: %s",
                    viz_cfg["save_path"],
                )
    finally:
        if infer is not None:
            close_hbm_servers(infer.model)
