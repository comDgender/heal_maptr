# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
# Export hbir from QAT model, and convert QAT hbir to Quantized hbir.
# Config must define either ``hbir_exporter`` (single) or
# ``hbir_exporters`` (dict of named entries); both together is an error.

import argparse
import logging
import os

import horizon_plugin_pytorch as horizon
from hat_engine.registry import RegistryContext, build_from_registry
from hat_engine.utils.hbdk4.hbir_exporter import HbirExporter
from hat_engine.utils.setup_env import setup_args_env

import heal  # noqa: F401
from heal.utils.config import (
    load_config_with_optional_base,
    resolve_config_file,
)


def parse_args():
    """Parse CLI arguments and passthrough env options."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Config file path, or config directory (uses deploy.py)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        required=False,
        help="Where to save exported hbir files.",
    )
    parser.add_argument(
        "--with-check",
        action="store_true",
        help="Whether check the mlir model forward with example input.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Save hbir for debug."
    )
    parser.add_argument(
        "--march",
        "-m",
        type=str,
        default=None,
        help="BPU march override, e.g. nash-b / nash-m / nash-p.",
    )
    parser.add_argument(
        "--exporter-name",
        type=str,
        default=None,
        help="When cfg.hbir_exporters is a dict, only run this entry.",
    )
    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args


def _collect_exporters(cfg, exporter_name=None):
    has_single = hasattr(cfg, "hbir_exporter")
    has_dict = hasattr(cfg, "hbir_exporters")
    if has_single and has_dict:
        raise ValueError(
            "cfg defines both `hbir_exporter` and `hbir_exporters`; "
            "choose one."
        )
    if has_dict:
        entries = cfg.hbir_exporters
        if exporter_name is not None:
            return [(exporter_name, entries[exporter_name])]
        return list(entries.items())
    return [(None, cfg.hbir_exporter)]


def export_hbir(
    config_file_path,
    save_path=None,
    with_check=False,
    debug=False,
    march=None,
    exporter_name=None,
):
    """Build and run one-or-many ``HbirExporter`` entries."""
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)-15s %(levelname)s %(message)s",
        level=logging.DEBUG if debug else logging.INFO,
    )

    cfg = load_config_with_optional_base(
        resolve_config_file(config_file_path, default_filename="deploy.py")
    )

    logger.info("=" * 50 + "BEGIN EXPORTING HBIR" + "=" * 50)

    for name, hbir_exporter in _collect_exporters(cfg, exporter_name):
        if march is not None:
            hbir_exporter["march_modify"] = march
            logger.info("override march: %s", march)

        if save_path is not None:
            hbir_exporter["save_path"] = save_path
        elif march is not None:
            hbir_exporter["save_path"] = os.path.join(
                hbir_exporter["save_path"], march.replace("-", "_")
            )

        hbir_exporter["with_check"] = (
            hbir_exporter.get("with_check", False) or with_check
        )

        if name is not None:
            logger.info(f"Export entry: {name}")
        logger.info("Export config:")
        logger.info(horizon.utils.misc.tensor_struct_repr(hbir_exporter))

        with RegistryContext():
            built: HbirExporter = build_from_registry(hbir_exporter)
            built(debug)

    logger.info("=" * 50 + "END EXPORTING HBIR" + "=" * 50)


if __name__ == "__main__":
    args, args_env = parse_args()
    if args_env:
        setup_args_env(args_env)
    export_hbir(
        args.config,
        args.save_path,
        args.with_check,
        args.debug,
        march=args.march,
        exporter_name=args.exporter_name,
    )
