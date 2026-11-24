import argparse
import logging
import os

import horizon_plugin_pytorch as horizon
from hat_engine.registry import RegistryContext, build_from_registry
from hat_engine.utils.hbdk4.hbir_compiler import HbirCompiler
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
        required=False,
        help="Config file path, or config directory (uses deploy.py)",
    )
    parser.add_argument(
        "--march",
        "-m",
        type=str,
        default=None,
        help="BPU march override, e.g. nash-b / nash-m / nash-p.",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=None,
        help="Override HBM output path.",
    )
    parser.add_argument(
        "--compiler-name",
        type=str,
        default=None,
        help="When cfg.hbir_compilers is a dict, only run this entry.",
    )
    known_args, unknown_args = parser.parse_known_args()
    return known_args, unknown_args


def apply_march_paths(hbir_compiler, march, out_path=None):
    """Point compiler at march-specific export/compile artifacts."""
    hbir_compiler["march"] = march
    orig_model = hbir_compiler["model_path"]
    march_dir = os.path.join(
        os.path.dirname(orig_model), march.replace("-", "_")
    )
    hbir_compiler["model_path"] = os.path.join(
        march_dir, os.path.basename(orig_model)
    )
    if out_path is not None:
        hbir_compiler["out_path"] = out_path
    else:
        hbir_compiler["out_path"] = os.path.join(
            march_dir, "compile", os.path.basename(hbir_compiler["out_path"])
        )
    cache = hbir_compiler.get("cache_path") or ""
    if cache:
        hbir_compiler["cache_path"] = os.path.join(
            march_dir, "compile", "cache"
        )


def _collect_compilers(cfg, compiler_name=None):
    has_single = hasattr(cfg, "hbir_compiler")
    has_dict = hasattr(cfg, "hbir_compilers")
    if has_single and has_dict:
        raise ValueError(
            "cfg defines both `hbir_compiler` and `hbir_compilers`; "
            "choose one."
        )
    if has_dict:
        entries = cfg.hbir_compilers
        if compiler_name is not None:
            return [(compiler_name, entries[compiler_name])]
        return list(entries.items())
    return [(None, cfg.hbir_compiler)]


def compile_perf_hbir(
    cfg_file,
    march=None,
    out_path=None,
    compiler_name=None,
):
    """Build and run one-or-many ``HbirCompiler`` entries."""
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)-15s %(levelname)s %(message)s",
        level=logging.INFO,
    )
    logger.info("=" * 50 + "BEGIN HBIR COMPILE" + "=" * 50)

    cfg = load_config_with_optional_base(
        resolve_config_file(cfg_file, default_filename="deploy.py")
    )

    for name, hbir_compiler in _collect_compilers(cfg, compiler_name):
        if march is not None:
            apply_march_paths(hbir_compiler, march, out_path=out_path)
            logger.info("override march: %s", march)
        elif out_path is not None:
            hbir_compiler["out_path"] = out_path

        if name is not None:
            logger.info(f"Compile entry: {name}")
        logger.info("Compile config:")
        logger.info(horizon.utils.misc.tensor_struct_repr(hbir_compiler))
        with RegistryContext():
            built: HbirCompiler = build_from_registry(hbir_compiler)
            built()

    # A partial compile would link the other entries' stale hbo.
    if compiler_name is None and hasattr(cfg, "hbm_linker"):
        logger.info("Link config:")
        logger.info(horizon.utils.misc.tensor_struct_repr(cfg.hbm_linker))
        with RegistryContext():
            build_from_registry(cfg.hbm_linker)()

    logger.info("=" * 50 + "END HBIR COMPILE" + "=" * 50)


if __name__ == "__main__":
    args, args_env = parse_args()
    if args_env:
        setup_args_env(args_env)
    compile_perf_hbir(
        cfg_file=args.config,
        march=args.march,
        out_path=args.out_path,
        compiler_name=args.compiler_name,
    )
