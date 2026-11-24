import argparse
import logging
import os

from hat_engine.registry import build_from_registry

import heal  # noqa: F401
from heal.utils.config import (
    load_config_with_optional_base,
    resolve_config_file,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args():
    """Parse CLI arguments for demo sample generation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="inference config file path, or config directory (uses infer.py)",
    )
    return parser.parse_args()


def main():
    """Build dataset from config and save selected demo samples."""
    args = parse_args()
    cfg = load_config_with_optional_base(
        resolve_config_file(args.config, default_filename="infer.py")
    )

    assert hasattr(
        cfg, "gen_demo_cfg"
    ), "you should define gen_demo_cfg in config file"

    gen_demo_cfg = cfg.gen_demo_cfg
    dataset = build_from_registry(gen_demo_cfg["dataset"])
    sample_idx = gen_demo_cfg.get("sample_idx", [0])
    if isinstance(sample_idx, int):
        sample_idx = [sample_idx]
    else:
        sample_idx = list(sample_idx)
    save_path = gen_demo_cfg["save_path"]

    data_saver = gen_demo_cfg.get("data_saver", None)
    assert (
        data_saver is not None
    ), "you should define data_saver in gen_demo_cfg"

    for idx in sample_idx:
        sample = dataset[idx]
        sample_path = os.path.join(save_path, f"sample_idx_{idx}")
        os.makedirs(sample_path, exist_ok=True)
        data_saver(sample, sample_path)
        logger.info(
            "save sample idx=%s to %s",
            idx,
            sample_path,
        )


if __name__ == "__main__":
    main()
