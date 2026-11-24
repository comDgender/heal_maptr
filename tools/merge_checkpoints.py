"""Merge weights from multiple training stages into one checkpoint."""

import argparse
import io
import logging
import os
from collections import OrderedDict
from typing import Dict, List

import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Generic checkpoint merger")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="merged_ema_ckpt.pth.tar",
        help="Path to save the merged checkpoint",
    )
    parser.add_argument(
        "--ckpts",
        nargs="+",
        required=True,
        help="Checkpoint paths or URLs, applied in order (later overwrites earlier)",  # noqa: E501
    )
    parser.add_argument(
        "--prefixes",
        nargs="+",
        required=True,
        help=(
            "Comma-separated key prefixes per checkpoint; use 'all' to keep all keys. "  # noqa: E501
            "Example: 'head.det,necks.det' 'all'"
        ),
    )
    return parser.parse_args()


def load_checkpoint(path_or_url: str, map_location="cpu") -> Dict:
    logger.info("Loading: %s", path_or_url)
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        import requests

        resp = requests.get(path_or_url, stream=True)
        resp.raise_for_status()
        ckpt = torch.load(
            io.BytesIO(resp.content),
            map_location=map_location,
            weights_only=False,
        )
    else:
        if not os.path.exists(path_or_url):
            raise FileNotFoundError(f"Path not found: {path_or_url}")
        ckpt = torch.load(
            path_or_url, map_location=map_location, weights_only=False
        )
    return ckpt


def extract_state_dict(ckpt: Dict) -> OrderedDict:
    if not isinstance(ckpt, dict):
        return ckpt
    for key in ("ema_model", "state_dict", "model"):
        if key in ckpt:
            return ckpt[key]
    return ckpt


def filter_ckpt(
    state_dict: Dict, prefixes: List[str], idx: int
) -> OrderedDict:
    new_state_dict = OrderedDict()
    if hasattr(state_dict, "_metadata"):
        new_state_dict._metadata = OrderedDict()

    keep_all = len(prefixes) == 0 or "all" in prefixes
    keep_count = 0
    for key, value in state_dict.items():
        if keep_all or any(key.startswith(prefix) for prefix in prefixes):
            new_state_dict[key] = value
            keep_count += 1

    if hasattr(state_dict, "_metadata"):
        for key, value in state_dict._metadata.items():
            if keep_all or any(key.startswith(prefix) for prefix in prefixes):
                new_state_dict._metadata[key] = value

    logger.info(
        "  -> [Source %s] kept %s keys, filters=%s",
        idx,
        keep_count,
        prefixes if not keep_all else "ALL",
    )
    return new_state_dict


def main():
    args = parse_args()
    if len(args.ckpts) != len(args.prefixes):
        raise ValueError(
            f"Got {len(args.ckpts)} checkpoints but {len(args.prefixes)} prefix groups"  # noqa: E501
        )

    final_ckpt = OrderedDict()
    final_ckpt["state_dict"] = OrderedDict()
    final_ckpt["state_dict"]._metadata = OrderedDict()
    loaded_keys = {}

    for idx, (ckpt_path, prefix_str) in enumerate(
        zip(args.ckpts, args.prefixes)
    ):
        if prefix_str.lower() in ("all", ""):
            current_prefixes = []
        else:
            current_prefixes = [
                prefix.strip()
                for prefix in prefix_str.split(",")
                if prefix.strip()
            ]

        weights = extract_state_dict(load_checkpoint(ckpt_path))
        filtered_weights = filter_ckpt(weights, current_prefixes, idx)
        for key, value in filtered_weights.items():
            if key in final_ckpt["state_dict"]:
                prev_src = loaded_keys.get(key, "Unknown")
                logger.warning(
                    "  [CONFLICT] Key '%s' overwrites previous value from source %s",  # noqa: E501
                    key,
                    prev_src,
                )
            final_ckpt["state_dict"][key] = value
            loaded_keys[key] = idx

        if hasattr(filtered_weights, "_metadata"):
            final_ckpt["state_dict"]._metadata.update(
                filtered_weights._metadata
            )

    logger.info("Saving to %s", args.output)
    torch.save(final_ckpt, args.output)
    logger.info("Done.")


if __name__ == "__main__":
    main()
