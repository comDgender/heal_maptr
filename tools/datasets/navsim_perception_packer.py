# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from __future__ import annotations
import argparse
import logging
import os
import time

from hat_engine.utils.logger import init_logger

from heal.data.datasets.navsim.navsim_perception_packer import (
    NavsimPerceptionPacker,
    save_token_to_idx_pkl,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pack navsim perception dataset."
    )
    parser.add_argument(
        "--src-data-dir",
        required=True,
        help="The directory that contains unpacked image files.",
    )
    parser.add_argument(
        "--pack-type",
        required=True,
        help="The pack data type for result of packer",
    )
    parser.add_argument(
        "--target-data-dir",
        default="",
        help="Output root. When packing all ranks, shards go to "
        "<target>/rank_<i>/; when --rank-id >= 0, write directly here.",
    )
    parser.add_argument(
        "--split-name", default="trainval", help="The split to pack."
    )
    parser.add_argument(
        "--filter-file-path",
        default="",
        help="YAML with train_log_names / val_log_names.",
    )
    parser.add_argument("--mode", default="train", help="train / val / test.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=20,
        help="The number of workers to load image.",
    )
    parser.add_argument(
        "--navsim-version",
        default="v1",
        help="navsim version, v1 or v2.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to pack (per rank).",
    )
    parser.add_argument(
        "--world_size",
        type=int,
        default=10,
        help="Number of log shards (default: 10, matches v4 train layout).",
    )
    parser.add_argument(
        "--rank_id",
        type=int,
        default=-1,
        help="Shard id. Default -1 = pack all ranks 0..world_size-1. "
        "Set >=0 to pack a single shard.",
    )
    parser.add_argument(
        "--pack-all-ranks",
        action="store_true",
        help="Deprecated alias: same as --rank-id -1.",
    )
    return parser.parse_args()


def _run_one_rank(args, directory, target_data_dir, world_size, rank_id):
    t0 = time.time()
    logger.info(
        "==== Pack start | mode=%s rank=%d/%d out=%s ====",
        args.mode,
        rank_id,
        world_size,
        target_data_dir,
    )
    packer = NavsimPerceptionPacker(
        src_data_dir=directory,
        target_data_dir=target_data_dir,
        split_name=args.split_name,
        num_workers=int(args.num_workers),
        pack_type=args.pack_type,
        filter_file_path=args.filter_file_path,
        mode=args.mode,
        navsim_version=args.navsim_version,
        max_samples=args.max_samples,
        world_size=world_size,
        rank_id=rank_id,
    )
    n = len(packer.dataset)
    logger.info(
        "Rank %d dataset ready: %d samples, %d logs, workers=%d, pack_type=%s",
        rank_id,
        n,
        len(getattr(packer.dataset, "log_files", []) or []),
        int(args.num_workers),
        args.pack_type,
    )
    if n == 0:
        logger.warning("Rank %d has 0 samples, skip packing.", rank_id)
        return
    packer()
    elapsed = time.time() - t0
    logger.info(
        "==== Pack done  | rank=%d/%d samples=%d elapsed=%.1fs out=%s ====",
        rank_id,
        world_size,
        n,
        elapsed,
        target_data_dir,
    )


if __name__ == "__main__":
    args = parse_args()

    init_logger(".hat_logs/navsim_packer")
    directory = os.path.expanduser(args.src_data_dir)

    if args.target_data_dir == "":
        args.target_data_dir = args.src_data_dir

    world_size = int(args.world_size)
    if world_size < 1:
        raise ValueError(f"--world_size must be >= 1, got {world_size}")

    pack_all = args.pack_all_ranks or int(args.rank_id) < 0
    logger.info(
        "Navsim perception packer | src=%s | target=%s | mode=%s | "
        "split=%s | filter=%s | navsim_version=%s | world_size=%d | "
        "rank_id=%s | pack_all=%s | max_samples=%s",
        directory,
        args.target_data_dir,
        args.mode,
        args.split_name,
        args.filter_file_path or "<none>",
        args.navsim_version,
        world_size,
        args.rank_id,
        pack_all,
        args.max_samples,
    )

    t_all = time.time()
    if pack_all:
        logger.info(
            "Packing all ranks sequentially: rank_0 .. rank_%d under %s",
            world_size - 1,
            args.target_data_dir,
        )
        rank_dirs = []
        for rank_id in range(world_size):
            rank_dir = os.path.join(args.target_data_dir, f"rank_{rank_id}")
            os.makedirs(rank_dir, exist_ok=True)
            _run_one_rank(args, directory, rank_dir, world_size, rank_id)
            rank_dirs.append(rank_dir)
        pkl_path = os.path.join(args.target_data_dir, "token_to_idx.pkl")
        save_token_to_idx_pkl(rank_dirs, pkl_path=pkl_path)
        logger.info(
            "All %d ranks finished in %.1fs -> %s (token_to_idx=%s)",
            world_size,
            time.time() - t_all,
            args.target_data_dir,
            pkl_path,
        )
    else:
        rank_id = int(args.rank_id)
        if rank_id >= world_size:
            raise ValueError(
                f"--rank-id {rank_id} out of range for world_size={world_size}"
            )
        os.makedirs(args.target_data_dir, exist_ok=True)
        _run_one_rank(
            args, directory, args.target_data_dir, world_size, rank_id
        )
        save_token_to_idx_pkl([args.target_data_dir])
        logger.info("Finished in %.1fs", time.time() - t_all)
