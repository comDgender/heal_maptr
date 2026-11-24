# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import argparse
import logging
import os
import pickle

import numpy as np
import torch
from hat_engine.data.transforms.compose_transform import Compose
from hat_engine.registry import build_from_registry
from hat_engine.utils.logger import init_logger
from PIL import Image

import heal  # noqa: F401
from heal.data.datasets.nuscenes.nuscenes_dataset import (
    CAM_NAMES,
    NuscenesBevDataset,
)
from heal.visualize.nuscenes import NuscenesViz


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize packed NuScenes BEV dataset."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Packed LMDB path (e.g. tmp_data/nuscenes/val_lmdb).",
    )
    parser.add_argument(
        "--viz-num",
        type=int,
        default=1,
        help="Max number of samples to visualize.",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Directory to save visualization PNGs.",
    )
    parser.add_argument(
        "--bev-size",
        type=float,
        nargs=3,
        default=[51.2, 51.2, 0.8],
        metavar=("H", "W", "RES"),
        help="BEV range for optional BEV-panel drawing.",
    )
    parser.add_argument(
        "--max-dist",
        type=float,
        default=55.0,
        help="Max GT box distance in lidar frame (meters).",
    )
    parser.add_argument(
        "--with-transforms",
        action="store_true",
        help="Apply val-like resize/crop + PILToTensor (default: raw RGB).",
    )
    return parser.parse_args()


def _format_gt_bboxes(bboxes):
    """Convert dataset GT ``(N, 10)`` to viz format ``(N, 11)``."""
    if bboxes is None or len(bboxes) == 0:
        return np.zeros((0, 11), dtype=np.float32)
    bboxes = np.asarray(bboxes, dtype=np.float32)
    if bboxes.shape[1] == 11:
        formatted = bboxes.copy()
    else:
        scores = np.ones((bboxes.shape[0], 1), dtype=np.float32)
        labels = bboxes[:, 9:10]
        formatted = np.concatenate([bboxes[:, :9], scores, labels], axis=1)
    # get_lidar_bboxes stores [length, width, h]; bbox_to_corner uses [w, l, h].  # noqa: E501
    formatted[:, [3, 4]] = formatted[:, [4, 3]]
    return formatted.astype(np.float32)


def _prepare_vis_img(img):
    if isinstance(img, torch.Tensor):
        image = img.detach().cpu().numpy()
    elif isinstance(img, Image.Image):
        image = np.array(img)
    else:
        image = np.asarray(img)
    image = image.squeeze()
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 3 and image.shape[2] == 3:
        image = image[..., ::-1].copy()
    return image.astype(np.uint8)


def _prepare_lidar2img(lidar2img):
    if isinstance(lidar2img, (list, tuple)):
        return [np.asarray(homo, dtype=np.float64) for homo in lidar2img]
    return np.asarray(lidar2img, dtype=np.float64)


def _build_dataset(data_path, max_dist, with_transforms):
    transforms = None
    if with_transforms:
        transforms = Compose(
            build_from_registry(
                [
                    {"type": "MultiViewsImgResize", "size": (396, 704)},
                    {"type": "MultiViewsImgCrop", "size": (256, 704)},
                    {
                        "type": "MultiViewsImgTransformWrapper",
                        "transforms": [{"type": "PILToTensor"}],
                    },
                ]
            )
        )
    return NuscenesBevDataset(
        data_path=data_path,
        max_dist=max_dist,
        transforms=transforms,
    )


if __name__ == "__main__":
    args = parse_args()
    init_logger(".hat_logs/nuscenes_viz", level=logging.INFO)

    dataset = _build_dataset(
        args.data_path, args.max_dist, args.with_transforms
    )
    dataset = pickle.loads(pickle.dumps(dataset))
    viz = NuscenesViz(
        bev_size=tuple(args.bev_size),
        score_thresh=0.0,
        is_plot=False,
    )

    num_saved = 0
    for i, data in enumerate(dataset):
        imgs = []
        for img, name in zip(data["img"], CAM_NAMES):
            imgs.append({"name": name, "img": _prepare_vis_img(img)})

        bboxes = _format_gt_bboxes(data["lidar_bboxes_labels"])
        preds = {"lidar_det": [bboxes]}
        meta = {"lidar2img": _prepare_lidar2img(data["lidar2img"])}
        viz(imgs, preds, meta, save_path=args.save_path)
        num_saved += 1
        if i >= int(args.viz_num) - 1:
            break

    if args.save_path is not None and num_saved > 0:
        abs_dir = os.path.abspath(args.save_path)
        print(f"Saved {num_saved} visualization(s) to {abs_dir}")
    elif num_saved == 0:
        print("No samples found in dataset; no images written.")
