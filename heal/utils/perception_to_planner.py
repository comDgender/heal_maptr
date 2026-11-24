# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from typing import Dict, List, Optional

import numpy as np
import torch


def _get_lane_gt_by_nms(pts_3d):
    from heal.data.datasets.navsim.navsim_perception_dataset_utils import (
        get_lane_gt_by_nms,
    )

    return get_lane_gt_by_nms(pts_3d)


def perception_dyn_to_planner_agents(
    dyn_results: List[Dict],
    agent_num: int = 32,
    model_time_len: int = 21,
    max_ped_bike: int = 10,
    return_perception_predict: bool = False,
) -> torch.Tensor:
    """Convert single-frame dynamic dets to ``neighbor_agents_past``."""
    device = dyn_results[0]["labels"].device
    TYPE_LOOKUP = torch.tensor(
        [
            [1, 0, 0],  # vehicle
            [0, 1, 0],  # pedestrian
            [0, 0, 1],  # bicycle
        ],
        dtype=torch.float32,
        device=device,
    )
    B = len(dyn_results)
    out = torch.zeros(
        B,
        agent_num,
        model_time_len,
        11,
        dtype=torch.float32,
        device=device,
    )
    if return_perception_predict:
        dyn_res = torch.zeros(
            B, agent_num, 10, dtype=torch.float32, device=device
        )
        dyn_labels = torch.zeros(
            B, agent_num, 1, dtype=torch.float32, device=device
        )
    for b, res in enumerate(dyn_results):
        bboxes = res["bboxes"]  # (N, 10)
        labels = res["labels"]  # (N,)
        if bboxes.numel() == 0:
            continue

        x = bboxes[:, 0]
        y = bboxes[:, 1]
        yaw = bboxes[:, 6]
        vx = bboxes[:, 7]
        vy = bboxes[:, 8]
        w = bboxes[:, 4]
        length = bboxes[:, 3]

        cur_state = torch.stack(
            [x, y, torch.cos(yaw), torch.sin(yaw), vx, vy, w, length],
            dim=-1,
        )

        agents_11 = torch.zeros(
            bboxes.shape[0],
            model_time_len,
            11,
            dtype=torch.float32,
            device=device,
        )
        agents_11[:, -1, :8] = cur_state
        agents_11[:, -1, 8:11] = TYPE_LOOKUP[labels][:, :]

        dist = torch.sum(agents_11[:, -1, :2] ** 2, dim=-1)
        sorted_idx = torch.argsort(dist)

        sorted_labels = labels[sorted_idx]
        pb_mask = sorted_labels != 0
        veh_mask = ~pb_mask
        pb_idx = sorted_idx[pb_mask]
        veh_idx = sorted_idx[veh_mask]

        if pb_idx.numel() + veh_idx.numel() <= agent_num:
            selected = sorted_idx[:agent_num]
        else:
            sel_pb = pb_idx[:max_ped_bike]
            remaining_pb = pb_idx[max_ped_bike:]
            selected = torch.cat([sel_pb, veh_idx], dim=0)
            remaining_slots = agent_num - selected.shape[0]
            if remaining_slots > 0 and remaining_pb.numel() > 0:
                fill = min(remaining_slots, remaining_pb.numel())
                selected = torch.cat(
                    [selected, remaining_pb[:fill]],
                    dim=0,
                )
            selected_dist = dist[selected]
            selected = selected[torch.argsort(selected_dist)][:agent_num]

        out[b, : selected.shape[0]] = agents_11[selected]

        if return_perception_predict:
            dyn_res[b, : selected.shape[0]] = bboxes[selected]
            dyn_labels[b, : selected.shape[0]] = labels[selected].unsqueeze(-1)
    if not return_perception_predict:
        return out
    else:
        perception_output = {
            "bboxes": dyn_res,
            "labels": dyn_labels,
        }
        return out, perception_output


def perception_static_to_planner_static(
    static_results: List[Dict],
    static_objects_num: int = 5,
    return_perception_predict: bool = False,
) -> torch.Tensor:
    """Convert static dets to planner ``static_objects``."""
    device = static_results[0]["bboxes"].device
    _STATIC_TYPE_LOOKUP = torch.tensor(
        [
            [1, 0, 0, 0],  # czone_sign
            [0, 1, 0, 0],  # barrier
            [0, 0, 1, 0],  # traffic_cone
            [0, 0, 0, 1],  # generic_object
        ],
        dtype=torch.float32,
        device=device,
    )
    B = len(static_results)
    out = torch.zeros(
        B,
        static_objects_num,
        10,
        dtype=torch.float32,
        device=device,
    )
    if return_perception_predict:
        static_res = torch.zeros(
            B,
            static_objects_num,
            10,
            dtype=torch.float32,
            device=device,
        )
        static_labels = torch.zeros(
            B,
            static_objects_num,
            1,
            dtype=torch.float32,
            device=device,
        )

    for b, res in enumerate(static_results):
        bboxes = res["bboxes"]
        labels = res["labels"]
        if bboxes.numel() == 0:
            continue

        x = bboxes[:, 0]
        y = bboxes[:, 1]
        yaw = bboxes[:, 6]
        w = bboxes[:, 3]
        length = bboxes[:, 4]

        cur_state = torch.stack(
            [x, y, torch.cos(yaw), torch.sin(yaw), length, w],
            dim=-1,
        )

        static_10 = torch.zeros(
            bboxes.shape[0],
            10,
            dtype=torch.float32,
            device=device,
        )
        static_10[:, :6] = cur_state
        static_10[:, 6:10] = _STATIC_TYPE_LOOKUP[labels]

        dist = torch.sum(static_10[:, :2] ** 2, dim=-1)
        sorted_idx = torch.argsort(dist)[:static_objects_num]
        out[b, : len(sorted_idx)] = static_10[sorted_idx]
        if return_perception_predict:
            static_res[b, : len(sorted_idx)] = bboxes[sorted_idx]
            static_labels[b, : len(sorted_idx)] = labels[sorted_idx].unsqueeze(
                -1
            )

    if not return_perception_predict:
        return out
    else:
        perception_output = {
            "bboxes": static_res,
            "labels": static_labels,
        }
        return out, perception_output


def perception_maptr_to_planner_lanes(
    maptr_results: List[Dict],
    lane_num: int = 70,
    lane_len: int = 20,
    min_length: Optional[float] = None,
    do_nms: bool = False,
) -> tuple:
    """Convert MapTR dets to planner ``lanes`` / speed-limit tensors."""
    B = len(maptr_results)
    lanes_out = torch.zeros(B, lane_num, lane_len, 12, dtype=torch.float32)
    speed_limit_out = torch.zeros(B, lane_num, 1, dtype=torch.float32)
    has_speed_limit_out = torch.zeros(B, lane_num, 1, dtype=torch.bool)

    for b, res in enumerate(maptr_results):
        pts_3d = res.get("pts_3d", None)

        if min_length is not None:
            lane_length = torch.sum(
                torch.linalg.norm(pts_3d[:, 1:] - pts_3d[:, :-1], dim=2),
                dim=1,
            )
            pts_3d = pts_3d[lane_length > min_length]
        if do_nms:
            pts_3d = _get_lane_gt_by_nms(pts_3d)

        if pts_3d is None or pts_3d.numel() == 0:
            continue

        if isinstance(pts_3d, np.ndarray):
            pts_3d = torch.tensor(pts_3d, dtype=torch.float32)

        N = pts_3d.shape[0]
        if N == 0:
            continue

        if pts_3d.shape[1] > lane_len:
            pts_3d = pts_3d[:, :lane_len, :]
        elif pts_3d.shape[1] < lane_len:
            pad = torch.zeros(
                N,
                lane_len - pts_3d.shape[1],
                2,
                dtype=torch.float32,
            )
            pts_3d = torch.cat([pts_3d, pad], dim=1)

        vectors = torch.zeros(N, lane_len, 2, dtype=torch.float32)
        vectors[:, :-1, :] = pts_3d[:, 1:, :] - pts_3d[:, :-1, :]

        dist_per_point = torch.sum(pts_3d[:, :, :2] ** 2, dim=-1)
        min_dist, _ = dist_per_point.min(dim=-1)
        sorted_idx = torch.argsort(min_dist)[:lane_num]
        if sorted_idx.numel() == 0:
            continue
        lanes_out[b, : len(sorted_idx), :, 0:2] = pts_3d[sorted_idx][:, :, 0:2]
        lanes_out[b, : len(sorted_idx), :, 2:4] = vectors[sorted_idx][
            :, :, 0:2
        ]

    return lanes_out, speed_limit_out, has_speed_limit_out
