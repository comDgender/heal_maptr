# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import numpy as np
import torch
from hat_engine.registry import OBJECT_REGISTRY


@OBJECT_REGISTRY.register
class NavsimPerceptionCacheWrapper(torch.nn.Module):
    """Run perception on every history frame and flatten the results.

    For each batch item *b* and each history frame *f*, the perception model
    is called with a single-frame batch dict built from
    ``data["history_frames"][b][f]``.  The per-frame detection results are
    collected and returned as flat lists (one entry per frame per batch item).

    Returns a dict with:
        sample_tokens       – list[str]
        scenes              – list[str]
        dynamic_objects_det – list[dict]  (one per frame)
        static_objects_det  – list[dict]
        maptr               – list[dict]
        bev_embeds          – list[np.ndarray | None]
        dynamic_bev_embeds  – list[np.ndarray | None]
        occ_bev_embeds      – list[np.ndarray | None]
        maptr_bev_embeds    – list[np.ndarray | None]
    """

    def __init__(self, perception_model: torch.nn.Module):
        super().__init__()
        self.perception = perception_model

    # ------------------------------------------------------------------
    # Collation helpers (mirroring _run_perception_multiframe)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_frame_batch(
        history_frames,
        f_idx: int,
        B: int,
        data: dict,
    ) -> dict:
        """Build a single-frame batch dict from ``history_frames``."""
        frame_batch = {}
        for key in [
            "img",
            "lidar2img",
            "cam2lidar",
            "camera_intrinsic",
            "cam_name",
            "layout",
            "color_space",
            "timestamp",
            "sample_token",
            "scene",
            "ego2global",
            "lidar2global",
            "lidar2ego",
            "osm_mask",
            "lidar2ego_translation",
            "lidar2ego_rotation",
            "ego2global_translation",
            "ego2global_rotation",
        ]:
            if key not in history_frames[0][f_idx]:
                continue
            vals = [history_frames[b][f_idx][key] for b in range(B)]

            if key == "img":
                cam_imgs = []
                for b in range(B):
                    for cam_img in vals[b]:
                        cam_imgs.append(cam_img)
                frame_batch["img"] = torch.stack(cam_imgs, dim=0)
            elif key in ["lidar2img", "cam2lidar", "camera_intrinsic"]:
                items = []
                for b in range(B):
                    for item in vals[b]:
                        if isinstance(item, np.ndarray):
                            item = torch.tensor(item)
                        items.append(item)
                frame_batch[key] = torch.stack(items, dim=0)
            elif key == "osm_mask":
                items = []
                for b in range(B):
                    for item in vals[b]:
                        if isinstance(item, np.ndarray):
                            item = torch.tensor(item)
                        item = item.unsqueeze(0)
                        items.append(item)
                frame_batch[key] = torch.stack(items, dim=0)
            elif key in [
                "lidar2ego",
                "lidar2ego_translation",
                "lidar2ego_rotation",
                "ego2global_translation",
                "ego2global_rotation",
                "lidar2global",
            ]:
                items = []
                for b in range(B):
                    v = vals[b]
                    if isinstance(v, np.ndarray):
                        v = torch.tensor(v)
                    items.append(v)
                frame_batch[key] = torch.stack(items, dim=0)
            elif key == "cam_name":
                frame_batch[key] = [list(vals[b]) for b in range(B)]
            elif key == "timestamp":
                frame_batch[key] = torch.tensor(vals)
            else:
                frame_batch[key] = vals

        # Build seq_meta[0]
        img_meta = {}
        for meta_key in [
            "lidar2img",
            "ego2global",
            "lidar2global",
            "ego2global_translation",
            "ego2global_rotation",
            "lidar2ego",
            "sample_token",
            "scene",
        ]:
            mvals = []
            for b in range(B):
                f_data = history_frames[b][f_idx]
                mvals.append(f_data.get(meta_key))
            img_meta[meta_key] = mvals

        img_meta["timestamp"] = torch.tensor(
            [history_frames[b][f_idx].get("timestamp", 0) for b in range(B)]
        )

        for gt_key in [
            "gt_moving_objects_info",
            "gt_static_objects_info",
            "gt_discrete_map_info",
        ]:
            if gt_key in data:
                src = data[gt_key]
                if isinstance(src, dict):
                    expanded = {}
                    for k2, v2 in src.items():
                        if isinstance(v2, list) and len(v2) == B:
                            expanded[k2] = v2
                        else:
                            expanded[k2] = [v2] * B
                    img_meta[gt_key] = expanded
                else:
                    img_meta[gt_key] = src

        frame_batch["seq_meta"] = [img_meta]
        return frame_batch

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, data: dict) -> dict:
        history_frames = data.get("history_frames", None)
        num_history_list = data.get("num_history_frames", None)

        if "img" in data:
            B = data["img"].shape[0] // 8
        else:
            B = 1

        use_multiframe = (
            history_frames is not None
            and num_history_list is not None
            and len(history_frames) > 0
        )

        # Per-batch-item containers
        all_frame_dyn = [[] for _ in range(B)]
        all_frame_static = [[] for _ in range(B)]
        all_frame_occ = [[] for _ in range(B)]
        all_frame_maptr = [None] * B
        all_frame_bev = [[] for _ in range(B)]

        if not use_multiframe:
            # Single-frame fallback(TODO: code not verified)
            with torch.no_grad():
                perc_outs = self.perception(data)

            dyn = perc_outs["dynamic_objects_det"]
            static = perc_outs["static_objects_det"]
            maptr = perc_outs["maptr"]
            occ_info = perc_outs["occ"][0]
            bev_embed = perc_outs.get("bev_embed", None)

            for b in range(B):
                if isinstance(dyn, list) and b < len(dyn):
                    all_frame_dyn[b].append(dyn[b])
                else:
                    all_frame_dyn[b].append({"bboxes": torch.zeros(0, 10)})

                if isinstance(static, list) and b < len(static):
                    all_frame_static[b].append(static[b])
                else:
                    all_frame_static[b].append({"bboxes": torch.zeros(0, 10)})

                if isinstance(occ_info, dict):
                    cur_occ = {
                        "occ_pre": occ_info["occ_pre"][b]
                        .detach()
                        .cpu()
                        .numpy()
                        .copy(),
                        "occ_bev_embed": occ_info["occ_bev_embed"][b]
                        .detach()
                        .cpu()
                        .numpy()
                        .copy(),
                    }
                    all_frame_occ[b].append(cur_occ)

                if isinstance(maptr, list) and b < len(maptr):
                    all_frame_maptr[b] = maptr[b]
                else:
                    all_frame_maptr[b] = {}

                if bev_embed is not None and isinstance(
                    bev_embed, torch.Tensor
                ):
                    all_frame_bev[b].append(
                        bev_embed[b].detach().cpu().numpy().copy()
                    )
                else:
                    all_frame_bev[b].append(None)

            # For single-frame, history_frames is not available; use
            # top-level sample_token/scene
            sample_tokens = data.get("sample_token", [])
            scenes = data.get("scene", [])

            flat_sample_tokens = []
            flat_scenes = []
            for b in range(B):
                st = (
                    sample_tokens[b]
                    if isinstance(sample_tokens, list)
                    and b < len(sample_tokens)
                    else ""
                )
                sc = (
                    scenes[b]
                    if isinstance(scenes, list) and b < len(scenes)
                    else ""
                )
                flat_sample_tokens.append(st)
                flat_scenes.append(sc)

        else:
            # Multi-frame path
            n_hist = len(history_frames[0]) if len(history_frames) > 0 else 1

            for f_idx in range(n_hist):
                frame_batch = self._build_frame_batch(
                    history_frames,
                    f_idx,
                    B,
                    data,
                )

                with torch.no_grad():
                    perc_outs = self.perception(frame_batch)

                dyn = perc_outs["dynamic_objects_det"]
                static = perc_outs["static_objects_det"]
                maptr = perc_outs["maptr"]
                occ_info = perc_outs["occ"][0]
                bev_embed = perc_outs.get("bev_embed", None)

                for b in range(B):
                    if f_idx == n_hist - 1:
                        if isinstance(dyn, list) and b < len(dyn):
                            all_frame_dyn[b].append(dyn[b])
                        else:
                            all_frame_dyn[b].append(
                                {"bboxes": torch.zeros(0, 10)}
                            )

                        if isinstance(static, list) and b < len(static):
                            all_frame_static[b].append(static[b])
                        else:
                            all_frame_static[b].append(
                                {"bboxes": torch.zeros(0, 10)}
                            )

                        if isinstance(occ_info, dict):
                            cur_occ = {
                                "occ_pre": occ_info["occ_pre"][b]
                                .detach()
                                .cpu()
                                .numpy()
                                .copy(),
                                "occ_bev_embed": occ_info["occ_bev_embed"][b]
                                .detach()
                                .cpu()
                                .numpy()
                                .copy(),
                            }
                            all_frame_occ[b].append(cur_occ)
                        else:
                            cur_occ = {
                                "occ_pre": occ_info[b]
                                .detach()
                                .cpu()
                                .numpy()
                                .copy(),
                            }
                            all_frame_occ[b].append(cur_occ)

                        # bev_embed
                        if bev_embed is not None and isinstance(
                            bev_embed, torch.Tensor
                        ):
                            all_frame_bev[b].append(
                                bev_embed[b].detach().cpu().numpy().copy()
                            )
                        else:
                            all_frame_bev[b].append(None)

                    if f_idx == n_hist - 1:
                        if isinstance(maptr, list) and b < len(maptr):
                            all_frame_maptr[b] = maptr[b]
                        else:
                            all_frame_maptr[b] = {}

                del perc_outs, dyn, static, maptr, bev_embed

            # Flatten across batch and frames
            flat_sample_tokens = []
            flat_scenes = []
            for b in range(B):
                # only record the key frame
                flat_sample_tokens.append(
                    history_frames[b][n_hist - 1].get("sample_token", "")
                )
                flat_scenes.append(
                    history_frames[b][n_hist - 1].get("scene", "")
                )

        # ── Build flat result lists ────────────────────────────────────────
        flat_dyn = []
        flat_static = []
        flat_maptr = []
        flat_bev = []
        flat_occ = []

        for b in range(B):
            n_frames = len(all_frame_dyn[b])
            for f_idx in range(n_frames):
                flat_dyn.append(all_frame_dyn[b][f_idx])
                flat_static.append(all_frame_static[b][f_idx])
                flat_occ.append(all_frame_occ[b][f_idx])
                flat_bev.append(
                    all_frame_bev[b][f_idx]
                    if f_idx < len(all_frame_bev[b])
                    else None
                )
                flat_maptr.append(all_frame_maptr[b])

        return {
            "sample_tokens": flat_sample_tokens,
            "scenes": flat_scenes,
            "dynamic_objects_det": flat_dyn,
            "static_objects_det": flat_static,
            "occ": flat_occ,
            "maptr": flat_maptr,
            "bev_embeds": flat_bev,
        }
