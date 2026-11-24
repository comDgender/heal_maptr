# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import io
import logging
import os
from dataclasses import dataclass
from typing import Any, List, Optional

import matplotlib

matplotlib.use("Agg")  # force headless

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# --- ImageIO (optional) ---
try:
    import imageio.v2 as imageio
except ImportError:
    try:
        import imageio
    except ImportError:
        logger.warning(
            "imageio not found. Please install: pip install imageio"
        )
        imageio = None


def _load_navsim_viz():
    """Lazy-import navsim / navsim_v2 BEV helpers."""
    try:
        from navsim.visualization.bev import add_configured_bev_on_ax
        from navsim.visualization.config import BEV_PLOT_CONFIG
    except ImportError:
        try:
            from navsim_v2.visualization.bev import add_configured_bev_on_ax
            from navsim_v2.visualization.config import BEV_PLOT_CONFIG
        except ImportError as exc:
            raise ImportError(
                "navsim visualization modules are required for "
                "visualize_stage2_gif"
            ) from exc
    return add_configured_bev_on_ax, BEV_PLOT_CONFIG


def _configure_bev_ax(ax):
    """Best-effort BEV axis setup; navsim packages differ across versions."""
    try:
        from navsim.visualization.bev import configure_bev_ax

        return configure_bev_ax(ax)
    except ImportError:
        try:
            from navsim_v2.visualization.bev import configure_bev_ax

            return configure_bev_ax(ax)
        except ImportError:
            ax.set_aspect("equal")
            ax.axis("off")
            return ax


# ==========================================
# 1. Adapter classes (RecogDrive / NAVSIM V2)
# ==========================================


@dataclass
class MockEgoStatus:
    """Mock EgoStatus that must contain ego_pose."""

    ego_pose: np.ndarray  # [x, y, heading]
    ego_velocity: Optional[np.ndarray] = None
    ego_acceleration: Optional[np.ndarray] = None
    driving_command: Any = None


@dataclass
class MockFrame:
    """Minimal frame for BEV plotting."""

    ego_status: Any
    annotations: Any


# ==========================================
# 2. Core visualization
# ==========================================


def visualize_stage2_gif(
    token: str,
    map_api: Any,
    ego_traj_global: np.ndarray,
    agent_tracks: List[Any],
    save_dir: str = "./debug_vis_stage2",
    fps: int = 10,
    max_frames: int = 60,
):
    """
    Generate a Stage-2 GIF.

    Args:
        ego_traj_global: (T, 3) numpy array [x, y, heading] in global coords.
    """
    if imageio is None:
        return

    if map_api is None:
        logger.warning(
            "Skipping visualization for %s: map_api is None.", token
        )
        return

    add_configured_bev_on_ax, BEV_PLOT_CONFIG = _load_navsim_viz()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    frames_buffer = []

    total_frames = min(len(ego_traj_global), len(agent_tracks))
    step = max(1, total_frames // max_frames)

    for i in range(0, total_frames, step):
        current_x = float(ego_traj_global[i, 0])
        current_y = float(ego_traj_global[i, 1])
        current_h = float(ego_traj_global[i, 2])

        mock_ego_status = MockEgoStatus(
            ego_pose=np.array(
                [current_x, current_y, current_h], dtype=np.float64
            )
        )

        current_agents = agent_tracks[i]
        if hasattr(current_agents, "tracked_objects"):
            annotations = current_agents.tracked_objects
        else:
            annotations = current_agents

        mock_frame = MockFrame(
            ego_status=mock_ego_status,
            annotations=annotations,
        )

        fig, ax = plt.subplots(
            1, 1, figsize=BEV_PLOT_CONFIG.get("figure_size", (10, 10))
        )

        add_configured_bev_on_ax(ax, map_api, mock_frame)
        _configure_bev_ax(ax)

        ax.set_title(f"{token} | T={i * 0.1:.1f}s", fontsize=10, color="blue")

        buf = io.BytesIO()
        fig.savefig(
            buf, format="png", bbox_inches="tight", pad_inches=0, dpi=100
        )
        buf.seek(0)
        frames_buffer.append(imageio.imread(buf))
        buf.close()
        plt.close(fig)

    if not frames_buffer:
        return

    out_path = os.path.join(save_dir, f"{token}_stage2.gif")
    imageio.mimsave(out_path, frames_buffer, fps=fps)
    logger.info("Saved stage2 gif to %s", out_path)
