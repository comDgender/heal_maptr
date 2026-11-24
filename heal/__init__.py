# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
import os

from hat_engine.registry import register_default_config

from heal import version

__version__ = version.__version__

# Private numba backends are imported on demand (rotate IoU / KITTI eval).
# Walking them here would call cuda.is_available() during every train job.
# Navsim / nuplan packages are optional; import them explicitly from configs
# that need those registry types (e.g. e2e_planner).
register_default_config(
    path=os.path.dirname(__file__),
    skip_files=(
        "_rotate_box_utils_v2_numba.py",
        "_kitti3d_detection_numba.py",
    ),
    skip_dirs=(
        "navsim",
        "navsim_vis",
        "nuplan_utils",
    ),
)
