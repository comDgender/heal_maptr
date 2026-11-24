# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import os
import subprocess
import sys

_BOUNDARY_CHILD = r"""
import sys

# Mask optional navsim stacks so any accidental import fails loudly.
for _name in ("navsim", "navsim_v2", "nuplan"):
    sys.modules[_name] = None

import heal  # noqa: F401

leaked = [
    m for m in sys.modules
    if m.startswith("heal.data.datasets.navsim")
    or m.startswith("heal.metrics.navsim")
    or m.startswith("heal.visualize.navsim_vis")
]
assert not leaked, f"navsim modules leaked into sys.modules: {leaked}"

from hat_engine.registry import OBJECT_REGISTRY

for name in (
    "NavsimDataset",
    "NavsimPerceptionMultitaskDataset",
    "NavsimE2EDataset",
    "NavsimMetric",
    "NavsimPerceptionMetric",
):
    assert name not in OBJECT_REGISTRY, (
        f"{name} registered despite navsim skip_dirs"
    )
"""


def test_import_heal_skips_navsim_modules():
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(sys.path),
    }
    subprocess.run(
        [sys.executable, "-c", _BOUNDARY_CHILD],
        env=env,
        check=True,
    )
