"""Tests for heal package __init__."""

import subprocess
import sys


def test_heal_package_exposes_version():
    import heal

    assert hasattr(heal, "__version__")
    assert isinstance(heal.__version__, str)


def test_import_heal_does_not_load_numba_backends():
    code = (
        "import sys, heal; "
        "assert 'heal.core._rotate_box_utils_v2_numba' not in sys.modules; "
        "assert 'heal.metrics._kitti3d_detection_numba' not in sys.modules"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
