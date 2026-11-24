# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

import json
import logging
import os
from typing import List, Optional

from hbdk4.compiler.apis import link
from hbdk4.compiler.hbm import Hbm, Hbo

from hat_engine.registry import OBJECT_REGISTRY
from hat_engine.utils.logger import MSGColor, format_msg

__all__ = ["HbmLinker"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class HbmLinker:
    """Link compiled hbo artifacts into one hbm and measure each graph.

    Args:
        hbo_paths: Filesystem paths of the ".hbo" artifacts to link.
        out_path: A filesystem path to save the linked hbm.
        desc: Description dict written into the hbm as json.
        remote_ip: IP address of the remote BPU board.
        remote_port: Port to login the remote BPU board.
        perf_output_dir: Perf result dir. Defaults beside out_path.
    """

    def __init__(
        self,
        hbo_paths: List[str],
        out_path: str,
        desc: Optional[dict] = None,
        remote_ip: Optional[str] = None,
        remote_port: int = 22,
        perf_output_dir: Optional[str] = None,
    ) -> None:
        if not out_path.endswith(".hbm"):
            raise ValueError(f'{out_path} must end with ".hbm"')

        self.hbo_paths = hbo_paths
        self.out_path = out_path
        self.desc = desc
        self.remote_ip = remote_ip
        self.remote_port = remote_port
        self.perf_output_dir = perf_output_dir or os.path.dirname(out_path)

        # hbdk-lld fails instead of creating the output directory.
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def _write_desc(self):
        tmp_path = self.out_path[: -len(".hbm")] + "_tmp.hbm"
        hbm = Hbm(self.out_path)
        hbm.staged_desc = json.dumps(self.desc)
        hbm.save_by_staged_info(tmp_path)
        os.replace(tmp_path, self.out_path)

    def __call__(self):
        link([Hbo(p) for p in self.hbo_paths], self.out_path)

        if self.desc is not None:
            self._write_desc()

        hbm = Hbm(self.out_path)
        for graph in hbm.graphs:
            graph.perf(
                output_dir=self.perf_output_dir,
                remote_ip=self.remote_ip,
                remote_port=self.remote_port,
                remote_work_root="/map/hbm_infer",
            )

        logger.info(
            format_msg("Linked model: %s" % self.out_path, MSGColor.GREEN)
        )
