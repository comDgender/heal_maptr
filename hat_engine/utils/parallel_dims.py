# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This file has been modified by Beijing Horizon Information Technology Co., Ltd. on 2026/6/9.
# Provides ParallelDims for managing model parallelism degrees
# and building device mesh.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE-BSD-style file.


from dataclasses import dataclass
from distutils.version import LooseVersion
from functools import cached_property

import torch

try:
    from torch.distributed.device_mesh import init_device_mesh
except ImportError:
    init_device_mesh = None

from hat_engine.utils.logger import logging
from hat_engine.utils.package_helper import require_packages

logger = logging.getLogger(__name__)
torch_version = torch.__version__


@dataclass
class ParallelDims:
    """
    ParallelDims is a dataclass to store the parallelism degree of the model.

    The parallelism degree is defined as the number of degaree that are used
    to parallelize the model.

    Args:
        dp_replicate: The number of degaree used to replicate the model.
        dp_shard: The number of degaree used to shard the model.
        cp: The number of degaree used to parallelize the context.
        tp: The number of degaree used to parallelize the tensor.
        pp: The number of degaree used to parallelize the pipeline.
        world_size: The total number of processer used to
            parallelize the model.
        enable_loss_parallel: Whether to enable loss parallelization.
    """

    dp_replicate: int
    dp_shard: int
    cp: int
    tp: int
    pp: int
    world_size: int
    enable_loss_parallel: bool

    def __post_init__(self):
        self._validate()

    def _validate(self):
        dp_replicate, dp_shard, cp, tp, pp = (
            self.dp_replicate,
            self.dp_shard,
            self.cp,
            self.tp,
            self.pp,
        )
        for d in (dp_replicate, cp, tp, pp):
            assert (
                d >= 1
            ), "Parallelism degree should be >= 1, except for dp_shard"
        assert dp_shard == -1 or dp_shard >= 1, " dp_shard must -1 or >=1."

        dp = dp_replicate * dp_shard
        if dp < 0:
            dp = self.world_size // (cp * tp * pp)
            self.dp_shard = dp_shard = dp // dp_replicate

        assert dp_replicate >= 1
        assert dp_shard >= 1
        assert cp >= 1, cp
        assert tp >= 1, tp
        assert pp >= 1, pp
        assert dp_replicate * dp_shard * cp * tp * pp == self.world_size, (
            f"Invalid parallel dims: dp_replicate({dp_replicate})"
            f" * dp_shard({dp_shard}) * "
            f"cp({cp}) * tp({tp}) * pp({pp}) != WORLD_SIZE({self.world_size})"
        )

    @require_packages("torch>=2.2")
    def build_mesh(self, device_type):
        dims = []
        names = []
        for d, name in zip(
            [self.pp, self.dp_replicate, self.dp_shard, self.cp, self.tp],
            ["pp", "dp_replicate", "dp_shard", "cp", "tp"],
        ):
            if d > 1:
                dims.append(d)
                if (name == "dp_replicate" and self.dp_shard == 1) or (
                    name == "dp_shard" and self.dp_replicate == 1
                ):
                    names.append("dp")
                else:
                    names.append(name)

        logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
        names = tuple(names)
        mesh = init_device_mesh(device_type, dims, mesh_dim_names=names)
        # Create all the submesh here to ensure all required process groups are
        # initialized
        if LooseVersion(torch_version) >= LooseVersion("2.5.0"):
            if self.dp_replicate > 1 and self.dp_shard > 1:
                mesh["dp_replicate", "dp_shard"]._flatten(mesh_dim_name="dp")

            if self.cp > 1:
                if self.dp_replicate > 1 and self.dp_shard > 1:
                    mesh["dp_replicate", "dp_shard", "cp"]._flatten(
                        mesh_dim_name="dp_cp"
                    )
                else:
                    mesh["dp", "cp"]._flatten(mesh_dim_name="dp_cp")

        return mesh

    @property
    def dp_enabled(self):
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def dp_replicate_enabled(self):
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self):
        return self.dp_shard > 1

    @property
    def cp_enabled(self):
        return self.cp > 1

    @property
    def tp_enabled(self):
        return self.tp > 1

    @property
    def pp_enabled(self):
        return self.pp > 1

    @property
    def loss_parallel_enabled(self):
        return self.tp > 1 and self.enable_loss_parallel

    @cached_property
    def non_data_parallel_size(self):
        return self.cp * self.tp * self.pp
