# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

"""Dataset subpackages with lazy loading via ``__getattr__``."""

from importlib import import_module
from typing import Any, Tuple

# NOTE: Generative-model datasets (MMLU / MMStar / PPL / ...) depend on
# heavy optional packages such as ``opencompass`` / ``tabulate``. Do NOT
# register them in ``_LAZY_ATTRS`` -- attribute access on this module
# would then drag ``opencompass`` into ``sys.modules`` for every plain
# ``import heal.data.datasets``, breaking the perception-stack dependency
# boundary. Use the explicit path

__all__ = [
    "Cityscapes",
    "CITYSCAPES_LABLE_MAPPINGS",
    "MergeDataset",
    "NuscenesBevDataset",
    "NuscenesBevSequenceDataset",
    "NuscenesLidarDataset",
    "NuscenesMapDataset",
    "NuscenesMapGtDataset",
    "NuscenesMonoDataset",
    "NuscenesOccDataset",
    "NuscenesSparseMapDataset",
]

_LAZY_ATTRS: dict[str, Tuple[str, str]] = {
    "Cityscapes": ("heal.data.datasets.cityscapes", "Cityscapes"),
    "CITYSCAPES_LABLE_MAPPINGS": (
        "heal.data.datasets.cityscapes",
        "CITYSCAPES_LABLE_MAPPINGS",
    ),
    "MergeDataset": ("heal.data.datasets.merge_dataset", "MergeDataset"),
    "NuscenesBevDataset": (
        "heal.data.datasets.nuscenes.nuscenes_dataset",
        "NuscenesBevDataset",
    ),
    "NuscenesBevSequenceDataset": (
        "heal.data.datasets.nuscenes.nuscenes_dataset",
        "NuscenesBevSequenceDataset",
    ),
    "NuscenesLidarDataset": (
        "heal.data.datasets.nuscenes.nuscenes_lidar",
        "NuscenesLidarDataset",
    ),
    "NuscenesMapDataset": (
        "heal.data.datasets.nuscenes.nuscenes_map_dataset",
        "NuscenesMapDataset",
    ),
    "NuscenesMapGtDataset": (
        "heal.data.datasets.nuscenes.nuscenes_map_gt",
        "NuscenesMapGtDataset",
    ),
    "NuscenesMonoDataset": (
        "heal.data.datasets.nuscenes.nuscenes_dataset",
        "NuscenesMonoDataset",
    ),
    "NuscenesOccDataset": (
        "heal.data.datasets.nuscenes.nuscenes_occ",
        "NuscenesOccDataset",
    ),
    "NuscenesSparseMapDataset": (
        "heal.data.datasets.nuscenes.nuscenes_sparse_map_dataset",
        "NuscenesSparseMapDataset",
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_ATTRS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
