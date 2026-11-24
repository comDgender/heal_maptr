# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501
import inspect
import logging
import os
from typing import Any, Callable, Mapping, Sequence, Union

import torch
from hat_engine.utils.config import Config

logger = logging.getLogger(__name__)

ExampleInputsKey = Union[str, Sequence[str]]


def replace_config_refs(root: Any, old: Any, new: Any) -> None:
    """Replace identity references to ``old`` with ``new``.

    Walk nested dict/list containers.
    """
    if isinstance(root, dict):
        for key, value in root.items():
            if value is old:
                root[key] = new
            else:
                replace_config_refs(value, old, new)
    elif isinstance(root, list):
        for idx, value in enumerate(root):
            if value is old:
                root[idx] = new
            else:
                replace_config_refs(value, old, new)


def _tensor_structure_signature(value: Any) -> Any:
    """Return a comparable structure signature (shapes/dtypes).

    Does not include tensor data.
    """
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), str(value.dtype))
    if isinstance(value, Mapping):
        return {k: _tensor_structure_signature(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_tensor_structure_signature(v) for v in value]
    return value


def _resolve_example_inputs_builder(
    cfg: Config,
    inputs_key: str,
    builder_key: str | None,
) -> Callable[..., Any] | None:
    if builder_key is not None:
        builder = cfg.get(builder_key)
        if callable(builder):
            return builder

    for candidate in (
        f"build_{inputs_key}",
        "build_example_inputs",
        "get_example_inputs",
    ):
        builder = cfg.get(candidate)
        if callable(builder):
            return builder
    return None


def _sync_example_knobs_to_builder_globals(
    cfg: Config, builder: Callable[..., Any]
) -> None:
    """Sync ``--opts`` sizing knobs back to the builder's module globals.

    Config sizing knobs (``example_num_points``, ``example_num_voxels``, ...)
    are module-level names that builders read as a fallback when their matching
    parameter is left at its default — e.g.::

        example_num_points = 150000
        def build_example_inputs(num_points=None):
            n = int(example_num_points if num_points is None else num_points)
            ...

    ``--opts example_num_points X`` updates the ``cfg`` dict (copied from the
    module namespace at load time) but **not** the live module global the
    builder actually reads. Without this sync the builder keeps using the stale
    original value and the ``--opts`` knob is silently ignored. We mirror every
    ``cfg`` key whose name starts with ``example_`` back into the builder's
    ``__globals__`` so the builder's own fallback path picks up the new value,
    regardless of how its parameter is named.
    """
    builder_globals = getattr(builder, "__globals__", None)
    if builder_globals is None:
        return
    for key, value in cfg.items():
        if isinstance(key, str) and key.startswith("example_"):
            builder_globals[key] = value


def _call_example_inputs_builder(
    cfg: Config, builder: Callable[..., Any]
) -> Any:
    """Invoke builder with cfg keys that match its parameter names."""
    _sync_example_knobs_to_builder_globals(cfg, builder)
    try:
        parameters = inspect.signature(builder).parameters.values()
    except (TypeError, ValueError):
        return builder()

    kwargs = {}
    for param in parameters:
        if param.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            continue
        if param.name in cfg:
            kwargs[param.name] = cfg[param.name]

    if kwargs:
        return builder(**kwargs)
    return builder()


def refresh_example_inputs(
    cfg: Config,
    inputs_keys: ExampleInputsKey = (
        "example_inputs",
        "fx_example_inputs",
    ),
    builder_key: str | None = None,
) -> None:
    """Rebuild trace ``example_inputs`` dicts after ``--opts`` merge.

    Config modules usually bind example-input dicts into nested trainer /
    predictor entries at import time. After ``--opts`` updates sizing knobs
    (e.g. ``example_num_points``), this helper rebuilds the dict and replaces
    every identity reference in ``cfg._cfg_dict``.

    Rebuild uses the first callable found in this order:

    1. ``cfg[builder_key]`` when ``builder_key`` is set
    2. ``cfg[build_{inputs_key}]`` (e.g. ``build_example_inputs``)
    3. ``cfg[build_example_inputs]``
    4. ``cfg[get_example_inputs]`` (legacy)

    The builder is called with keyword arguments taken from ``cfg`` for each
    parameter name that exists in the config. If no parameter names match,
    the builder is called with no arguments.

    When no builder is found for an ``inputs_key``, that key is skipped.
    """
    if isinstance(inputs_keys, str):
        keys = (inputs_keys,)
    else:
        keys = tuple(inputs_keys)

    for inputs_key in keys:
        if inputs_key not in cfg:
            continue

        old_inputs = cfg.get(inputs_key)
        builder = _resolve_example_inputs_builder(cfg, inputs_key, builder_key)
        if builder is None:
            continue

        new_inputs = _call_example_inputs_builder(cfg, builder)
        if _tensor_structure_signature(
            old_inputs
        ) == _tensor_structure_signature(new_inputs):
            continue

        cfg[inputs_key] = new_inputs
        replace_config_refs(cfg._cfg_dict, old_inputs, new_inputs)
        logger.debug(
            "Refreshed %s via %s",
            inputs_key,
            getattr(builder, "__name__", repr(builder)),
        )


def _deep_merge_config_dict(
    base_cfg: dict, override_cfg: dict, overwrite: bool = False
) -> dict:
    """Merge nested config dictionaries recursively.

    Args:
        base_cfg: Base config dictionary.
        override_cfg: Config dictionary providing additional keys or values.
        overwrite: Whether to overwrite duplicated non-dict keys.

    Returns:
        dict: Merged config dictionary.
    """
    merged_cfg = dict(base_cfg)
    for key, value in override_cfg.items():
        if (
            key in merged_cfg
            and isinstance(merged_cfg[key], dict)
            and isinstance(value, dict)
        ):
            merged_cfg[key] = _deep_merge_config_dict(
                merged_cfg[key], value, overwrite=overwrite
            )
        else:
            if key in merged_cfg:
                if overwrite:
                    logger.warning(
                        f"Overwrite config key '{key}' with new value: {value}"
                    )
                    merged_cfg[key] = value
            else:
                merged_cfg[key] = value
    return merged_cfg


def resolve_config_file(
    cfg_path: str, default_filename: str = "infer.py"
) -> str:
    """Resolve config path.

    When ``cfg_path`` is a directory, append ``default_filename``.

    Args:
        cfg_path: Config file path or directory containing the config file.
        default_filename: Config file name inside a directory,
            e.g. ``infer.py``.

    Returns:
        str: Absolute path to the resolved config file.

    Raises:
        FileNotFoundError: If the resolved config file does not exist.
    """
    cfg_path = os.path.abspath(os.path.expanduser(cfg_path))
    if os.path.isdir(cfg_path):
        cfg_path = os.path.join(cfg_path, default_filename)
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    return cfg_path


def _fromfile_isolated(filename: str) -> Config:
    """Load a .py config with per-load ``sys.modules`` isolation.

    ``hat_engine.utils.config.Config.fromfile`` loads the target module via
    ``import_module`` and pops only that top-level module name from
    ``sys.modules`` afterwards. But config files use sibling imports
    (``from common import ...``, ``from data import ...``, ``from callback
    import ...``, ``from qconfig import ...``, ``from model import ...``),
    which register *those* module names too — and they are never cleaned up.
    Loading a second config whose siblings share those names then resolves
    the sibling imports to the *first* config's modules (cross-config
    pollution → ``ImportError: cannot import name X from 'callback'``).

    This wrapper records the config-directory modules present in
    ``sys.modules`` before loading, and after loading pops any sibling-module
    entries that live under the loaded config's directory — so each config
    load starts from a clean sibling namespace.
    """
    import sys

    cfg_file_abs = os.path.abspath(os.path.expanduser(filename))
    cfg_dir = os.path.dirname(cfg_file_abs)
    before = set(sys.modules.keys())
    try:
        cfg = Config.fromfile(filename)
    finally:
        # Pop sibling config modules that this load newly registered under the
        # config's own directory (e.g. common/data/callback/qconfig/model for
        # configs/<name>/train.py). Leave unrelated modules untouched.
        for name in list(sys.modules.keys()):
            if name in before:
                continue
            mod = sys.modules.get(name)
            mod_file = getattr(mod, "__file__", None) if mod else None
            if (
                mod_file
                and os.path.dirname(os.path.abspath(mod_file)) == cfg_dir
            ):
                sys.modules.pop(name, None)
    return cfg


def load_config_with_optional_base(cfg_file: str) -> Config:
    """Load a config file and merge sibling ``common.py`` when available.

    Args:
        cfg_file: Path to the target config file.

    Returns:
        Config: Loaded config object, optionally merged with ``common.py``.
    """
    cfg = _fromfile_isolated(cfg_file)

    if not cfg_file.endswith(".py"):
        return cfg

    cfg_dir = os.path.dirname(os.path.abspath(os.path.expanduser(cfg_file)))
    base_cfg_path = os.path.join(cfg_dir, "common.py")
    target_cfg_path = os.path.abspath(os.path.expanduser(cfg_file))

    if not os.path.isfile(base_cfg_path):
        return cfg
    if os.path.abspath(base_cfg_path) == target_cfg_path:
        return cfg

    base_cfg = _fromfile_isolated(base_cfg_path)
    merged_cfg_dict = _deep_merge_config_dict(
        cfg._cfg_dict, base_cfg._cfg_dict
    )
    return Config(merged_cfg_dict, filename=cfg.filename)
