import textwrap

import pytest
import torch
from hat_engine.utils.config import Config

from heal.utils.config import (
    _deep_merge_config_dict,
    load_config_with_optional_base,
    refresh_example_inputs,
    replace_config_refs,
    resolve_config_file,
)


def test_resolve_config_file_uses_infer_py_in_directory(tmp_path):
    infer_file = tmp_path / "infer.py"
    infer_file.write_text("task_name = 'demo'\n")

    resolved = resolve_config_file(str(tmp_path))

    assert resolved == str(infer_file.resolve())


def test_resolve_config_file_keeps_file_path(tmp_path):
    infer_file = tmp_path / "infer.py"
    infer_file.write_text("task_name = 'demo'\n")

    resolved = resolve_config_file(str(infer_file))

    assert resolved == str(infer_file.resolve())


def test_resolve_config_file_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_config_file(str(tmp_path))


def test_deep_merge_config_dict_keeps_base_value_by_default():
    base_cfg = {"a": 1, "nested": {"x": 1, "y": 2}}
    override_cfg = {"a": 9, "nested": {"x": 9, "z": 3}, "b": 2}

    merged = _deep_merge_config_dict(base_cfg, override_cfg)

    assert merged["a"] == 1
    assert merged["b"] == 2
    assert merged["nested"] == {"x": 1, "y": 2, "z": 3}


def test_deep_merge_config_dict_overwrites_when_enabled():
    base_cfg = {"a": 1, "nested": {"x": 1}}
    override_cfg = {"a": 2, "nested": {"x": 3}}

    merged = _deep_merge_config_dict(base_cfg, override_cfg, overwrite=True)

    assert merged["a"] == 2
    assert merged["nested"]["x"] == 3


def test_replace_config_refs_updates_nested_identity():
    old = {"points": [torch.randn(4, 4)]}
    new = {"points": [torch.randn(8, 4)]}
    root = {"trainer": {"model_convert": {"example_inputs": old}}}

    replace_config_refs(root, old, new)

    assert root["trainer"]["model_convert"]["example_inputs"] is new


def test_refresh_example_inputs_rebuilds_from_builder():
    old = {"points": [torch.randn(150000, 4)]}

    # Mirrors the production convention (pointpillars / centerpoint configs):
    # the builder leaves the sizing param at None and falls back to the module
    # global ``example_num_points``. refresh_example_inputs makes an
    # ``--opts example_num_points`` knob reach the builder by syncing the cfg
    # value back into the builder's __globals__ before invoking it, so the
    # builder's own fallback reads the new value. ``example_num_points`` below
    # is that synced global (not a local — resolved at call time from the
    # builder's __globals__).
    def build_example_inputs(num_points=None):
        # example_num_points is injected into this builder's __globals__ by
        # refresh_example_inputs before this runs.
        if num_points is None:
            num_points = example_num_points  # noqa: F821
        n = int(num_points)
        return {"points": [torch.randn(int(n), 4)]}

    cfg = Config(
        {
            "num_points": 20000,
            "example_inputs": old,
            "trainer": {"example_inputs": old},
            "build_example_inputs": build_example_inputs,
        }
    )

    refresh_example_inputs(cfg)

    assert cfg.example_inputs["points"][0].shape[0] == 20000
    assert cfg.trainer["example_inputs"] is cfg.example_inputs


def test_refresh_example_inputs_skips_when_structure_unchanged():
    old = {"points": [torch.randn(150000, 4)]}

    def build_example_inputs(num_points=150000):
        return {"points": [torch.randn(int(num_points), 4)]}

    cfg = Config(
        {
            "num_points": 150000,
            "example_inputs": old,
            "build_example_inputs": build_example_inputs,
        }
    )

    refresh_example_inputs(cfg)

    assert cfg.example_inputs is old


def test_load_config_with_optional_base_merges_common_py(tmp_path):
    common_file = tmp_path / "common.py"
    common_file.write_text(textwrap.dedent("""
            common_only = 100
            nested = dict(from_common=1, shared=10)
            """))

    target_file = tmp_path / "exp.py"
    target_file.write_text(textwrap.dedent("""
            exp_only = 200
            nested = dict(from_exp=2, shared=20)
            """))

    cfg = load_config_with_optional_base(str(target_file))

    # Target config has higher priority for duplicated keys.
    assert cfg.exp_only == 200
    assert cfg.common_only == 100
    assert cfg.nested["from_exp"] == 2
    assert cfg.nested["from_common"] == 1
    assert cfg.nested["shared"] == 20
