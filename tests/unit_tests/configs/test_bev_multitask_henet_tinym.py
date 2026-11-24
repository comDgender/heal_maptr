# Copyright (c) Beijing Horizon Information Technology Co., Ltd. All rights reserved.  # noqa: E501

from pathlib import Path

from heal.utils.config import load_config_with_optional_base

CONFIG_DIR = (
    Path(__file__).resolve().parents[3] / "configs/bev_multitask_henet_tinym"
)


def _load_config(filename: str):
    return load_config_with_optional_base(str(CONFIG_DIR / filename))


def _input_source_count(value) -> int:
    """Count input-source entries whether given as a list or a comma string."""
    if isinstance(value, str):
        return len([s for s in value.split(",") if s.strip()])
    return len(value)


def test_bev_multitask_common_task_name():
    cfg = _load_config("common.py")
    assert cfg.task_name == "bev_multitask_henet_tinym"
    assert cfg.num_classes_occ == 18
    # Named semantic classes (free/empty is an extra class index).
    assert len(cfg.occ3d_seg_class) == 17
    assert cfg.camera_input
    assert not cfg.lidar_input
    assert cfg.use_dynamic_det_head
    assert cfg.use_static_objects_det_head
    assert cfg.use_occ_head
    assert cfg.use_maptr_head
    assert cfg.use_discrete_map_head


def test_bev_multitask_train_stages():
    cfg = _load_config("train.py")
    for name in (
        "float_stage1_trainer",
        "float_stage2_trainer",
        "float_stage3_trainer",
        "calibration_trainer",
    ):
        assert hasattr(cfg, name), f"missing {name}"
    # No single float_trainer / QAT stage.
    assert not hasattr(cfg, "float_trainer")
    assert not hasattr(cfg, "qat_trainer")


def test_bev_multitask_predict_stages():
    cfg = _load_config("predict.py")
    for name in (
        "float_stage1_predictor",
        "float_stage2_predictor",
        "float_stage3_predictor",
        "float_predictor",
        "calibration_predictor",
        "hbir_predictor",
        "hbm_predictor",
    ):
        assert hasattr(cfg, name), f"missing {name}"
    assert not hasattr(cfg, "qat_predictor")
    # Alias points at stage3.
    assert cfg.float_predictor is cfg.float_stage3_predictor


def test_bev_multitask_infer_stages():
    cfg = _load_config("infer.py")
    for name in (
        "gen_demo_cfg",
        "float_infer",
        "calibration_infer",
        "hbir_infer",
        "hbm_infer",
    ):
        assert hasattr(cfg, name), f"missing {name}"
    assert not hasattr(cfg, "qat_infer")
    assert cfg.gen_demo_cfg["data_saver"] is not None


def test_bev_multitask_deploy_artifacts():
    cfg = _load_config("deploy.py")
    assert hasattr(cfg, "hbir_exporter")
    assert hasattr(cfg, "hbir_compiler")
    assert hasattr(cfg, "calops_cfg")
    assert cfg.hbir_exporter["march"] == cfg.march
    assert cfg.hbir_compiler["march"] == cfg.march
    # deploy_inputs: img + projection_mat + 4 VT grids + 4 memory banks = 10.
    assert _input_source_count(cfg.hbir_compiler["input_source"]) == 10
    assert cfg.hbir_compiler["input_source"][0] == "pyramid"
    assert cfg.hbir_compiler["model_path"].endswith("qat.bc")


def test_bev_multitask_model_ir_infer_nv12_contract():
    cfg = _load_config("model.py")
    assert len(cfg.deploy_inputs) == 10
    assert cfg.hbir_infer_model["use_nv12"] is False
    assert cfg.hbm_infer_model["use_nv12"] is True
    assert "occ" in cfg.output_names


def test_bev_multitask_qconfig_calibration_only_pipeline():
    cfg = _load_config("qconfig.py")
    for name in (
        "calibration_train_model_convert_pipeline",
        "calibration_predict_model_convert_pipeline",
        "deploy_model_convert_pipeline",
        "cali_qconfig_setter",
    ):
        assert hasattr(cfg, name), f"missing {name}"
    assert not hasattr(cfg, "qat_train_model_convert_pipeline")
    converters = cfg.deploy_model_convert_pipeline["converters"]
    assert any(c["type"] == "Float2QAT" for c in converters)


def test_bev_multitask_hbm_data_loader_strips_yuv_normalize():
    cfg = _load_config("data.py")

    def _collect_types(node, found):
        if isinstance(node, dict):
            if "type" in node:
                found.add(node["type"])
            for value in node.values():
                _collect_types(value, found)
        elif isinstance(node, list):
            for value in node:
                _collect_types(value, found)

    hbm_types: set[str] = set()
    _collect_types(cfg.hbm_data_loader["dataset"], hbm_types)
    assert "BgrToYuv444" not in hbm_types
    assert "Normalize" not in hbm_types

    val_types: set[str] = set()
    _collect_types(cfg.val_data_loader["dataset"], val_types)
    assert "BgrToYuv444" in val_types
    assert "Normalize" in val_types
